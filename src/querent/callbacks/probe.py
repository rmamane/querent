"""Deep probe on the fixed 512-image batch every N epochs.

Runs the UNCOMPILED ``pl_module.net`` (stats collection must never trigger
recompiles), chunked, under collect_stats="full" — modules take the math path
and export exact attention probs. Computes: per-layer attention entropy, mean
attention distance, effective rank of Q (participation ratio), routing entropy
+ position<->latent MI + load, gate position/content split, alpha per layer;
plus routing / positional-prior heatmap images for a few layers.

Everything logs against the ``epoch`` axis (see define_metric in train.py).
"""

from __future__ import annotations

import lightning.pytorch as pl
import torch

from ..data.probe import build_probe_batch
from ..diagnostics.attn_probs import resolve_attn_probs
from ..diagnostics.metrics import (
    attention_entropy,
    gate_position_content_split,
    grid_distance_matrix,
    mean_attention_distance,
    position_routing_mi,
    q_participation_ratios,
    routing_entropy,
)
from ..diagnostics.plots import fig_to_wandb, latent_grid_figure
from ..diagnostics.stats_api import iter_stat_modules, set_collect_stats

EPS = 1e-9


def _short(layer_name: str) -> str:
    return layer_name.replace("blocks.", "L").replace(".attn", "")


class ProbeCallback(pl.Callback):
    def __init__(self, cfg):
        d = cfg.diagnostics
        self.every = int(d.probe_every_epochs)
        self.size = int(d.probe_size)
        self.seed = int(d.probe_seed)
        self.chunk = int(d.probe_chunk)
        self.data_dir = str(cfg.paths.data_dir)
        self.x = None

    def _ensure_probe(self):
        if self.x is None:
            self.x, self.y = build_probe_batch(self.data_dir, n=self.size, seed=self.seed)

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.fast_dev_run:
            return
        ep = int(trainer.current_epoch)
        last = trainer.max_epochs is not None and ep == trainer.max_epochs - 1
        if ep % self.every != 0 and not last:
            return
        if trainer.logger is None or not hasattr(trainer.logger, "experiment"):
            return

        self._ensure_probe()
        net = pl_module.net  # ALWAYS the uncompiled module
        was_training = net.training
        net.eval()
        set_collect_stats(net, "full")

        grid = net.grid_size
        patch_px = float(net.patch_embed.proj.stride[0])
        dist = grid_distance_matrix(grid, patch_px=patch_px)

        agg: dict[str, dict] = {}
        n_chunks = 0
        for xb in self.x.split(self.chunk):
            net(xb.to(pl_module.device))
            n_chunks += 1
            for lname, mod in iter_stat_modules(net):
                s = mod.stats or {}
                d = agg.setdefault(lname, {"scalars": {}, "routing_sum": None, "gates": None, "q_rows": None})

                def add(key, val):
                    d["scalars"][key] = d["scalars"].get(key, 0.0) + float(val)

                probs = resolve_attn_probs(s)
                if probs is not None:
                    add("attn_entropy", attention_entropy(probs))
                    add("mean_attn_dist", mean_attention_distance(probs, dist))
                if "q" in s and d["q_rows"] is None:
                    d["q_rows"] = s["q"].cpu()  # first chunk: 128*N rows >> 4096, plenty
                for side in ("q", "k"):
                    if f"routing_{side}" in s:
                        a = s[f"routing_{side}"].float()
                        rs = a.sum(dim=0).cpu()  # [N, M]
                        d["routing_sum"] = rs if d["routing_sum"] is None else d["routing_sum"] + rs
                        d["routing_n"] = d.get("routing_n", 0) + a.shape[0]
                        add("routing_entropy", routing_entropy(a)[0])
                    if f"gates_{side}" in s and d["gates"] is None:
                        d["gates"] = s[f"gates_{side}"].cpu()
                    if f"delta_ratio_{side}" in s:
                        add(f"delta_ratio_{side}", float(s[f"delta_ratio_{side}"].mean()))

        set_collect_stats(net, False)
        if was_training:
            net.train()

        exp = trainer.logger.experiment
        log: dict = {"epoch": ep}
        depth = len(net.blocks)
        image_layers = {0, depth // 2, depth - 1}

        for i, (lname, mod) in enumerate(iter_stat_modules(net)):
            d = agg.get(lname)
            if d is None:
                continue
            short = _short(lname)
            for key, total in d["scalars"].items():
                log[f"probe/{short}/{key}"] = total / n_chunks
            if d["q_rows"] is not None:
                prs = q_participation_ratios(d["q_rows"])
                log[f"probe/{short}/q_pr_mean"] = float(prs.mean())
                log[f"probe/{short}/q_pr_min"] = float(prs.min())
                log[f"probe/{short}/q_pr_frac"] = float(prs.mean() / d["q_rows"].shape[-1])
            if d["routing_sum"] is not None:
                rows = d["routing_sum"] / max(1, d.get("routing_n", 1))  # [N, M] mean routing
                mi_nats, mi_norm = position_routing_mi(rows.unsqueeze(0))
                log[f"probe/{short}/routing_mi"] = mi_nats
                log[f"probe/{short}/routing_mi_norm"] = mi_norm
                log[f"probe/{short}/load_max"] = float(rows.mean(0).max())
                if i in image_layers:
                    log[f"probe_img/{short}/routing"] = fig_to_wandb(
                        latent_grid_figure(rows, grid, f"{short} routing (ep {ep})")
                    )
            if d["gates"] is not None:
                for key, val in gate_position_content_split(d["gates"]).items():
                    log[f"probe/{short}/gates_{key}"] = val
            for side in ("q", "k"):
                alpha = getattr(mod, f"alpha_{side}", None)
                if alpha is not None:
                    log[f"probe/{short}/alpha_{side}_absmax"] = float(alpha.detach().abs().max())
            # positional-prior maps: <p_i, e_m> for B1-style banks (duck-typed)
            gen = getattr(mod, "gen", None)
            if gen is not None and i in image_layers:
                for side, bank in gen.items():
                    p, E = getattr(bank, "p", None), getattr(bank, "E", None)
                    if p is not None and E is not None:
                        prior = p.materialize() @ E.t() / (E.shape[-1] ** 0.5)
                        log[f"probe_img/{short}/prior_{side}"] = fig_to_wandb(
                            latent_grid_figure(prior, grid, f"{short} prior_{side} (ep {ep})")
                        )
                    b = getattr(bank, "b", None)
                    if b is not None:
                        log[f"probe_img/{short}/gate_table_{side}"] = fig_to_wandb(
                            latent_grid_figure(torch.sigmoid(b.materialize()), grid,
                                               f"{short} gate table {side} (ep {ep})")
                        )

        exp.log(log)

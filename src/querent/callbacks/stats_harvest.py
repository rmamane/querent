"""Cheap "light"-tier stats harvest every K training steps.

Arms the collect flag BEFORE the forward, harvests reductions (means +
entropies) AFTER it, then disarms — 1/K forwards pay a small stats cost, the
rest pay exactly zero by the base-class contract.
"""

from __future__ import annotations

import lightning.pytorch as pl

from ..diagnostics.stats_api import iter_stat_modules, set_collect_stats

EPS = 1e-9
_DIST_PREFIXES = ("routing_", "gates_", "remix")


def _short(layer_name: str) -> str:
    return layer_name.replace("blocks.", "L").replace(".attn", "")


class StatsHarvestCallback(pl.Callback):
    def __init__(self, every_steps: int = 100):
        self.every = max(1, int(every_steps))
        self._armed = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.global_step % self.every == 0:
            set_collect_stats(pl_module.net, "light")
            self._armed = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._armed:
            return
        out = {}
        for lname, mod in iter_stat_modules(pl_module.net):
            if not mod.stats:
                continue
            short = _short(lname)
            for key, t in mod.stats.items():
                t = t.float()
                out[f"stats/{short}/{key}_mean"] = t.mean()
                if key.startswith(_DIST_PREFIXES) and t.ndim >= 2:
                    p = t / (t.sum(dim=-1, keepdim=True) + EPS)
                    out[f"stats/{short}/{key}_entropy"] = -(p * (p + EPS).log()).sum(-1).mean()
                if key.startswith("alpha_"):
                    out[f"stats/{short}/{key}_absmax"] = t.abs().max()
        if out:
            pl_module.log_dict(out, on_step=True, on_epoch=False)
        set_collect_stats(pl_module.net, False)
        self._armed = False

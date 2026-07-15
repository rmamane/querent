"""CIFAR-100-C evaluation on the best checkpoint at fit end.

Skips silently (with a log line) when the corruption data isn't on disk —
local smokes don't download the 2.9 GB tar; vast.ai onstart does.
"""

from __future__ import annotations

import logging

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from ..data.cifar100c import CIFAR100C, CORRUPTIONS_15, SEVERITIES, is_available

log = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_corruptions(net, data_dir: str, device, batch_size: int = 1024,
                         corruptions=CORRUPTIONS_15) -> dict[str, float]:
    net.eval()
    out: dict[str, float] = {}
    per_corruption = []
    for c in corruptions:
        sev_accs = []
        for s in SEVERITIES:
            loader = DataLoader(CIFAR100C(data_dir, c, s), batch_size=batch_size, num_workers=2)
            correct = total = 0
            for x, y in loader:
                pred = net(x.to(device)).argmax(-1).cpu()
                correct += int((pred == y).sum())
                total += len(y)
            acc = correct / total
            out[f"{c}/sev{s}"] = acc
            sev_accs.append(acc)
        out[f"{c}/mean"] = sum(sev_accs) / len(sev_accs)
        per_corruption.append(out[f"{c}/mean"])
    out["mCA"] = sum(per_corruption) / len(per_corruption)
    return out


class CorruptionEvalCallback(pl.Callback):
    def __init__(self, cfg):
        self.enabled = bool(cfg.eval.corruptions_at_end)
        self.data_dir = str(cfg.paths.data_dir)

    def on_fit_end(self, trainer, pl_module):
        if not self.enabled or trainer.fast_dev_run or trainer.logger is None:
            return
        if not is_available(self.data_dir):
            log.info("CIFAR-100-C not on disk — skipping corruption eval")
            return

        # Evaluate the BEST checkpoint (fall back to current weights).
        best_path = None
        for cb in trainer.checkpoint_callbacks:
            if getattr(cb, "monitor", None) == "val/acc1" and cb.best_model_path:
                best_path = cb.best_model_path
        if best_path:
            state = torch.load(best_path, map_location=pl_module.device, weights_only=False)
            pl_module.load_state_dict(state["state_dict"])

        results = evaluate_corruptions(pl_module.net, self.data_dir, pl_module.device)
        exp = trainer.logger.experiment
        exp.log({f"final/c100c/{k}": v for k, v in results.items()})
        exp.summary["final/c100c/mCA"] = results["mCA"]
        clean = trainer.callback_metrics.get("val/acc1_best")
        if clean is not None:
            exp.summary["final/c100c/rel_delta"] = results["mCA"] - float(clean)

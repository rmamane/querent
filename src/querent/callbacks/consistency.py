"""Translation- and flip-consistency evals on a fixed test subset.

CIFAR is 32 px, so translations are ±2/±4 px crops from a reflect-padded 40 px
canvas (built once, cached). The subset is fixed (seed 1234) so deltas are
comparable across every run in the program; the full 10k test set is used once
at fit end under the ``final/`` prefix. This is the primary evidence channel
for arms whose whole point is positional structure.
"""

from __future__ import annotations

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR100

from ..data.cifar100 import eval_transform

MAGNITUDES = (2, 4)


def _predict(net, xs: torch.Tensor, device, chunk: int = 512) -> torch.Tensor:
    preds = []
    for xb in xs.split(chunk):
        preds.append(net(xb.to(device)).argmax(dim=-1).cpu())
    return torch.cat(preds)


class ConsistencyCallback(pl.Callback):
    def __init__(self, cfg):
        self.every = int(cfg.eval.consistency_every_epochs)
        self.subset = int(cfg.eval.consistency_subset)
        self.data_dir = str(cfg.paths.data_dir)
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def _padded_set(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        if n not in self._cache:
            ds = CIFAR100(self.data_dir, train=False, transform=eval_transform())
            if n < len(ds):
                g = torch.Generator().manual_seed(1234)
                idx = torch.randperm(len(ds), generator=g)[:n].tolist()
            else:
                idx = list(range(len(ds)))
            xs = torch.stack([ds[i][0] for i in idx])
            ys = torch.tensor([ds[i][1] for i in idx])
            self._cache[n] = (F.pad(xs, (4, 4, 4, 4), mode="reflect"), ys)
        return self._cache[n]

    @staticmethod
    def _crop(padded: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        return padded[:, :, 4 + dy : 36 + dy, 4 + dx : 36 + dx]

    @torch.no_grad()
    def _run(self, pl_module, n: int) -> dict[str, float]:
        padded, ys = self._padded_set(n)
        net = pl_module.net
        was_training = net.training
        net.eval()
        device = pl_module.device

        center = _predict(net, self._crop(padded, 0, 0), device)
        clean_acc = float((center == ys).float().mean())
        out = {"clean_acc": clean_acc}

        for mag in MAGNITUDES:
            accs, agrees = [], []
            for dy, dx in ((mag, 0), (-mag, 0), (0, mag), (0, -mag)):
                p = _predict(net, self._crop(padded, dy, dx), device)
                accs.append(float((p == ys).float().mean()))
                agrees.append(float((p == center).float().mean()))
            out[f"shift{mag}/top1_delta"] = sum(accs) / 4 - clean_acc
            out[f"shift{mag}/agreement"] = sum(agrees) / 4

        flipped = _predict(net, torch.flip(self._crop(padded, 0, 0), dims=[3]), device)
        out["flip/top1_delta"] = float((flipped == ys).float().mean()) - clean_acc
        out["flip/agreement"] = float((flipped == center).float().mean())

        if was_training:
            net.train()
        return out

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.fast_dev_run:
            return
        ep = int(trainer.current_epoch)
        last = trainer.max_epochs is not None and ep == trainer.max_epochs - 1
        if ep % self.every != 0 and not last:
            return
        if trainer.logger is None or not hasattr(trainer.logger, "experiment"):
            return
        metrics = self._run(pl_module, self.subset)
        trainer.logger.experiment.log({f"consist/{k}": v for k, v in metrics.items()} | {"epoch": ep})

    def on_fit_end(self, trainer, pl_module):
        if trainer.fast_dev_run or trainer.logger is None:
            return
        metrics = self._run(pl_module, 10_000)  # full test set, once
        exp = trainer.logger.experiment
        exp.log({f"final/consist/{k}": v for k, v in metrics.items()})
        for k, v in metrics.items():
            exp.summary[f"final/consist/{k}"] = v

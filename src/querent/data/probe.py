"""The fixed probe batch: 512 train images, eval transforms, seed 1234.

Independent of cfg.seed by design — every run in the program computes
diagnostics on byte-identical images, so probe metrics are comparable across
arms, phases, and machines.
"""

from __future__ import annotations

import torch
from torchvision.datasets import CIFAR100

from .cifar100 import eval_transform


def build_probe_batch(data_dir: str, n: int = 512, seed: int = 1234) -> tuple[torch.Tensor, torch.Tensor]:
    ds = CIFAR100(str(data_dir), train=True, transform=eval_transform())
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    xs = torch.stack([ds[i][0] for i in idx])
    ys = torch.tensor([ds[i][1] for i in idx])
    return xs, ys

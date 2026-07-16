"""CIFAR-100-C corruption datasets (Hendrycks & Dietterich, Zenodo 3555552).

Layout on disk: ``<root>/CIFAR-100-C/{corruption}.npy`` each [50000, 32, 32, 3]
uint8 (the 10k test images x severities 1..5, stacked), plus ``labels.npy``.
``scripts/download_data.py --with-c100c`` fetches and verifies it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .cifar100 import CIFAR100_MEAN, CIFAR100_STD

CORRUPTIONS_15 = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]
CORRUPTIONS_EXTRA = ["speckle_noise", "gaussian_blur", "spatter", "saturate"]
SEVERITIES = (1, 2, 3, 4, 5)


def c100c_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / "CIFAR-100-C"


def is_available(data_dir: str | Path) -> bool:
    d = c100c_dir(data_dir)
    return (d / "labels.npy").exists() and all((d / f"{c}.npy").exists() for c in CORRUPTIONS_15)


class CIFAR100C(Dataset):
    """One (corruption, severity) slice, normalized like the clean eval stream."""

    def __init__(self, data_dir: str | Path, corruption: str, severity: int):
        if severity not in SEVERITIES:
            raise ValueError(f"severity must be 1..5, got {severity}")
        d = c100c_dir(data_dir)
        lo, hi = 10000 * (severity - 1), 10000 * severity
        self.images = np.load(d / f"{corruption}.npy", mmap_mode="r")[lo:hi]
        self.labels = np.load(d / "labels.npy")[lo:hi].astype(np.int64)
        self.mean = torch.tensor(CIFAR100_MEAN).view(3, 1, 1)
        self.std = torch.tensor(CIFAR100_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        x = torch.from_numpy(np.ascontiguousarray(self.images[i])).permute(2, 0, 1).float() / 255.0
        return (x - self.mean) / self.std, int(self.labels[i])

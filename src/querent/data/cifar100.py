"""CIFAR-100 DataModule with the DeiT-style augmentation stack.

timm (not torchvision) for RandAugment and RandomErasing — torchvision's are
not DeiT-equivalent (different op set, no magnitude-std, constant-fill
erasing). Mixup/CutMix happen on-device in the LightningModule, not here —
DataLoader workers stay dumb and fast.

No RandomResizedCrop at 32 px: reflect-pad(4) + crop is the correct CIFAR analog.
"""

from __future__ import annotations

import lightning.pytorch as pl
import torch
from timm.data.auto_augment import rand_augment_transform
from timm.data.random_erasing import RandomErasing
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def train_transform(recipe) -> transforms.Compose:
    t: list = [
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
    ]
    if recipe.randaug:
        hparams = dict(
            translate_const=10,
            img_mean=tuple(int(255 * m + 0.5) for m in CIFAR100_MEAN),
        )
        t.append(rand_augment_transform(str(recipe.randaug), hparams))
    t += [transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)]
    if recipe.random_erasing.p > 0:
        # timm RandomErasing runs on the normalized tensor (noise fill, per DeiT).
        t.append(RandomErasing(probability=recipe.random_erasing.p,
                               mode=str(recipe.random_erasing.mode), device="cpu"))
    return transforms.Compose(t)


def eval_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])


class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.data_dir = str(cfg.paths.data_dir)
        self.train_set = None
        self.val_set = None

    def prepare_data(self) -> None:
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        self.train_set = CIFAR100(self.data_dir, train=True, transform=train_transform(self.cfg.recipe))
        self.val_set = CIFAR100(self.data_dir, train=False, transform=eval_transform())
        frac = float(self.cfg.data.subset_fraction)
        if frac < 1.0:
            # Data-level subsetting uses its own fixed seed: identical subsets
            # for every arm and every cfg.seed (smoke determinism).
            g = torch.Generator().manual_seed(0)
            n_tr = max(256, int(len(self.train_set) * frac))
            idx_tr = torch.randperm(len(self.train_set), generator=g)[:n_tr].tolist()
            self.train_set = Subset(self.train_set, idx_tr)
            n_va = max(512, int(len(self.val_set) * frac * 4))
            idx_va = torch.randperm(len(self.val_set), generator=g)[:n_va].tolist()
            self.val_set = Subset(self.val_set, idx_va)

    def _loader(self, ds, *, shuffle: bool, drop_last: bool) -> DataLoader:
        workers = int(self.cfg.data.num_workers)
        return DataLoader(
            ds,
            batch_size=int(self.cfg.recipe.batch_size),
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
            drop_last=drop_last,  # drop_last on train keeps step counts identical across arms
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_set, shuffle=False, drop_last=False)

"""Optimizer param groups + warmup-cosine schedule.

The no-decay set matters more than usual here: weight decay on the residual
gates (alpha) is a spring pinning them at 0 — it silently kills every variant.
Two mechanisms, belt and suspenders: (a) every param with ndim <= 1, and
(b) every name in model.no_weight_decay() (covers 2-d+ tables: QK-norm gamma,
per-position tables, latent keys, pos_emb).
"""

from __future__ import annotations

import math

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    no_wd_names = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name in no_wd_names:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def warmup_cosine(optimizer: Optimizer, *, warmup_steps: int, total_steps: int,
                  min_lr_ratio: float) -> LambdaLR:
    """Linear warmup -> cosine to min_lr_ratio * base_lr. Stepped per optimizer step."""

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos

    return LambdaLR(optimizer, fn)

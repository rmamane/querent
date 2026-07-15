"""Harness side of the stats contract (see querent/models/attention/base.py).

Deliberately duck-typed: a "stat module" is any module with a ``collect_stats``
attribute. This file must not import querent.models — layering is
models -> torch only; diagnostics -> torch only.

Contract recap:
- ``module.collect_stats`` is one of ``False | "light" | "full"``.
- After a forward with a truthy level, ``module.stats`` is a dict of DETACHED
  tensors; with ``False`` it is ``None`` and no stats work happened.
- "light": routing weights, gates, alpha, position priors, delta/base norm
  ratios — cheap, harvested every K steps.
- "full": adds exact ``attn_probs`` (module switches to the math path for that
  forward) plus post-transform ``q``/``k``. Probe-only; training stays on SDPA.
- ``module.last_aux_loss`` is ``None`` or a scalar tensor, reset each forward.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn

STATS_LEVELS = (False, "light", "full")


def is_stat_module(m: nn.Module) -> bool:
    return hasattr(m, "collect_stats")


def set_collect_stats(model: nn.Module, level: bool | str) -> None:
    if level not in STATS_LEVELS:
        raise ValueError(f"level must be one of {STATS_LEVELS}, got {level!r}")
    for m in model.modules():
        if is_stat_module(m):
            m.set_collect_stats(level) if hasattr(m, "set_collect_stats") else setattr(
                m, "collect_stats", level
            )


def iter_stat_modules(model: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    """Yield (dotted_name, module) for every stat-emitting module, e.g. ('blocks.3.attn', m)."""
    for name, m in model.named_modules():
        if is_stat_module(m):
            yield name, m


def harvest_and_clear(model: nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    """Collect every module's stats dict (keyed by module name) and clear them."""
    out: dict[str, dict[str, torch.Tensor]] = {}
    for name, m in iter_stat_modules(model):
        if getattr(m, "stats", None):
            out[name] = m.stats
            m.stats = None
    return out


def collect_aux_losses(model: nn.Module) -> torch.Tensor | None:
    """Sum of per-module aux losses from the last forward (None if no module set one)."""
    total: torch.Tensor | None = None
    for _, m in iter_stat_modules(model):
        loss = getattr(m, "last_aux_loss", None)
        if loss is not None:
            total = loss if total is None else total + loss
    return total

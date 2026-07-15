"""A0 — vanilla attention baseline (plus its capacity controls).

q = W_Q x, k = W_K x, QK-RMSNorm (as in every arm). Controls that live here:
- talking_heads: static learned mixing of attention *logits* across heads,
  W_th init to identity so the arm starts exactly at vanilla. Needs
  materialized logits => breaks SDPA; runs on the math path. Compare accuracy,
  never wall-clock, against SDPA arms.
- 2D-RoPE: base-level ``rope=True`` flag.
- wider-MLP param matching is a ViT-level knob (``mlp_ratio``), not ours.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import STATS_SCHEMA, DynamicQKBase
from .registry import register_attention


@register_attention("vanilla")
class VanillaDQK(DynamicQKBase):
    def __init__(self, dim, num_heads, grid_size, *, talking_heads: bool = False, **kw):
        super().__init__(dim, num_heads, grid_size, has_delta=False, **kw)
        self.talking_heads = talking_heads
        if talking_heads:
            # Identity init => math path starts exactly equal to plain A0.
            # No weight decay: WD pulls toward the zero matrix, not identity.
            self.w_th = nn.Parameter(torch.eye(num_heads))
            self._no_wd.append("w_th")

    def _needs_math(self) -> bool:
        return self.talking_heads

    def _mix_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.talking_heads:
            return logits
        # L'_g = sum_h W_th[h, g] * L_h
        return torch.einsum("bhij,hg->bgij", logits, self.w_th)


STATS_SCHEMA["vanilla"] = {}

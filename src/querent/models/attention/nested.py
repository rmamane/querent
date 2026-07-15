"""B3 — nested Q: an inner N x N attention generates the query delta.

    delta_i = InnerAttn(q = W_a x_i, k = W_b x, v = W_c x)_i
    q_i     = W_Q x_i + alpha * delta_i

Nexus-style (transfer study). Nearly a whole extra attention module in cost —
its honest control is the depth-matched A0 + 1 block (config-level, not here).

Saturation guard: parameter-free RMSNorm on inner q/k, default ON — as W_a/W_b
grow, inner logits blow up, the inner attention saturates, and gradients
through it vanish; the alpha-gated outer path hides this until alpha is large.
Monitored via ``inner_entropy``. At init the inner softmax is near-uniform, so
delta ~ global mean of W_c x — a sensible nonzero start for alpha's gradient.
Q-only, residual-only by design.
"""

from __future__ import annotations

import math

import torch.nn.functional as F
from torch import nn

from .base import STATS_SCHEMA, DynamicQKBase
from .registry import register_attention


@register_attention("nested")
class NestedQDQK(DynamicQKBase):
    def __init__(self, dim, num_heads, grid_size, *, inner_qk_norm: bool = True, **kw):
        if kw.get("apply_to", "q") != "q":
            raise ValueError("nested is q-only")
        if kw.get("mode", "residual") != "residual":
            raise ValueError("nested has no replace mode")
        super().__init__(dim, num_heads, grid_size, **kw)
        self.inner_qk_norm = inner_qk_norm
        self.w_a = nn.Linear(dim, dim, bias=False)
        self.w_b = nn.Linear(dim, dim, bias=False)
        self.w_c = nn.Linear(dim, dim, bias=False)  # doubles as value/output proj
        for lin in (self.w_a, self.w_b, self.w_c):
            nn.init.trunc_normal_(lin.weight, std=0.02)

    def delta_q(self, x, *, base, ctx):
        qi, ki, vi = self.w_a(x), self.w_b(x), self.w_c(x)
        if self.inner_qk_norm:
            qi = F.rms_norm(qi, (qi.shape[-1],))
            ki = F.rms_norm(ki, (ki.shape[-1],))
        if self.collect_stats:
            probs = ((qi @ ki.transpose(-2, -1)) / math.sqrt(qi.shape[-1])).softmax(dim=-1)
            self.stats["inner_entropy"] = (
                -(probs * (probs + 1e-9).log()).sum(dim=-1).detach()
            )
            return probs @ vi
        # single-head SDPA (add a head axis)
        return F.scaled_dot_product_attention(
            qi.unsqueeze(1), ki.unsqueeze(1), vi.unsqueeze(1)
        ).squeeze(1)


STATS_SCHEMA["nested"] = {
    "inner_entropy": ("B", "N"),
}

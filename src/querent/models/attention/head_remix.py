"""A5 — content-dependent query-space head mixing (near-free).

    q_i^h = q~_i^h + alpha * sum_{h'} a_i^{h,h'} q~_i^{h'}
    a_i = softmax_{h'}( Linear(d, h^2)(x_i) )

Contrast to file away: A5 is content-dependent, query-space, pre-softmax;
talking-heads (A0 control) is static, logit-space, post-QK. At init the mix is
~uniform, so the delta ~ head-mean of q~ — nonzero, so dL/dalpha != 0 and the
zero-init alpha can open. Q-only by design. Reuses ``base`` (= W_Q x) passed
into the hook — the queries are not recomputed.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import STATS_SCHEMA, DynamicQKBase
from .registry import register_attention


@register_attention("head_remix")
class HeadRemixDQK(DynamicQKBase):
    def __init__(self, dim, num_heads, grid_size, **kw):
        if kw.get("apply_to", "q") != "q":
            raise ValueError("head_remix is q-only")
        if kw.get("mode", "residual") != "residual":
            raise ValueError("head_remix has no replace mode")
        super().__init__(dim, num_heads, grid_size, **kw)
        self.mix = nn.Linear(dim, num_heads * num_heads, bias=True)
        nn.init.trunc_normal_(self.mix.weight, std=0.02)
        nn.init.zeros_(self.mix.bias)

    def delta_q(self, x, *, base, ctx):
        B, N, _ = x.shape
        h, dh = self.num_heads, self.head_dim
        a = self.mix(x).view(B, N, h, h).softmax(dim=-1)  # g receives, h' sources
        qt = base.view(B, N, h, dh)
        delta = torch.einsum("bngh,bnhc->bngc", a, qt)
        if self.collect_stats:
            self.stats["remix"] = a.detach()
        return delta.reshape(B, N, self.dim)


STATS_SCHEMA["head_remix"] = {
    "remix": ("B", "N", "h", "h"),
}

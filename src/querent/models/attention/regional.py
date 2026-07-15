"""B2 — regional context: pooled scene summaries steer per-token queries.

    regions g = summary( avg_pool(token grid -> region_grid) )      [B, R, d]
    c_i = softmax(<W_qc x_i, W_kc g> / sqrt(d_c)) W_vc g            [B, N, d_c]
    delta_i = W_oc c_i                                              [B, N, d]

Ablation mapping (all through base machinery): context-into-Q = apply_to="q",
into-K = "k", both = "qk" (shared context c from ``prepare``, separate W_oc and
separate alpha per side); GC-ViT-style replace-Q = mode="replace".

The inner cross-attention is deliberately the math path — [B, N, R] with
R <= 49 is tiny, and it makes ``region_attn`` stats free. SDPA is reserved for
the main attention. ``adaptive_avg_pool2d`` allows non-divisible grids
(documented: uneven windows); prefer region grids that divide the token grid.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .base import STATS_SCHEMA, DynamicQKBase
from .registry import register_attention


@register_attention("regional")
class RegionalContextDQK(DynamicQKBase):
    def __init__(self, dim, num_heads, grid_size, *, region_grid=(4, 4),
                 summary: str = "linear", ctx_dim: int = 64, **kw):
        if summary not in ("linear", "mlp"):
            raise ValueError(f"summary must be linear|mlp, got {summary!r}")
        super().__init__(dim, num_heads, grid_size, **kw)
        rh, rw = region_grid
        gh, gw = self.grid_size
        if rh > gh or rw > gw:
            raise ValueError(f"region_grid {region_grid} exceeds token grid {self.grid_size}")
        self.region_grid = (rh, rw)
        self.num_regions = rh * rw
        self.ctx_dim = ctx_dim

        self.summary = (
            nn.Linear(dim, dim)
            if summary == "linear"
            else nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        )
        self.w_qc = nn.Linear(dim, ctx_dim, bias=False)
        self.w_kc = nn.Linear(dim, ctx_dim, bias=False)
        self.w_vc = nn.Linear(dim, ctx_dim, bias=False)
        sides = ["q"] if self.mode == "replace" else [s for s in ("q", "k") if s in self.apply_to]
        self.w_oc = nn.ModuleDict({s: nn.Linear(ctx_dim, dim, bias=False) for s in sides})

        # Variance-preserving fan-in init on the context chain (NOT 0.02): the
        # path is three stacked linears plus pooling/attention averaging, and a
        # 0.02 chain shrinks the delta to ~1e-6 of the base query — a
        # structurally dead arm at init (violates the "delta is O(1)" contract).
        for m in self.modules():
            if isinstance(m, nn.Linear) and m not in (self.w_q, self.w_k, self.w_v, self.proj):
                nn.init.trunc_normal_(m.weight, std=m.in_features**-0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def prepare(self, x: torch.Tensor) -> torch.Tensor:
        B, N, d = x.shape
        gh, gw = self.grid_size
        g = x.transpose(1, 2).reshape(B, d, gh, gw)
        g = F.adaptive_avg_pool2d(g, self.region_grid).flatten(2).transpose(1, 2)  # [B, R, d]
        g = self.summary(g)
        attn = (self.w_qc(x) @ self.w_kc(g).transpose(-2, -1)) / math.sqrt(self.ctx_dim)
        attn = attn.softmax(dim=-1)  # [B, N, R] — math path on purpose (tiny; stats free)
        if self.collect_stats:
            self.stats["region_attn"] = attn.detach()
        return attn @ self.w_vc(g)  # [B, N, ctx_dim]

    def delta_q(self, x, *, base, ctx):
        return self.w_oc["q"](ctx)

    def delta_k(self, x, *, base, ctx):
        return self.w_oc["k"](ctx)


STATS_SCHEMA["regional"] = {
    "region_attn": ("B", "N", "R"),
}

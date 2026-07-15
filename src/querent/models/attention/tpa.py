"""A4 — Tensor Product Attention query factorization (Zhang et al. 2025) in a ViT.

Per token, the per-head query matrix is generated as a rank-r tensor product:

    Q_i = (1/r) * A(x_i) B(x_i)^T,   A: R^d -> R^{h×r},  B: R^d -> R^{r×d_h}

Modes: "replace" (faithful TPA-Q — w_q is None, exempt from the alpha=0
equivalence, covered by the scale-sanity contract instead) or "residual"
(our extension: q = W_Q x + alpha * TPA(x)). ``pos_bias`` adds per-position
tables inside the generators (our extension; "a" | "b" | "ab" | "none").

Init derivation (replace mode must start near vanilla scale): an entry of
W_Q x has Var ≈ d * sigma_W^2 with sigma_W = 0.02 on unit-RMS LN input; a TPA
entry has Var = (d sigma_A^2)(d sigma_B^2)/r. Matching gives
sigma_A * sigma_B = sigma_W * sqrt(r/d); we take the symmetric solution.
Honest cost note: w_b alone is d*r*d_h params — 2.7x W_Q at Ti scale.
Param-matched comparisons are mandatory for this arm.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .base import STATS_SCHEMA, DynamicQKBase
from .pos_utils import HalfGridParam
from .registry import register_attention


class _TPAGen(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, rank: int,
                 grid_size, pos_bias: str, flip_tie: bool):
        super().__init__()
        self.num_heads, self.head_dim, self.rank = num_heads, head_dim, rank
        self.w_a = nn.Linear(dim, num_heads * rank, bias=True)
        self.w_b = nn.Linear(dim, rank * head_dim, bias=True)
        sigma = math.sqrt(0.02 * math.sqrt(rank / dim))
        for lin in (self.w_a, self.w_b):
            nn.init.trunc_normal_(lin.weight, std=sigma)
            nn.init.zeros_(lin.bias)
        self.pos_a = (
            HalfGridParam(grid_size, (num_heads, rank), flip_tie=flip_tie, init="zeros")
            if "a" in pos_bias else None
        )
        self.pos_b = (
            HalfGridParam(grid_size, (rank, head_dim), flip_tie=flip_tie, init="zeros")
            if "b" in pos_bias else None
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        h, r, dh = self.num_heads, self.rank, self.head_dim
        a = self.w_a(x).view(B, N, h, r)
        b = self.w_b(x).view(B, N, r, dh)
        if self.pos_a is not None:
            a = a + self.pos_a.materialize().view(1, N, h, r)
        if self.pos_b is not None:
            b = b + self.pos_b.materialize().view(1, N, r, dh)
        q = torch.einsum("bnhr,bnrc->bnhc", a, b) / r
        return q.reshape(B, N, h * dh), a, b


@register_attention("tpa")
class TPAQueryDQK(DynamicQKBase):
    def __init__(self, dim, num_heads, grid_size, *, tpa_rank: int = 8,
                 pos_bias: str = "none", flip_tie: bool = False, **kw):
        if pos_bias not in ("none", "a", "b", "ab"):
            raise ValueError(f"pos_bias must be none|a|b|ab, got {pos_bias!r}")
        super().__init__(dim, num_heads, grid_size, **kw)
        self.tpa_rank = tpa_rank
        sides = ["q"] if self.mode == "replace" else [s for s in ("q", "k") if s in self.apply_to]
        self.gen = nn.ModuleDict(
            {s: _TPAGen(dim, num_heads, self.head_dim, tpa_rank, self.grid_size, pos_bias, flip_tie)
             for s in sides}
        )
        for s in sides:
            for t in ("pos_a", "pos_b"):
                if getattr(self.gen[s], t) is not None:
                    self._no_wd.append(f"gen.{s}.{t}.weight")

    def _delta(self, side: str, x: torch.Tensor) -> torch.Tensor:
        q, a, b = self.gen[side](x)
        if self.collect_stats:
            self.stats[f"tpa_a_fro_{side}"] = a.detach().flatten(2).norm(dim=-1)
            self.stats[f"tpa_b_fro_{side}"] = b.detach().flatten(2).norm(dim=-1)
        return q

    def delta_q(self, x, *, base, ctx):
        return self._delta("q", x)

    def delta_k(self, x, *, base, ctx):
        return self._delta("k", x)


STATS_SCHEMA["tpa"] = {
    "tpa_a_fro_q": ("B", "N"),
    "tpa_b_fro_q": ("B", "N"),
    "tpa_a_fro_k": ("B", "N"),
    "tpa_b_fro_k": ("B", "N"),
}

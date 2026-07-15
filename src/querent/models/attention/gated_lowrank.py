"""A1/A2/A3 — gated low-rank query/key deltas (one class, three gate modes).

    delta_i = U (c_i ⊙ V^T x_i)
    c_i = sigmoid(G x_i)          gate_mode="content"   (A1)
    c_i = sigmoid(b_i)            gate_mode="position"  (A2, b ∈ R^{N×r})
    c_i = sigmoid(G x_i + b_i)    gate_mode="both"      (A3)

Init: down/up/G trunc_normal(0.02) — NOT zero (alpha is the only zero); G bias
and b start at 0 so gates open at sigmoid(0)=0.5, a benign half-scale on the
delta. Positional tables are flip-tieable and excluded from weight decay.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import STATS_SCHEMA, DynamicQKBase
from .pos_utils import HalfGridParam
from .registry import register_attention


class _GatedLowRankGen(nn.Module):
    def __init__(self, dim: int, rank: int, gate_mode: str, grid_size, flip_tie: bool):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.G = nn.Linear(dim, rank, bias=True) if gate_mode in ("content", "both") else None
        self.b = (
            HalfGridParam(grid_size, (rank,), flip_tie=flip_tie, init="zeros")
            if gate_mode in ("position", "both")
            else None
        )
        nn.init.trunc_normal_(self.down.weight, std=0.02)
        nn.init.trunc_normal_(self.up.weight, std=0.02)
        if self.G is not None:
            nn.init.trunc_normal_(self.G.weight, std=0.02)
            nn.init.zeros_(self.G.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.down(x)  # [B, N, r]
        pre = None
        if self.G is not None:
            pre = self.G(x)
        if self.b is not None:
            table = self.b.materialize()[None]  # [1, N, r]
            pre = table if pre is None else pre + table
        gates = torch.sigmoid(pre)
        return self.up(gates * z), gates


@register_attention("gated_lowrank")
class GatedLowRankDQK(DynamicQKBase):
    def __init__(self, dim, num_heads, grid_size, *, rank: int = 8,
                 gate_mode: str = "content", flip_tie: bool = False, **kw):
        if gate_mode not in ("content", "position", "both"):
            raise ValueError(f"gate_mode must be content|position|both, got {gate_mode!r}")
        super().__init__(dim, num_heads, grid_size, **kw)
        self.rank = rank
        self.gate_mode = gate_mode
        sides = [s for s in ("q", "k") if s in self.apply_to]
        self.gen = nn.ModuleDict(
            {s: _GatedLowRankGen(dim, rank, gate_mode, self.grid_size, flip_tie) for s in sides}
        )
        for s in sides:
            if gate_mode in ("position", "both"):
                self._no_wd.append(f"gen.{s}.b.weight")

    def _delta(self, side: str, x: torch.Tensor) -> torch.Tensor:
        delta, gates = self.gen[side](x)
        if self.collect_stats:
            self.stats[f"gates_{side}"] = gates.detach()
        return delta

    def delta_q(self, x, *, base, ctx):
        return self._delta("q", x)

    def delta_k(self, x, *, base, ctx):
        return self._delta("k", x)


STATS_SCHEMA["gated_lowrank"] = {
    "gates_q": ("B", "N", "r"),
    "gates_k": ("B", "N", "r"),
}

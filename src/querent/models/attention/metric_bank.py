"""B1 — metric bank: competitively-routed query metrics (the headline arm).

M latents with routing keys e_m and payloads; per token:

    l_im = <W_r x_i + p_i, e_m> / sqrt(d_r)        (p_i: optional positional prior)
    a_i  = route(l_i),  route in {softmax, sigmoid, topk}
    delta_i = sum_m a_im * payload_m(x_i)

Payload granularity (the spectrum MoA/SwitchHead/MoH never ran):
    vector:  sum_m a_im u_m                        (content-independent directions)
    rank1:   sum_m a_im u_m (v_m^T x_i)            (routed rank-1 metric bases)
    head:    sum_m a_im W_m x_i                    (routed full/low-rank projections)

Guards baked in (see plan):
- sigmoid routing divides by M — raw sigmoids sum ~M/2 payloads, a 32x scale
  confound across the M in {4,16,64} sweep.
- top-k is capacity-free DENSE (static-k topk -> k-hot mask -> masked softmax
  -> dense einsum): compile-safe, no data-dependent shapes.
- E init N(0, d_r^{-1/2}) => near-uniform routing at start (no day-one collapse).
- head payload requires M <= 8 or a low-rank head_rank.
- aux losses (entropy anti-collapse, Switch-style load balance) live behind
  lambda flags, default 0; the trainer delays them until alpha unfreezes.
- replace mode re-derives payload init so ||q_replace||/||q_vanilla|| ~ 1 at
  init (scale-sanity contract; residual mode keeps 0.02 — alpha owns scale).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .base import STATS_SCHEMA, DynamicQKBase
from .pos_utils import HalfGridParam
from .registry import register_attention

_ROUTES = ("softmax", "sigmoid", "topk")
_PAYLOADS = ("vector", "rank1", "head")


class _MetricBank(nn.Module):
    def __init__(self, dim: int, grid_size, *, num_latents: int, d_route: int, route: str,
                 topk_k: int, payload: str, head_rank: int | None, pos_prior: bool,
                 flip_tie: bool, aux_entropy: float, aux_load_balance: float, replace: bool):
        super().__init__()
        M = num_latents
        self.M, self.d_route, self.route_kind, self.k = M, d_route, route, topk_k
        self.payload_kind, self.head_rank = payload, head_rank
        self.aux_entropy, self.aux_load_balance = float(aux_entropy), float(aux_load_balance)

        self.w_r = nn.Linear(dim, d_route, bias=False)
        nn.init.trunc_normal_(self.w_r.weight, std=0.02)
        # init logit std ~ 0.03 => routing starts near-uniform, deliberately.
        self.E = nn.Parameter(torch.randn(M, d_route) * d_route**-0.5)
        self.p = HalfGridParam(grid_size, (d_route,), flip_tie=flip_tie, init="zeros") if pos_prior else None

        # Payload init: 0.02 in residual mode; replace mode matches vanilla q scale
        # (softmax-uniform a at init => Var(delta entry) = sigma^2-dependent, see module doc).
        if payload == "vector":
            std = 0.02 * math.sqrt(dim * M) if replace else 0.02
            self.U = nn.Parameter(torch.empty(M, dim))
            nn.init.trunc_normal_(self.U, std=std)
        elif payload == "rank1":
            std = math.sqrt(0.02 * math.sqrt(M)) if replace else 0.02
            self.U = nn.Parameter(torch.empty(M, dim))
            self.Vp = nn.Parameter(torch.empty(M, dim))
            nn.init.trunc_normal_(self.U, std=std)
            nn.init.trunc_normal_(self.Vp, std=std)
        else:  # head
            if head_rank is None:
                std = 0.02 * math.sqrt(M) if replace else 0.02
                self.W = nn.Parameter(torch.empty(M, dim, dim))
                nn.init.trunc_normal_(self.W, std=std)
            else:
                std = math.sqrt(0.02 * math.sqrt(M / head_rank)) if replace else 0.02
                self.P = nn.Parameter(torch.empty(M, dim, head_rank))
                self.Qm = nn.Parameter(torch.empty(M, head_rank, dim))
                nn.init.trunc_normal_(self.P, std=std)
                nn.init.trunc_normal_(self.Qm, std=std)

    def forward(self, x: torch.Tensor):
        t = self.w_r(x)
        if self.p is not None:
            t = t + self.p.materialize()[None]
        logits = t @ self.E.t() / math.sqrt(self.d_route)  # [B, N, M]
        probs = logits.softmax(dim=-1)  # full softmax: aux losses + Switch P-term

        if self.route_kind == "softmax":
            a = probs
        elif self.route_kind == "sigmoid":
            a = logits.sigmoid() / self.M  # mean-of-payloads semantics
        else:  # capacity-free dense top-k
            idx = logits.topk(self.k, dim=-1).indices
            mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, idx, True)
            a = logits.masked_fill(~mask, float("-inf")).softmax(dim=-1)

        aux = None
        if self.aux_entropy > 0:
            load = a.mean(dim=(0, 1))
            pl = load / (load.sum() + 1e-9)
            ent = -(pl * (pl + 1e-9).log()).sum()
            aux = self.aux_entropy * (math.log(self.M) - ent)  # maximize entropy of average routing
        if self.aux_load_balance > 0 and self.route_kind == "topk":
            top1 = logits.argmax(dim=-1)
            f = torch.nn.functional.one_hot(top1, self.M).float().mean(dim=(0, 1))  # dispatch fracs
            lb = self.M * (f * probs.mean(dim=(0, 1))).sum()  # Switch-style
            aux = lb * self.aux_load_balance if aux is None else aux + self.aux_load_balance * lb

        if self.payload_kind == "vector":
            delta = a @ self.U
        elif self.payload_kind == "rank1":
            s = x @ self.Vp.t()  # [B, N, M]
            delta = (a * s) @ self.U
        elif self.head_rank is None:
            delta = torch.einsum("bnm,mde,bne->bnd", a, self.W, x)
        else:
            tmp = torch.einsum("bne,mre->bnmr", x, self.Qm)  # [B, N, M, rho]
            delta = torch.einsum("bnm,bnmr,mdr->bnd", a, tmp, self.P)
        return delta, a, aux


@register_attention("metric_bank")
class MetricBankDQK(DynamicQKBase):
    def __init__(self, dim, num_heads, grid_size, *, num_latents: int = 16, d_route: int = 64,
                 route: str = "softmax", topk_k: int = 2, payload: str = "rank1",
                 head_rank: int | None = None, pos_prior: bool = False, flip_tie: bool = False,
                 aux_entropy: float = 0.0, aux_load_balance: float = 0.0, **kw):
        if route not in _ROUTES:
            raise ValueError(f"route must be one of {_ROUTES}, got {route!r}")
        if payload not in _PAYLOADS:
            raise ValueError(f"payload must be one of {_PAYLOADS}, got {payload!r}")
        if payload == "head" and head_rank is None and num_latents > 8:
            raise ValueError("head payload with full W_m needs M <= 8; set head_rank for larger M")
        super().__init__(dim, num_heads, grid_size, **kw)
        sides = ["q"] if self.mode == "replace" else [s for s in ("q", "k") if s in self.apply_to]
        self.gen = nn.ModuleDict(
            {
                s: _MetricBank(
                    dim, self.grid_size, num_latents=num_latents, d_route=d_route, route=route,
                    topk_k=topk_k, payload=payload, head_rank=head_rank, pos_prior=pos_prior,
                    flip_tie=flip_tie, aux_entropy=aux_entropy, aux_load_balance=aux_load_balance,
                    replace=self.mode == "replace",
                )
                for s in sides
            }
        )
        for s in sides:
            self._no_wd.append(f"gen.{s}.E")
            if pos_prior:
                self._no_wd.append(f"gen.{s}.p.weight")

    def _delta(self, side: str, x: torch.Tensor) -> torch.Tensor:
        delta, a, aux = self.gen[side](x)
        if aux is not None:
            self.last_aux_loss = aux if self.last_aux_loss is None else self.last_aux_loss + aux
        if self.collect_stats:
            ad = a.detach()
            an = ad / (ad.sum(dim=-1, keepdim=True) + 1e-9)
            self.stats[f"routing_{side}"] = ad
            self.stats[f"routing_entropy_{side}"] = -(an * (an + 1e-9).log()).sum(dim=-1)
            self.stats[f"load_{side}"] = ad.mean(dim=(0, 1))
        return delta

    def delta_q(self, x, *, base, ctx):
        return self._delta("q", x)

    def delta_k(self, x, *, base, ctx):
        return self._delta("k", x)


STATS_SCHEMA["metric_bank"] = {
    "routing_q": ("B", "N", "M"),
    "routing_entropy_q": ("B", "N"),
    "load_q": ("M",),
    "routing_k": ("B", "N", "M"),
    "routing_entropy_k": ("B", "N"),
    "load_k": ("M",),
}

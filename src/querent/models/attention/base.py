"""DynamicQKBase — the single attention forward every arm runs.

Frozen contracts (tests enforce all of these):

1. Constructor: ``AttnCls(dim, num_heads, grid_size=(gh, gw), **arm_kwargs)``;
   forward ``(B, N, D) -> (B, N, D)``, asserts ``N == gh * gw``. Fixed
   resolution is a contract — per-position checkpoints are resolution-locked.
2. Single forward: subclasses implement the hooks ``prepare`` / ``delta_q`` /
   ``delta_k`` and NEVER override ``forward``. Every arm therefore runs the
   identical code path around its delta. The residual add is the single
   expression ``base + alpha * delta`` — never in-place, never folded into
   ``w_q``.
3. alpha=0 equivalence: every residual-mode variant loads A0's ``state_dict``
   with ``strict=False`` and zero unexpected keys; with copied weights and
   alpha=0 the output equals A0 bitwise in fp32. Replace-mode arms (which set
   ``w_q = None``) are exempt; their contract is scale sanity at init:
   ``||q_replace|| / ||q_vanilla||`` in [0.5, 2].
4. Exactly one zero: alpha is the only zero-initialized parameter. Generators
   use trunc_normal(0.02) so the delta is O(1) at init and dL/dalpha != 0.
5. Stats: ``collect_stats`` is one of ``False | "light" | "full"``. When
   truthy, forward rebuilds ``self.stats`` (all values detached, on-device).
   "full" switches attention to the math path for that forward and exports
   exact ``attn_probs`` plus post-transform ``q``/``k`` (probe-only — training
   always stays on SDPA). ``False`` => ``self.stats is None`` and no stats
   work happens (zero hot-path overhead).
6. Q/K pipeline order (fixed): delta-add -> QK-RMSNorm (learned per-head
   gamma, fp32 internals) -> token_temp scale -> RoPE rotation.
7. Aux losses: ``self.last_aux_loss`` is reset to ``None`` each forward;
   routing arms may populate it (see B1). The trainer collects via
   ``querent.diagnostics.stats_api.collect_aux_losses``.

Weight-decay: parameters named by ``no_weight_decay()`` (alpha, QK-norm gamma,
per-position tables, w_temp, latent keys) must be excluded from decay — WD on
alpha is a spring pinning it at 0 and silently kills every variant.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .pos_utils import Rope2D

# Symbolic-shape schema for variant-specific "light"-tier stats keys.
# Symbols: B batch, N tokens, h heads, dh head_dim, r rank, M latents, R regions, k top-k.
# Variant modules add their entry at import time: STATS_SCHEMA["metric_bank"] = {...}.
STATS_SCHEMA: dict[str, dict[str, tuple[str, ...]]] = {}

# Common keys owned by the base class (present when applicable, any variant).
COMMON_STATS_LIGHT: dict[str, tuple[str, ...]] = {
    "alpha_q": ("A",),           # A = h (alpha_per_head) or 1
    "alpha_k": ("A",),
    "q_base_norm": ("B", "N"),   # ||W_Q x||_2 pre-QK-norm
    "q_delta_norm": ("B", "N"),  # ||alpha * delta_q||_2
    "delta_ratio_q": ("B", "N"),  # the meaningful influence dial (post-norm ||q|| is clamped)
    "k_base_norm": ("B", "N"),
    "k_delta_norm": ("B", "N"),
    "delta_ratio_k": ("B", "N"),
    "token_temp": ("B", "N"),
}
COMMON_STATS_FULL: dict[str, tuple[str, ...]] = {
    "q": ("B", "h", "N", "dh"),
    "k": ("B", "h", "N", "dh"),
    "attn_probs": ("B", "h", "N", "N"),
}

_STATS_LEVELS = (False, "light", "full")


class QKRMSNorm(nn.Module):
    """RMSNorm over head_dim with learned per-head scale gamma [h, d_h]; fp32 internals."""

    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # [B, h, N, d_h]
        dtype = t.dtype
        x = t.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * self.gamma.float()[None, :, None, :]
        return x.to(dtype)


class DynamicQKBase(nn.Module):
    registry_name = "base"

    def __init__(
        self,
        dim: int,
        num_heads: int,
        grid_size,
        *,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        apply_to: str = "q",
        mode: str = "residual",
        has_delta: bool = True,
        alpha_per_head: bool = True,
        alpha_init: float = 0.0,
        alpha_trainable: bool = True,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        token_temp: bool = False,
        rope: bool = False,
        rope_base: float = 100.0,
    ):
        super().__init__()
        gh, gw = grid_size
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        if apply_to not in ("q", "k", "qk"):
            raise ValueError(f"apply_to must be q|k|qk, got {apply_to!r}")
        if mode not in ("residual", "replace"):
            raise ValueError(f"mode must be residual|replace, got {mode!r}")
        if mode == "replace":
            if apply_to != "q":
                raise ValueError("replace mode is q-only")
            if not has_delta:
                raise ValueError("replace mode requires a delta-producing subclass")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.grid_size = (gh, gw)
        self.num_tokens = gh * gw
        self.apply_to = apply_to
        self.mode = mode
        self.has_delta = has_delta
        self.attn_drop_p = attn_drop
        self.use_qk_norm = qk_norm
        self.use_token_temp = token_temp

        # w_q is None in replace mode: replace arms must not carry dead vanilla
        # params that pollute param counts.
        self.w_q = None if mode == "replace" else nn.Linear(dim, dim, bias=qkv_bias)
        self.w_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

        if qk_norm:
            self.q_norm = QKRMSNorm(num_heads, self.head_dim, eps=qk_norm_eps)
            self.k_norm = QKRMSNorm(num_heads, self.head_dim, eps=qk_norm_eps)

        # alpha_init != 0 / alpha_trainable=False are ABLATION knobs ("forced-on"
        # trials: the mechanism must prove itself without the adoption dynamics).
        # Defaults preserve the alpha=0 equivalence contract.
        n_alpha = num_heads if alpha_per_head else 1
        if has_delta and mode == "residual":
            if "q" in apply_to:
                self.alpha_q = nn.Parameter(
                    torch.full((n_alpha,), float(alpha_init)), requires_grad=alpha_trainable
                )
            if "k" in apply_to:
                self.alpha_k = nn.Parameter(
                    torch.full((n_alpha,), float(alpha_init)), requires_grad=alpha_trainable
                )

        if token_temp:
            self.w_temp = nn.Parameter(torch.zeros(dim))  # s_i = 1 + tanh(w^T x) = 1 at init

        self.rope = Rope2D((gh, gw), self.head_dim, base=rope_base) if rope else None

        self.collect_stats: bool | str = False
        self.stats: dict[str, torch.Tensor] | None = None
        self.last_aux_loss: torch.Tensor | None = None

        self._no_wd: list[str] = [n for n in ("alpha_q", "alpha_k", "w_temp") if hasattr(self, n)]
        if qk_norm:
            self._no_wd += ["q_norm.gamma", "k_norm.gamma"]

        self._init_base_weights()

    # ------------------------------------------------------------------ hooks
    def prepare(self, x: torch.Tensor):
        """Computed once per forward, before either side; B2 builds region context here."""
        return None

    def delta_q(self, x: torch.Tensor, *, base: torch.Tensor | None, ctx) -> torch.Tensor | None:
        return None

    def delta_k(self, x: torch.Tensor, *, base: torch.Tensor | None, ctx) -> torch.Tensor | None:
        return None

    def _needs_math(self) -> bool:
        """Subclasses that must materialize logits (talking-heads) return True."""
        return False

    def _mix_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, d = x.shape
        if N != self.num_tokens:
            raise ValueError(f"expected N={self.num_tokens} tokens (grid {self.grid_size}), got {N}")
        self.stats = {} if self.collect_stats else None
        self.last_aux_loss = None

        ctx = self.prepare(x)
        q = self._make_side(x, "q", ctx)
        k = self._make_side(x, "k", ctx)
        v = self.w_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if self.collect_stats == "full":
            self.stats["q"] = q.detach()
            self.stats["k"] = k.detach()
        if self.collect_stats:
            for side in ("q", "k"):
                a = getattr(self, f"alpha_{side}", None)
                if a is not None:
                    self.stats[f"alpha_{side}"] = a.detach().clone()

        if self._use_sdpa():
            o = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop_p if self.training else 0.0
            )
        else:
            o = self._attend_math(q, k, v)

        o = o.transpose(1, 2).reshape(B, N, d)
        return self.proj_drop(self.proj(o))

    def _make_side(self, x: torch.Tensor, side: str, ctx) -> torch.Tensor:
        B, N, _ = x.shape
        if side == "q" and self.mode == "replace":
            full = self.delta_q(x, base=None, ctx=ctx)
        else:
            base = (self.w_q if side == "q" else self.w_k)(x)
            delta = None
            if self.has_delta and side in self.apply_to:
                hook = self.delta_q if side == "q" else self.delta_k
                delta = hook(x, base=base, ctx=ctx)
            if delta is not None:
                a = self._alpha_vec(side)
                scaled = a * delta
                full = base + scaled  # the single residual expression — never in-place
                if self.collect_stats:
                    self._record_delta_stats(side, base, scaled)
            else:
                full = base

        t = full.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        if self.use_qk_norm:
            t = (self.q_norm if side == "q" else self.k_norm)(t)
        if side == "q" and self.use_token_temp:
            s = 1 + torch.tanh(x @ self.w_temp)  # [B, N], in (0, 2), exactly 1 at init
            if self.collect_stats:
                self.stats["token_temp"] = s.detach()
            t = t * s.view(B, 1, N, 1)
        if self.rope is not None:
            t = self.rope(t)
        return t

    def _attend_math(self, q, k, v) -> torch.Tensor:
        # Same scale SDPA defaults to (1/sqrt(head_dim)).
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        logits = self._mix_logits(logits)
        a = logits.softmax(dim=-1)
        if self.collect_stats == "full":
            self.stats["attn_probs"] = a.detach()
        a = F.dropout(a, p=self.attn_drop_p, training=self.training)
        return a @ v

    # ---------------------------------------------------------------- helpers
    def _use_sdpa(self) -> bool:
        return not (self.collect_stats == "full" or self._needs_math())

    def _alpha_vec(self, side: str) -> torch.Tensor:
        a = getattr(self, f"alpha_{side}")
        # per-head [h] -> [d]; channel c belongs to head c // head_dim
        return a.repeat_interleave(self.head_dim) if a.numel() > 1 else a

    def _record_delta_stats(self, side: str, base: torch.Tensor, scaled: torch.Tensor) -> None:
        bn = base.detach().norm(dim=-1)
        dn = scaled.detach().norm(dim=-1)
        self.stats[f"{side}_base_norm"] = bn
        self.stats[f"{side}_delta_norm"] = dn
        self.stats[f"delta_ratio_{side}"] = dn / (bn + 1e-8)

    def set_collect_stats(self, level: bool | str) -> None:
        if level not in _STATS_LEVELS:
            raise ValueError(f"collect_stats must be one of {_STATS_LEVELS}, got {level!r}")
        self.collect_stats = level
        if not level:
            self.stats = None

    def no_weight_decay(self) -> set[str]:
        """Param names (relative to this module) that must be excluded from weight decay."""
        return set(self._no_wd)

    def _init_base_weights(self) -> None:
        for m in (self.w_q, self.w_k, self.w_v, self.proj):
            if m is not None:
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

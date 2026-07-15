"""Position-indexed parameter utilities for fixed-resolution attention.

Everything here indexes tokens row-major: i = row * gw + col.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class HalfGridParam(nn.Module):
    """Per-position parameter table [N, *tail], optionally tied under horizontal flip.

    With ``flip_tie=True`` the table is stored on a half-grid
    ``[gh, ceil(gw/2), *tail]`` and mirrored so that
    ``materialize().view(gh, gw, *tail).flip(1)`` equals itself. Autograd
    accumulates mirrored gradients into the shared storage automatically.

    Honesty note: tying makes the *module using it* hflip-equivariant, not the
    full model — the ViT's learned pos_emb stays untied.
    """

    def __init__(self, grid_size, tail_shape, *, flip_tie: bool = False, init: str = "zeros",
                 std: float = 0.02):
        super().__init__()
        gh, gw = grid_size
        self.grid_size = (gh, gw)
        self.tail_shape = tuple(tail_shape)
        self.flip_tie = flip_tie
        cols = math.ceil(gw / 2) if flip_tie else gw
        w = torch.zeros(gh, cols, *self.tail_shape)
        if init == "trunc_normal":
            nn.init.trunc_normal_(w, std=std)
        elif init != "zeros":
            raise ValueError(f"unknown init {init!r}")
        self.weight = nn.Parameter(w)

    def materialize(self) -> torch.Tensor:
        """Full per-position table, shape [gh*gw, *tail]. Static shapes; compile-safe."""
        gh, gw = self.grid_size
        w = self.weight
        if self.flip_tie:
            if gw % 2 == 0:
                full = torch.cat([w, w.flip(1)], dim=1)
            else:  # middle column is self-mirrored
                full = torch.cat([w, w[:, : gw // 2].flip(1)], dim=1)
        else:
            full = w
        return full.reshape(gh * gw, *self.tail_shape)


class Rope2D(nn.Module):
    """Axial 2D rotary embedding for fixed grids (GPT-NeoX half-split pairing).

    Per head dim ``d_h`` (must be divisible by 4): the first d_h//4 channel
    pairs rotate with the row coordinate, the next d_h//4 with the column.
    Channel c pairs with channel c + d_h//2. Angles are precomputed buffers
    (fixed N). ``base=100``: small grids want a higher minimum frequency than
    the LLM default 10000. Applied after QK-norm — rotation preserves norms.
    """

    def __init__(self, grid_size, head_dim: int, *, base: float = 100.0):
        super().__init__()
        gh, gw = grid_size
        if head_dim % 4 != 0:
            raise ValueError(f"Rope2D needs head_dim % 4 == 0, got {head_dim}")
        m = head_dim // 4
        freqs = base ** (-torch.arange(m, dtype=torch.float32) / m)
        rows = torch.arange(gh, dtype=torch.float32).repeat_interleave(gw)
        cols = torch.arange(gw, dtype=torch.float32).repeat(gh)
        ang = torch.cat([torch.outer(rows, freqs), torch.outer(cols, freqs)], dim=-1)  # [N, d_h/2]
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # t: [B, h, N, d_h]
        d2 = t.shape[-1] // 2
        t1, t2 = t[..., :d2], t[..., d2:]
        cos = self.cos.to(t.dtype)
        sin = self.sin.to(t.dtype)
        return torch.cat([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1)

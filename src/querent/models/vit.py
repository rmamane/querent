"""Compact fixed-resolution Vision Transformer with registry-injected attention.

Design points (frozen by the experiment plan):
- GAP head, NO CLS token — every token is positional (per-position parameters
  in attention arms index the full token grid).
- Learned absolute pos_emb (disable with pos_emb="none" for RoPE-only arms).
- Pre-norm blocks, per-sample stochastic depth (linearly spaced over depth).
- Attention modules fully own their init; the ViT's blanket init never
  descends into ``.attn`` subtrees (variant-specific init scales — e.g. TPA's
  sigma_A*sigma_B = 0.02*sqrt(r/d) — must not be clobbered).
- Fixed input size is a contract: forward asserts the spatial size.

Param-matching knob for capacity controls: fractional ``mlp_ratio``. Per-layer
MLP params ~= 2 * ratio * dim^2, so to match a variant adding P params/layer
use  delta_ratio = P / (2 * dim^2)  (e.g. B3's ~3d^2/layer => ratio 4.0 -> 5.5).

This module (and everything under querent/models/) imports torch only.
"""

from __future__ import annotations

import torch
from torch import nn

from .attention import build_attention
from .attention.base import DynamicQKBase


class PatchEmbed(nn.Module):
    """Conv patch embedding -> [B, N, D] in row-major token order (i = row*gw + col)."""

    def __init__(self, img_size: int, patch_size: int, in_chans: int, dim: int):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} not divisible by patch_size {patch_size}")
        self.img_size = img_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_tokens = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H != self.img_size or W != self.img_size:
            raise ValueError(f"expected {self.img_size}x{self.img_size} input, got {H}x{W}")
        return self.proj(x).flatten(2).transpose(1, 2)  # [B, N, D]


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DropPath(nn.Module):
    """Per-sample stochastic depth (scales by 1/keep_p at train time)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        mask = x.new_empty(x.shape[0], *([1] * (x.ndim - 1))).bernoulli_(keep)
        return x * mask / keep


class Block(nn.Module):
    def __init__(self, dim, num_heads, grid_size, *, mlp_ratio=4.0, drop_path=0.0,
                 proj_drop=0.0, attn_name="vanilla", attn_kwargs=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = build_attention(
            attn_name, dim=dim, num_heads=num_heads, grid_size=grid_size,
            proj_drop=proj_drop, **(attn_kwargs or {}),
        )
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(mlp_ratio * dim), drop=proj_drop)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 100,
        dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        proj_drop: float = 0.0,
        pos_emb: str = "learned",  # "learned" | "none" (RoPE-only arms)
        attn_name: str = "vanilla",
        attn_kwargs: dict | None = None,
    ):
        super().__init__()
        if pos_emb not in ("learned", "none"):
            raise ValueError(f"pos_emb must be learned|none, got {pos_emb!r}")
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, dim)
        self.grid_size = self.patch_embed.grid_size
        self.num_tokens = self.patch_embed.num_tokens

        self.pos_emb = (
            nn.Parameter(torch.zeros(1, self.num_tokens, dim)) if pos_emb == "learned" else None
        )

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            Block(
                dim, num_heads, self.grid_size, mlp_ratio=mlp_ratio, drop_path=dpr[i],
                proj_drop=proj_drop, attn_name=attn_name, attn_kwargs=attn_kwargs,
            )
            for i in range(depth)
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        self._init_non_attn_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_emb is not None:
            x = x + self.pos_emb
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x.mean(dim=1))  # GAP, no CLS

    def no_weight_decay(self) -> set[str]:
        out = {"pos_emb"} if self.pos_emb is not None else set()
        for i, blk in enumerate(self.blocks):
            attn = blk.attn
            if hasattr(attn, "no_weight_decay"):
                out |= {f"blocks.{i}.attn.{n}" for n in attn.no_weight_decay()}
        return out

    def _init_non_attn_weights(self) -> None:
        # Never descend into attention subtrees: attention modules own their init.
        for name, m in self.named_modules():
            if isinstance(m, DynamicQKBase) or ".attn" in name or name.endswith("attn"):
                continue
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if self.pos_emb is not None:
            nn.init.trunc_normal_(self.pos_emb, std=0.02)


def create_vit(model_cfg: dict, attn_name: str, attn_kwargs: dict | None = None,
               drop_path_rate: float | None = None) -> VisionTransformer:
    """Factory used by the trainer: model config dict + arm selection -> ViT."""
    cfg = dict(model_cfg)
    if drop_path_rate is not None:
        cfg["drop_path_rate"] = drop_path_rate
    return VisionTransformer(attn_name=attn_name, attn_kwargs=attn_kwargs, **cfg)

"""Attention-variant registry.

Frozen contract (consumed by configs, tests, and the ViT):

    AttnCls(dim: int, num_heads: int, grid_size: tuple[int, int], **arm_kwargs)
    forward: (B, N, D) -> (B, N, D), asserts N == gh * gw

Token order is row-major over the patch grid: i = row * gw + col — exactly what
``Conv2d -> flatten(2) -> transpose(1, 2)`` produces in PatchEmbed. RoPE and
flip-tying index against this convention.

Tests parametrize over ``names()`` so newly registered arms are auto-covered.
"""

from __future__ import annotations

from torch import nn

ATTENTION_REGISTRY: dict[str, type[nn.Module]] = {}


def register_attention(name: str):
    def deco(cls):
        if name in ATTENTION_REGISTRY:
            raise KeyError(f"attention variant {name!r} already registered")
        ATTENTION_REGISTRY[name] = cls
        cls.registry_name = name
        return cls

    return deco


def build_attention(name: str, *, dim: int, num_heads: int, grid_size, **kwargs) -> nn.Module:
    if name not in ATTENTION_REGISTRY:
        raise KeyError(f"unknown attention variant {name!r}; known: {sorted(ATTENTION_REGISTRY)}")
    return ATTENTION_REGISTRY[name](dim=dim, num_heads=num_heads, grid_size=tuple(grid_size), **kwargs)


def names() -> list[str]:
    return sorted(ATTENTION_REGISTRY)

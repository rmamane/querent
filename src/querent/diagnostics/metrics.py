"""Mechanism metrics, each unit-tested against closed forms.

Closed forms used by tests:
- uniform routing        -> entropy = log M, MI(position; latent) = 0
- one-hot positional     -> MI = log M (bijective positions->latents)
- isotropic Gaussian Q   -> participation ratio ~ d
- rank-1 Q               -> participation ratio ~ 1
"""

from __future__ import annotations

import math

import torch

EPS = 1e-9


def participation_ratio(rows: torch.Tensor, max_rows: int = 4096) -> float:
    """Effective rank of a [n, d] matrix: PR = (sum lambda)^2 / sum lambda^2."""
    x = rows.float()
    if x.shape[0] > max_rows:
        x = x[torch.randperm(x.shape[0], device=x.device)[:max_rows]]
    x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    lam = s.pow(2)
    return float(lam.sum().pow(2) / (lam.pow(2).sum() + EPS))


def q_participation_ratios(q: torch.Tensor) -> torch.Tensor:
    """Per-head PR of queries. q: [B, h, N, d_h] -> [h] (compare against d_h)."""
    B, h, N, dh = q.shape
    return torch.tensor([participation_ratio(q[:, i].reshape(B * N, dh)) for i in range(h)])


def routing_entropy(a: torch.Tensor) -> tuple[float, float]:
    """Mean per-token routing entropy. a: [..., M] rows (re-normalized defensively).
    Returns (nats, normalized-by-log-M)."""
    M = a.shape[-1]
    p = a / (a.sum(dim=-1, keepdim=True) + EPS)
    ent = -(p * (p + EPS).log()).sum(dim=-1).mean()
    return float(ent), float(ent / math.log(M))


def position_routing_mi(a: torch.Tensor) -> tuple[float, float]:
    """Mutual information between token position and routing assignment.

    a: [B, N, M] routing rows (each ~sums to 1). Positions get a uniform prior.
    High MI => routing is a positional lookup; low => content-driven.
    Returns (nats, normalized by log(min(M, N)))."""
    B, N, M = a.shape
    rows = a / (a.sum(dim=-1, keepdim=True) + EPS)
    joint = rows.mean(dim=0) / N  # [N, M], sums to 1
    p_pos = joint.sum(dim=1, keepdim=True)  # ~1/N
    p_lat = joint.sum(dim=0, keepdim=True)
    mi = (joint * ((joint + EPS) / (p_pos @ p_lat + EPS)).log()).sum()
    return float(mi), float(mi / math.log(min(M, N)))


def attention_entropy(probs: torch.Tensor) -> float:
    """Mean row entropy of attention. probs: [B, h, N, N]."""
    ent = -(probs * (probs + EPS).log()).sum(dim=-1)
    return float(ent.mean())


def grid_distance_matrix(grid_size, patch_px: float = 1.0) -> torch.Tensor:
    """[N, N] Euclidean distances between token centers (row-major order)."""
    gh, gw = grid_size
    rows = torch.arange(gh, dtype=torch.float32).repeat_interleave(gw)
    cols = torch.arange(gw, dtype=torch.float32).repeat(gh)
    pts = torch.stack([rows, cols], dim=1) * patch_px
    return torch.cdist(pts, pts)


def mean_attention_distance(probs: torch.Tensor, dist: torch.Tensor) -> float:
    """Attention-weighted mean distance. probs: [B, h, N, N]; dist: [N, N]."""
    return float((probs * dist.to(probs)).sum(dim=-1).mean())


def gate_position_content_split(g: torch.Tensor) -> dict[str, float]:
    """For gate/routing values [B, N, C]: how much variation is positional vs content?

    std_positions: std of per-position means (structure across the grid)
    std_content:   mean of per-position stds (variation across images at a position)
    Near-zero both => the mechanism degenerated to a constant."""
    per_pos_mean = g.float().mean(dim=0)  # [N, C]
    return {
        "mean": float(g.float().mean()),
        "std_positions": float(per_pos_mean.std(dim=0).mean()),
        "std_content": float(g.float().std(dim=0).mean()),
    }

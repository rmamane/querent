"""Exact attention-prob resolution from module stats (the preference chain).

Training uses SDPA — probabilities never exist. Under collect_stats="full" our
modules switch to the math path and export "attn_probs" directly (chain step 1
always hits for in-repo arms, exact including any logit mixing). Steps 2/3
exist for robustness with future arms that export logits or raw q/k only.
"""

from __future__ import annotations

import math

import torch


def resolve_attn_probs(stats: dict[str, torch.Tensor]) -> torch.Tensor | None:
    if "attn_probs" in stats:
        return stats["attn_probs"].float()
    if "attn_logits" in stats:
        return stats["attn_logits"].float().softmax(dim=-1)
    if "q" in stats and "k" in stats:
        q, k = stats["q"].float(), stats["k"].float()
        logits = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        if "attn_bias" in stats:
            logits = logits + stats["attn_bias"].float()
        return logits.softmax(dim=-1)
    return None

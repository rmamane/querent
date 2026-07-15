"""Closed-form checks for every probe metric."""

import math

import torch

from querent.diagnostics.attn_probs import resolve_attn_probs
from querent.diagnostics.metrics import (
    attention_entropy,
    gate_position_content_split,
    grid_distance_matrix,
    mean_attention_distance,
    participation_ratio,
    position_routing_mi,
    routing_entropy,
)


def test_uniform_routing_entropy_and_mi():
    a = torch.full((4, 16, 8), 1 / 8)
    nats, norm = routing_entropy(a)
    assert abs(nats - math.log(8)) < 1e-4
    assert abs(norm - 1.0) < 1e-4
    mi, mi_norm = position_routing_mi(a)
    assert abs(mi) < 1e-5 and abs(mi_norm) < 1e-5


def test_positional_onehot_mi_is_log_m():
    N = M = 16
    a = torch.eye(N).unsqueeze(0).repeat(3, 1, 1)  # position i -> latent i
    mi, mi_norm = position_routing_mi(a)
    assert abs(mi - math.log(M)) < 1e-3
    assert abs(mi_norm - 1.0) < 1e-3


def test_pr_isotropic_close_to_d():
    torch.manual_seed(0)
    pr = participation_ratio(torch.randn(4096, 32))
    assert pr > 25.0, pr


def test_pr_rank1_close_to_1():
    torch.manual_seed(0)
    x = torch.randn(4096, 1) @ torch.randn(1, 32) + 1e-4 * torch.randn(4096, 32)
    pr = participation_ratio(x)
    assert pr < 1.5, pr


def test_resolve_chain_prefers_probs_then_logits_then_qk():
    torch.manual_seed(0)
    q = torch.randn(2, 2, 8, 16)
    k = torch.randn(2, 2, 8, 16)
    manual = (q @ k.transpose(-2, -1) / math.sqrt(16)).softmax(-1)

    out = resolve_attn_probs({"q": q, "k": k})
    torch.testing.assert_close(out, manual, atol=1e-6, rtol=0)

    logits = torch.randn(2, 2, 8, 8)
    torch.testing.assert_close(resolve_attn_probs({"attn_logits": logits, "q": q, "k": k}),
                               logits.softmax(-1), atol=1e-6, rtol=0)

    probs = torch.rand(2, 2, 8, 8)
    torch.testing.assert_close(resolve_attn_probs({"attn_probs": probs, "attn_logits": logits}),
                               probs, atol=0, rtol=0)

    bias = torch.randn(2, 8, 8)
    torch.testing.assert_close(
        resolve_attn_probs({"q": q, "k": k, "attn_bias": bias}),
        (q @ k.transpose(-2, -1) / math.sqrt(16) + bias).softmax(-1), atol=1e-6, rtol=0)

    assert resolve_attn_probs({}) is None


def test_uniform_attention_metrics():
    N = 16
    probs = torch.full((1, 1, N, N), 1 / N)
    assert abs(attention_entropy(probs) - math.log(N)) < 1e-4
    dist = grid_distance_matrix((4, 4), patch_px=1.0)
    assert abs(mean_attention_distance(probs, dist) - float(dist.mean())) < 1e-5


def test_gate_split_directions():
    N, C = 16, 4
    pos_only = torch.linspace(0, 1, N)[None, :, None].expand(8, N, C)
    s = gate_position_content_split(pos_only)
    assert s["std_positions"] > 0.2 and s["std_content"] < 1e-6

    torch.manual_seed(0)
    content_only = torch.randn(8, 1, C).expand(8, N, C)
    s2 = gate_position_content_split(content_only)
    assert s2["std_content"] > 0.5 and s2["std_positions"] < 0.4

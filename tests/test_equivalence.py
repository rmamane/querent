"""THE linchpin test: with alpha = 0, every residual-mode variant IS vanilla A0.

Construction rules that make this structural (see base.py):
- variant loads A0's full state_dict with strict=False and ZERO unexpected keys
- alpha is zero at init, and `base + alpha*delta` is IEEE-exact `base` then
- the code path around the delta is byte-identical (single shared forward)

Replace-mode arms have no W_Q + alpha structure; their contract is scale
sanity instead: ||q_replace|| / ||q_vanilla|| in [0.5, 2] at init.
"""

import pytest
import torch
import torch.nn.functional as F
from conftest import TINY, tiny_x

from querent.models import VisionTransformer
from querent.models.attention import build_attention

RESIDUAL_CASES = [
    # gated_lowrank: full gate-mode x flip-tie grid
    *[("gated_lowrank", dict(gate_mode=m, rank=8, flip_tie=ft))
      for m in ("content", "position", "both") for ft in (False, True)],
    # tpa residual, with/without positional bias in the generators
    ("tpa", dict(mode="residual", tpa_rank=4, pos_bias="none")),
    ("tpa", dict(mode="residual", tpa_rank=4, pos_bias="ab")),
    ("head_remix", {}),
    # metric bank: routes x payloads x positional prior
    *[("metric_bank", dict(num_latents=8, d_route=32, route=r, payload=p, pos_prior=pp))
      for r in ("softmax", "sigmoid", "topk") for p in ("vector", "rank1") for pp in (False, True)],
    ("metric_bank", dict(num_latents=4, d_route=32, payload="head")),
    ("metric_bank", dict(num_latents=8, d_route=32, payload="head", head_rank=4)),
    ("regional", dict(region_grid=(2, 2), ctx_dim=32)),
    ("nested", {}),
    # apply_to coverage on representatives
    ("gated_lowrank", dict(gate_mode="both", rank=8, apply_to="k")),
    ("gated_lowrank", dict(gate_mode="both", rank=8, apply_to="qk")),
    ("metric_bank", dict(num_latents=8, d_route=32, payload="rank1", apply_to="qk")),
    ("regional", dict(region_grid=(2, 2), ctx_dim=32, apply_to="k")),
    # base-option coverage: rope / token_temp still equivalence-safe
    ("gated_lowrank", dict(gate_mode="content", rank=8, rope=True)),
    ("gated_lowrank", dict(gate_mode="content", rank=8, token_temp=True)),
]


@pytest.mark.parametrize("name,kw", RESIDUAL_CASES,
                         ids=[f"{n}-{i}" for i, (n, _) in enumerate(RESIDUAL_CASES)])
def test_alpha0_equals_a0(name, kw):
    shared = {k: kw[k] for k in ("rope", "token_temp") if k in kw}
    torch.manual_seed(0)
    a0 = build_attention("vanilla", **TINY, **shared)
    torch.manual_seed(1)  # different init RNG on purpose — copied weights must win
    var = build_attention(name, **TINY, **kw)

    missing, unexpected = var.load_state_dict(a0.state_dict(), strict=False)
    assert not unexpected, f"A0 params missing in {name}: {unexpected}"
    for aname in ("alpha_q", "alpha_k"):
        if hasattr(var, aname):
            assert getattr(var, aname).abs().max().item() == 0.0

    x = tiny_x()
    a0.eval()
    var.eval()
    with torch.no_grad():
        torch.testing.assert_close(var(x), a0(x), atol=1e-6, rtol=0)


def test_full_vit_equivalence():
    """Depth-2 whole-model case — catches Block-level wiring bugs."""
    common = dict(img_size=16, patch_size=4, num_classes=10, dim=64, depth=2, num_heads=2)
    torch.manual_seed(0)
    vit_a0 = VisionTransformer(**common)
    torch.manual_seed(1)
    vit_var = VisionTransformer(
        **common, attn_name="metric_bank",
        attn_kwargs=dict(num_latents=8, d_route=32, payload="rank1", pos_prior=True),
    )
    missing, unexpected = vit_var.load_state_dict(vit_a0.state_dict(), strict=False)
    assert not unexpected
    x = torch.randn(2, 3, 16, 16)
    vit_a0.eval()
    vit_var.eval()
    with torch.no_grad():
        torch.testing.assert_close(vit_var(x), vit_a0(x), atol=1e-6, rtol=0)


REPLACE_CASES = [
    ("tpa", dict(tpa_rank=4)),
    ("metric_bank", dict(num_latents=8, d_route=32, payload="vector")),
    ("metric_bank", dict(num_latents=8, d_route=32, payload="rank1")),
    ("metric_bank", dict(num_latents=4, d_route=32, payload="head")),
    ("metric_bank", dict(num_latents=8, d_route=32, payload="head", head_rank=4)),
]


@pytest.mark.parametrize("name,kw", REPLACE_CASES,
                         ids=[f"{n}-{i}" for i, (n, _) in enumerate(REPLACE_CASES)])
def test_replace_scale_sanity(name, kw):
    """Replace arms start near vanilla query scale (their equivalence surrogate)."""
    torch.manual_seed(0)
    a0 = build_attention("vanilla", **TINY)
    torch.manual_seed(1)
    var = build_attention(name, **TINY, mode="replace", **kw)
    assert var.w_q is None, "replace arms must not carry a dead w_q"

    x = F.layer_norm(tiny_x(batch=8), (TINY["dim"],))  # unit-RMS, like real block input
    with torch.no_grad():
        q_ref = a0.w_q(x)
        q_rep = var.delta_q(x, base=None, ctx=var.prepare(x))
    ratio = (q_rep.norm() / q_ref.norm()).item()
    assert 0.5 <= ratio <= 2.0, f"{name} replace-mode init scale off: ratio {ratio:.3f}"


def test_regional_replace_scale_wider_band():
    """Regional's context chain uses fan-in init, but pooling + attention
    averaging still shrink it somewhat — checked with an honest wider band."""
    torch.manual_seed(0)
    a0 = build_attention("vanilla", **TINY)
    torch.manual_seed(1)
    var = build_attention("regional", **TINY, mode="replace", region_grid=(2, 2), ctx_dim=32)
    x = F.layer_norm(tiny_x(batch=8), (TINY["dim"],))
    with torch.no_grad():
        ratio = (var.delta_q(x, base=None, ctx=var.prepare(x)).norm() / a0.w_q(x).norm()).item()
    assert 0.1 <= ratio <= 2.5, f"regional replace init scale off: {ratio:.4f}"


def test_regional_residual_delta_alive():
    """The residual-mode regional delta must be O(0.01+) of the base query at
    init — the three-linear chain with naive 0.02 init lands at ~1e-6 (dead)."""
    torch.manual_seed(0)
    m = build_attention("regional", **TINY, region_grid=(2, 2), ctx_dim=32)
    x = F.layer_norm(tiny_x(batch=8), (TINY["dim"],))
    with torch.no_grad():
        base = m.w_q(x)
        delta = m.delta_q(x, base=base, ctx=m.prepare(x))
    ratio = (delta.norm() / base.norm()).item()
    assert ratio > 0.02, f"regional residual delta degenerately small: {ratio:.5f}"

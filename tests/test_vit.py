"""ViT wiring: shapes, GAP/no-CLS, fixed-size assertion, init ownership, no-WD names."""

import pytest
import torch

from querent.models import VisionTransformer


def _tiny_vit(**kw):
    base = dict(img_size=16, patch_size=4, num_classes=10, dim=64, depth=2, num_heads=2)
    base.update(kw)
    return VisionTransformer(**base)


def test_forward_shape_and_no_cls():
    m = _tiny_vit()
    assert m.num_tokens == 16  # 4x4 grid, no CLS token
    out = m(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 10)


def test_wrong_input_size_raises():
    m = _tiny_vit()
    with pytest.raises(ValueError):
        m(torch.randn(1, 3, 32, 32))


def test_eval_deterministic_under_drop_path():
    m = _tiny_vit(drop_path_rate=0.5)
    m.eval()
    x = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(m(x), m(x), atol=0, rtol=0)


def test_no_weight_decay_names_resolve():
    m = _tiny_vit(attn_name="metric_bank",
                  attn_kwargs=dict(num_latents=8, d_route=32, pos_prior=True))
    params = dict(m.named_parameters())
    nwd = m.no_weight_decay()
    assert "pos_emb" in nwd
    assert any(".gen.q.E" in n for n in nwd)
    for n in nwd:
        assert n in params, f"no_weight_decay name {n!r} not a parameter"


def test_vit_tiny_param_count_sane():
    m = VisionTransformer(img_size=32, patch_size=4, num_classes=100,
                          dim=192, depth=12, num_heads=3)
    n = sum(p.numel() for p in m.parameters())
    assert 5.0e6 < n < 5.8e6, f"ViT-Ti/4 param count off: {n/1e6:.2f}M"


def test_attn_init_ownership():
    """The ViT blanket init must not clobber attention-owned init (e.g. TPA sigma)."""
    m = _tiny_vit(attn_name="tpa", attn_kwargs=dict(tpa_rank=4, mode="replace"))
    w = m.blocks[0].attn.gen["q"].w_a.weight
    # TPA sigma = sqrt(0.02 * sqrt(r/d)) ~ 0.0728 for r=4, d=64 — far from the 0.02 blanket.
    assert 0.05 < w.std().item() < 0.10, f"TPA init clobbered: std {w.std().item():.4f}"


def test_pos_emb_none():
    m = _tiny_vit(pos_emb="none", attn_name="vanilla", attn_kwargs=dict(rope=True))
    assert m.pos_emb is None
    assert m(torch.randn(1, 3, 16, 16)).shape == (1, 10)

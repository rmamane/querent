"""Flip-tying: (a) materialized tables are mirror-symmetric, (b) modules with
tied per-position params are hflip-equivariant at the delta level (module-level
only — the ViT pos_emb stays untied; documented honesty note)."""

import torch
from conftest import TINY, flip_cols, tiny_x

from querent.models.attention import build_attention
from querent.models.attention.pos_utils import HalfGridParam


def _assert_mirror(grid):
    hp = HalfGridParam(grid, (3,), flip_tie=True, init="trunc_normal")
    T = hp.materialize().view(*grid, 3)
    torch.testing.assert_close(T.flip(1), T, atol=0, rtol=0)


def test_table_mirror_even():
    _assert_mirror((4, 4))


def test_table_mirror_odd():
    _assert_mirror((3, 3))


def test_untied_table_shape_matches():
    hp = HalfGridParam((4, 4), (5,), flip_tie=False, init="trunc_normal")
    assert hp.materialize().shape == (16, 5)


def _delta_equivariant(module):
    x = tiny_x()
    with torch.no_grad():
        d_flip_in = module.delta_q(flip_cols(x), base=None, ctx=None)
        d_flip_out = flip_cols(module.delta_q(x, base=None, ctx=None))
    torch.testing.assert_close(d_flip_in, d_flip_out, atol=1e-6, rtol=0)


def test_gated_position_equivariance():
    torch.manual_seed(3)
    m = build_attention("gated_lowrank", **TINY, gate_mode="position", rank=4, flip_tie=True)
    # make the positional table non-trivial (zeros would pass vacuously)
    with torch.no_grad():
        m.gen["q"].b.weight.normal_(0, 0.5)
    _delta_equivariant(m)


def test_gated_both_equivariance():
    torch.manual_seed(4)
    m = build_attention("gated_lowrank", **TINY, gate_mode="both", rank=4, flip_tie=True)
    with torch.no_grad():
        m.gen["q"].b.weight.normal_(0, 0.5)
    _delta_equivariant(m)


def test_metric_bank_prior_equivariance():
    torch.manual_seed(5)
    m = build_attention("metric_bank", **TINY, num_latents=8, d_route=32,
                        payload="rank1", pos_prior=True, flip_tie=True)
    with torch.no_grad():
        m.gen["q"].p.weight.normal_(0, 0.5)
    _delta_equivariant(m)


def test_untied_position_is_not_equivariant():
    """Sanity that the test itself has teeth: untied tables must break equivariance."""
    torch.manual_seed(6)
    m = build_attention("gated_lowrank", **TINY, gate_mode="position", rank=4, flip_tie=False)
    with torch.no_grad():
        m.gen["q"].b.weight.normal_(0, 1.0)
    x = tiny_x()
    with torch.no_grad():
        d1 = m.delta_q(flip_cols(x), base=None, ctx=None)
        d2 = flip_cols(m.delta_q(x, base=None, ctx=None))
    assert not torch.allclose(d1, d2, atol=1e-4)

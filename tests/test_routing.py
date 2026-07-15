"""B1 routing semantics: simplex/k-hot/bounded-sigmoid + aux-loss plumbing."""

import torch
from conftest import TINY, tiny_x

from querent.models.attention import build_attention

M = 8


def _routing(route, **kw):
    m = build_attention("metric_bank", **TINY, num_latents=M, d_route=32,
                        route=route, payload="vector", **kw)
    m.set_collect_stats("light")
    m.eval()
    m(tiny_x())
    return m, m.stats["routing_q"]


def test_softmax_rows_sum_to_one():
    _, a = _routing("softmax")
    torch.testing.assert_close(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5, rtol=0)


def test_topk_rows_exactly_k_hot():
    m, a = _routing("topk", topk_k=2)
    nz = (a > 0).sum(-1)
    assert (nz == 2).all(), "top-2 routing must be exactly 2-hot"
    torch.testing.assert_close(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5, rtol=0)


def test_sigmoid_bounded_by_inv_M():
    _, a = _routing("sigmoid")
    assert (a > 0).all() and (a < 1.0 / M).all(), "sigmoid routing must divide by M"


def test_aux_losses_off_by_default():
    m, _ = _routing("softmax")
    assert m.last_aux_loss is None


def test_entropy_aux_nonnegative_and_present():
    m, _ = _routing("softmax", aux_entropy=0.1)
    assert m.last_aux_loss is not None
    assert m.last_aux_loss.item() >= -1e-6  # log M - H(mean routing) >= 0


def test_load_balance_aux_topk():
    m, _ = _routing("topk", topk_k=2, aux_load_balance=0.01)
    assert m.last_aux_loss is not None
    assert m.last_aux_loss.item() > 0


def test_aux_accumulates_across_sides():
    m = build_attention("metric_bank", **TINY, num_latents=M, d_route=32,
                        payload="vector", apply_to="qk", aux_entropy=0.1)
    m.eval()
    m(tiny_x())
    assert m.last_aux_loss is not None  # q-side + k-side both contribute

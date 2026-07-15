"""Every registered arm: forward shape, finite grads, and a LIVE alpha gradient.

The alpha-grad check is the arm-not-dead check: a zero-init generator would
make dL/dalpha = 0 forever while looking like "the variant learned nothing".
Parametrized over the registry, so new arms are auto-covered.
"""

import pytest
import torch
from conftest import CANONICAL_KWARGS, TINY, tiny_x

from querent.models.attention import build_attention, names


@pytest.mark.parametrize("name", names())
def test_fwd_bwd_and_alpha_alive(name):
    m = build_attention(name, **TINY, **CANONICAL_KWARGS.get(name, {}))
    x = tiny_x()
    out = m(x)
    assert out.shape == x.shape

    out.sum().backward()
    for pname, p in m.named_parameters():
        assert p.grad is not None, f"{name}: no grad for {pname}"
        assert torch.isfinite(p.grad).all(), f"{name}: non-finite grad in {pname}"

    if hasattr(m, "alpha_q"):
        assert m.alpha_q.grad.abs().sum().item() > 0, f"{name}: dead delta path (alpha grad = 0)"


@pytest.mark.parametrize("apply_to", ["k", "qk"])
def test_k_side_deltas(apply_to):
    m = build_attention("gated_lowrank", **TINY, gate_mode="both", rank=4, apply_to=apply_to)
    out = m(tiny_x())
    out.sum().backward()
    assert m.alpha_k.grad.abs().sum().item() > 0
    if apply_to == "qk":
        assert m.alpha_q.grad.abs().sum().item() > 0


def test_no_weight_decay_names_exist():
    for name in names():
        m = build_attention(name, **TINY, **CANONICAL_KWARGS.get(name, {}))
        params = dict(m.named_parameters())
        for nwd in m.no_weight_decay():
            assert nwd in params, f"{name}: no_weight_decay names unknown param {nwd!r}"

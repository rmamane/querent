"""SDPA-vs-math agreement, and talking-heads' identity-init == plain vanilla."""

import torch
from conftest import TINY, tiny_x

from querent.models.attention import build_attention


def test_sdpa_equals_math_path():
    m = build_attention("vanilla", **TINY)
    m.eval()
    x = tiny_x()
    with torch.no_grad():
        o_sdpa = m(x)
        m.set_collect_stats("full")  # forces the math path
        o_math = m(x)
    torch.testing.assert_close(o_sdpa, o_math, atol=1e-5, rtol=1e-5)


def test_talking_heads_identity_equals_vanilla():
    torch.manual_seed(0)
    plain = build_attention("vanilla", **TINY)
    torch.manual_seed(1)
    th = build_attention("vanilla", **TINY, talking_heads=True)
    missing, unexpected = th.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected
    torch.testing.assert_close(th.w_th, torch.eye(TINY["num_heads"]))  # identity init

    plain.eval()
    th.eval()
    x = tiny_x()
    with torch.no_grad():
        torch.testing.assert_close(th(x), plain(x), atol=1e-5, rtol=1e-5)


def test_talking_heads_nonidentity_differs():
    m = build_attention("vanilla", **TINY, talking_heads=True)
    m.eval()
    x = tiny_x()
    with torch.no_grad():
        o_id = m(x)
        m.w_th.copy_(torch.tensor([[0.5, 0.5], [0.5, 0.5]]))
        o_mix = m(x)
    assert not torch.allclose(o_id, o_mix, atol=1e-5)

"""Stats-contract conformance: off => None; light/full => detached tensors
matching STATS_SCHEMA with bound symbolic shapes."""

import pytest
import torch
from conftest import CANONICAL_KWARGS, TINY, tiny_x

from querent.models.attention import (
    COMMON_STATS_FULL,
    COMMON_STATS_LIGHT,
    STATS_SCHEMA,
    build_attention,
    names,
)

# Symbol bindings for the canonical tiny configs (see CANONICAL_KWARGS).
SYMBOLS = {"B": 2, "N": 16, "h": 2, "dh": 32, "r": 4, "M": 8, "R": 4, "k": 2, "A": 2}

# Variant-specific keys expected to APPEAR for the canonical config (q-side arms).
EXPECTED_VARIANT_KEYS = {
    "vanilla": set(),
    "gated_lowrank": {"gates_q"},
    "tpa": {"tpa_a_fro_q", "tpa_b_fro_q"},
    "head_remix": {"remix"},
    "metric_bank": {"routing_q", "routing_entropy_q", "load_q"},
    "regional": {"region_attn"},
    "nested": {"inner_entropy"},
}


def _check_shapes(name, stats):
    schema = {**COMMON_STATS_LIGHT, **COMMON_STATS_FULL, **STATS_SCHEMA[name]}
    for key, t in stats.items():
        assert key in schema, f"{name}: stats key {key!r} not in schema"
        expected = tuple(SYMBOLS[s] for s in schema[key])
        assert tuple(t.shape) == expected, f"{name}.{key}: {tuple(t.shape)} != {expected}"
        assert t.grad_fn is None, f"{name}.{key} not detached"


@pytest.mark.parametrize("name", names())
def test_stats_levels(name):
    m = build_attention(name, **TINY, **CANONICAL_KWARGS.get(name, {}))
    m.eval()
    x = tiny_x()

    # off: no stats work at all
    m(x)
    assert m.stats is None

    # light
    m.set_collect_stats("light")
    m(x)
    assert m.stats is not None
    missing = EXPECTED_VARIANT_KEYS.get(name, set()) - set(m.stats)
    assert not missing, f"{name}: light-tier keys missing: {missing}"
    assert "attn_probs" not in m.stats and "q" not in m.stats
    if name != "vanilla":  # every delta arm records the influence dial
        assert "delta_ratio_q" in m.stats
    _check_shapes(name, m.stats)

    # full: exact attention probs (math path) + post-transform q/k
    m.set_collect_stats("full")
    m(x)
    for key in ("attn_probs", "q", "k"):
        assert key in m.stats, f"{name}: full tier missing {key}"
    probs = m.stats["attn_probs"]
    torch.testing.assert_close(probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=1e-5, rtol=0)
    _check_shapes(name, m.stats)

    # back off: stats cleared
    m.set_collect_stats(False)
    assert m.stats is None


def test_schema_declared_for_all_variants():
    for name in names():
        assert name in STATS_SCHEMA, f"{name} has no STATS_SCHEMA entry"

import pytest
import torch

# Tiny geometry used across the suite: 4x4 grid (N=16), dim 64, 2 heads (d_h=32,
# divisible by 4 for RoPE). Small enough for CPU, real enough to exercise grids.
TINY = dict(dim=64, num_heads=2, grid_size=(4, 4))
N_TINY = 16

# Canonical kwargs per registered arm (used by contract tests that parametrize
# over the whole registry). New arms without an entry run with their defaults.
CANONICAL_KWARGS = {
    "vanilla": {},
    "gated_lowrank": dict(gate_mode="both", rank=4),
    "tpa": dict(tpa_rank=4),
    "head_remix": {},
    "metric_bank": dict(num_latents=8, d_route=32, payload="rank1"),
    "regional": dict(region_grid=(2, 2), ctx_dim=32),
    "nested": {},
}


@pytest.fixture(autouse=True)
def _cpu_and_seed():
    torch.manual_seed(0)
    yield


def tiny_x(batch: int = 2) -> torch.Tensor:
    return torch.randn(batch, N_TINY, TINY["dim"])


def flip_cols(x: torch.Tensor, grid=(4, 4)) -> torch.Tensor:
    """Horizontally flip a row-major token sequence [B, N, D]."""
    B, N, D = x.shape
    gh, gw = grid
    return x.view(B, gh, gw, D).flip(2).reshape(B, N, D)

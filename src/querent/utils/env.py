"""Device/runtime setup and the MPS guard rails.

MPS pitfalls handled here (dev Mac runs smoke tests only):
- bf16-mixed is unreliable on MPS -> downgrade to 32-true with a warning.
- torch.compile is CUDA-only in this project.
- PYTORCH_ENABLE_MPS_FALLBACK=1 so exotic-arm ops fall back to CPU in smoke.

On CUDA: TF32 on (affects all arms equally; part of the frozen recipe) and
cudnn benchmark (fixed shapes).
"""

from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger(__name__)


def resolve_accelerator(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "gpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def setup_runtime(cfg) -> tuple[str, str]:
    """Returns (accelerator, precision), possibly downgraded for the platform."""
    accelerator = resolve_accelerator(cfg.trainer.accelerator)
    precision = str(cfg.trainer.precision)

    if accelerator == "gpu":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    else:
        if accelerator == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        if "bf16" in precision or "16" in precision:
            log.warning("precision %s unsupported on %s — using 32-true", precision, accelerator)
            precision = "32-true"

    return accelerator, precision


def compile_allowed(cfg, accelerator: str) -> bool:
    return bool(cfg.compile.enabled) and accelerator == "gpu"

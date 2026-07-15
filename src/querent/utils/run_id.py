"""Deterministic, human-readable run identity.

Single source of truth for the {phase}_{arm}_s{seed} scheme; the YAML
interpolation in configs/config.yaml mirrors this and a unit test keeps the
two in sync. The run id doubles as the wandb run id (resume="allow"), so a
relaunched vast.ai instance continues the SAME wandb run. ``run_suffix``
(e.g. "-r2") is the escape hatch to intentionally start a cell over.
"""

from __future__ import annotations

import re

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def exp_name(phase: str, arm_name: str, seed: int) -> str:
    return f"{phase}_{arm_name}_s{seed}"


def run_id(phase: str, arm_name: str, seed: int, run_suffix: str = "") -> str:
    rid = exp_name(phase, arm_name, seed) + run_suffix
    if not _RUN_ID_RE.match(rid):
        raise ValueError(f"run id {rid!r} is not wandb-safe")
    return rid

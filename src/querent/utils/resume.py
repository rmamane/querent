"""Preemption-safe resume resolution (exact order, see plan):

1. ``fit_completed`` wandb-summary guard -> the cell is done; re-running the
   onstart (vast replays it on every restart!) must be a no-op.
2. local ``last.ckpt``      -> same-instance stop/restart.
3. wandb artifact ``:latest`` -> instance destroyed; fresh box resumes the run.
4. fresh start.

Offline boxes: the wandb API raises, we fall through to the local tier —
artifact resume needs connectivity by definition.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

COMPLETED = "COMPLETED"


def resolve_resume(cfg) -> str | None:
    run = None
    api = None
    entity = None
    try:
        import wandb

        api = wandb.Api(timeout=20)
        entity = cfg.wandb.entity or api.default_entity
        run = api.run(f"{entity}/{cfg.wandb.project}/{cfg.run_id}")
    except Exception:  # never started, offline, or no API key — all fine
        run = None

    if run is not None and run.summary.get("fit_completed"):
        log.info("run %s already completed — nothing to do", cfg.run_id)
        return COMPLETED

    local = Path(str(cfg.ckpt.dir)) / "last.ckpt"
    if local.exists():
        log.info("resuming from local %s", local)
        return str(local)

    if run is not None and api is not None:
        try:
            art = api.artifact(f"{entity}/{cfg.wandb.project}/ckpt-{cfg.run_id}:latest")
            root = art.download(root=str(cfg.ckpt.dir))
            p = Path(root) / "last.ckpt"
            if p.exists():
                log.info("resuming from artifact ckpt-%s:latest", cfg.run_id)
                return str(p)
        except Exception as e:  # no artifact yet
            log.info("no checkpoint artifact for %s (%s) — fresh start", cfg.run_id, type(e).__name__)

    return None

"""Periodic checkpoint-artifact upload — the cross-instance resume fallback.

``last.ckpt`` on local disk dies with a destroyed instance; the wandb artifact
``ckpt-{run_id}:latest`` survives. Skipped in offline mode (artifacts need
connectivity; offline boxes rely on same-disk resume only).
"""

from __future__ import annotations

import logging
from pathlib import Path

import lightning.pytorch as pl

log = logging.getLogger(__name__)


class CkptArtifactCallback(pl.Callback):
    def __init__(self, cfg):
        self.every = int(cfg.ckpt.artifact_every_epochs)
        self.ckpt_dir = Path(str(cfg.ckpt.dir))
        self.run_id = str(cfg.run_id)

    def _upload(self, trainer, epoch: int) -> None:
        if trainer.logger is None or not hasattr(trainer.logger, "experiment"):
            return
        import wandb

        exp = trainer.logger.experiment
        if getattr(exp, "offline", False) or wandb.run is None or wandb.run.offline:
            return
        last = self.ckpt_dir / "last.ckpt"
        if not last.exists():
            return
        art = wandb.Artifact(f"ckpt-{self.run_id}", type="checkpoint")
        art.add_file(str(last))
        exp.log_artifact(art, aliases=["latest", f"ep{epoch}"])
        log.info("uploaded ckpt artifact for %s at epoch %d", self.run_id, epoch)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.fast_dev_run or trainer.sanity_checking:
            return
        ep = int(trainer.current_epoch)
        if ep > 0 and ep % self.every == 0:
            self._upload(trainer, ep)

    def on_fit_end(self, trainer, pl_module):
        if not trainer.fast_dev_run:
            self._upload(trainer, int(trainer.current_epoch))

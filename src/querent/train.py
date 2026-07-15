"""Hydra entry point: resume resolution -> logger -> trainer -> fit.

Canonical invocation (works verbatim from a fresh clone):
    python train.py experiment=p1/a3 seed=0
"""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from .callbacks.ckpt_artifact import CkptArtifactCallback
from .callbacks.consistency import ConsistencyCallback
from .callbacks.corruption import CorruptionEvalCallback
from .callbacks.probe import ProbeCallback
from .callbacks.stats_harvest import StatsHarvestCallback
from .data.cifar100 import CIFAR100DataModule
from .efficiency.flops import count_flops, param_breakdown
from .training.lit_module import ClassificationLitModule
from .utils.env import setup_runtime
from .utils.resume import COMPLETED, resolve_resume


def build_callbacks(cfg) -> list:
    ckpt_dir = str(cfg.ckpt.dir)
    return [
        StatsHarvestCallback(every_steps=int(cfg.diagnostics.harvest_every_steps)),
        ProbeCallback(cfg),
        ConsistencyCallback(cfg),
        CorruptionEvalCallback(cfg),
        CkptArtifactCallback(cfg),
        # rolling: last.ckpt refreshed every epoch (preemption loses <= 1 epoch)
        ModelCheckpoint(dirpath=ckpt_dir, save_last=True, save_top_k=0, every_n_epochs=1),
        # periodic archive
        ModelCheckpoint(
            dirpath=ckpt_dir, filename="ep{epoch:03d}", auto_insert_metric_name=False,
            every_n_epochs=int(cfg.ckpt.periodic_every_epochs), save_top_k=-1,
        ),
        # best on val accuracy
        ModelCheckpoint(
            dirpath=ckpt_dir, filename="best", monitor="val/acc1", mode="max", save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]


def build_logger(cfg) -> WandbLogger:
    return WandbLogger(
        project=str(cfg.wandb.project),
        entity=cfg.wandb.entity,
        name=str(cfg.exp_name),
        version=str(cfg.run_id),  # wandb run id == run_id -> preemption-safe resume
        group=str(cfg.wandb.group),
        job_type=str(cfg.wandb.job_type),
        offline=str(cfg.wandb.mode) == "offline",
        save_dir="outputs",
        resume="allow",
    )


# Absolute anchor: src/querent/train.py -> repo root /configs. Works for the
# editable install everywhere (dev Mac and the Docker clone at /workspace/querent);
# this project is never installed as a non-editable wheel.
CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(int(cfg.seed), workers=True)
    accelerator, precision = setup_runtime(cfg)

    module = ClassificationLitModule(cfg)
    datamodule = CIFAR100DataModule(cfg)

    fast = bool(cfg.trainer.fast_dev_run)
    logger = False if fast else build_logger(cfg)
    callbacks = [] if fast else build_callbacks(cfg)
    if isinstance(logger, WandbLogger):
        # Epoch-cadence metrics get their own axis — mixing them with the
        # global-step axis trips wandb's monotonic-step drops.
        exp = logger.experiment
        for prefix in ("probe", "probe_img", "consist", "final"):
            exp.define_metric(f"{prefix}/*", step_metric="epoch")
        # Efficiency accounting, once, on the CPU model before the trainer moves it.
        eff = param_breakdown(module.net)
        eff.update(count_flops(module.net, int(cfg.model.img_size)))
        for k, v in eff.items():
            exp.summary[k] = v

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=cfg.trainer.devices,
        precision=precision,
        max_epochs=int(cfg.recipe.epochs),
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        fast_dev_run=fast,
        callbacks=callbacks,
        logger=logger,
        deterministic=False,  # routing ops lack deterministic kernels; see plan
    )

    ckpt_path = None
    if not fast:
        resolved = resolve_resume(cfg)
        if resolved == COMPLETED:
            print(f"run {cfg.run_id} already completed — exiting (idempotent onstart)")
            return
        ckpt_path = resolved

    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)

    if not fast and isinstance(logger, WandbLogger):
        logger.experiment.summary["fit_completed"] = True
        logger.experiment.summary["config_yaml"] = OmegaConf.to_yaml(cfg, resolve=True)

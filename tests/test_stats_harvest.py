"""Regression: harvested stats must actually REACH the logger.

The original implementation used pl_module.log_dict, whose flush cadence
((step+1) % log_every_n_steps == 0) never coincides with harvest steps
(multiples of K) — values were buffered and silently dropped.
"""

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from querent.callbacks.stats_harvest import StatsHarvestCallback
from querent.training.lit_module import ClassificationLitModule


class CaptureLogger(pl.loggers.logger.Logger):
    def __init__(self):
        super().__init__()
        self.metrics: list[dict] = []

    @property
    def name(self):
        return "capture"

    @property
    def version(self):
        return "0"

    def log_metrics(self, metrics, step=None):
        self.metrics.append(dict(metrics))

    def log_hyperparams(self, *a, **k):
        pass


def _tiny_cfg():
    return OmegaConf.create({
        "model": {"img_size": 16, "patch_size": 4, "num_classes": 10, "dim": 64,
                  "depth": 2, "num_heads": 2, "mlp_ratio": 2.0, "pos_emb": "learned"},
        "arm": {"name": "b1", "attention": {"name": "metric_bank",
                "kwargs": {"num_latents": 8, "d_route": 32, "payload": "rank1"}}},
        "recipe": {"drop_path_rate": 0.0, "label_smoothing": 0.0, "epochs": 1,
                   "batch_size": 4, "aux_loss_start_epoch": 0,
                   "mixup": {"mixup_alpha": 0.0, "cutmix_alpha": 0.0, "prob": 0.0,
                             "switch_prob": 0.5, "mode": "batch"},
                   "optimizer": {"lr": 1e-3, "weight_decay": 0.0,
                                 "betas": [0.9, 0.999], "eps": 1e-8},
                   "schedule": {"warmup_epochs": 0, "min_lr": 1e-5}},
        "data": {"num_classes": 10},
        "compile": {"enabled": False, "mode": "default"},
    })


def test_harvest_reaches_logger():
    module = ClassificationLitModule(_tiny_cfg())
    ds = TensorDataset(torch.randn(8, 3, 16, 16), torch.randint(0, 10, (8,)))
    logger = CaptureLogger()
    trainer = pl.Trainer(
        accelerator="cpu", max_steps=2, logger=logger, limit_val_batches=0,
        callbacks=[StatsHarvestCallback(every_steps=100)],  # harvests at step 0 only
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0,
        log_every_n_steps=50,  # the cadence that used to swallow the harvest
    )
    trainer.fit(module, train_dataloaders=DataLoader(ds, batch_size=4))

    stats_keys = {k for m in logger.metrics for k in m if k.startswith("stats/")}
    assert any("routing_q_mean" in k for k in stats_keys), stats_keys
    assert any("alpha_q_absmax" in k for k in stats_keys), stats_keys
    # collect flag must be off again after harvest
    assert all(m.collect_stats is False for _, m in
               __import__("querent.diagnostics.stats_api", fromlist=["iter_stat_modules"])
               .iter_stat_modules(module.net))

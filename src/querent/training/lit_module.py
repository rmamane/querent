"""ClassificationLitModule — the one LightningModule for every arm.

Compile aliasing contract: ``torch.compile`` wraps ``self._fwd`` — never the
LightningModule, never reassigning ``self.net`` — so checkpoint keys stay
clean and resume works with compile on or off. Diagnostics probes always run
the uncompiled ``self.net``.
"""

from __future__ import annotations

import torch
import torchmetrics
from lightning import pytorch as pl
from omegaconf import OmegaConf
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from torch import nn

from ..diagnostics.stats_api import collect_aux_losses
from ..models.vit import create_vit
from .optim import build_param_groups, warmup_cosine


class ClassificationLitModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        if isinstance(cfg, dict):  # load_from_checkpoint round-trip
            cfg = OmegaConf.create(cfg)
        # Stored under the "cfg" key so load_from_checkpoint calls cls(cfg=...)
        # instead of splatting every config key as a constructor kwarg.
        self.save_hyperparameters({"cfg": OmegaConf.to_container(cfg, resolve=True)})
        self.cfg = cfg

        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        attn_kwargs = OmegaConf.to_container(cfg.arm.attention.kwargs, resolve=True) or {}
        self.net = create_vit(
            model_cfg, cfg.arm.attention.name, attn_kwargs,
            drop_path_rate=float(cfg.recipe.drop_path_rate),
        )
        self._fwd = self.net  # possibly-compiled alias; self.net stays the source of truth

        mix = cfg.recipe.mixup
        use_mixup = float(mix.prob) > 0 and (float(mix.mixup_alpha) > 0 or float(mix.cutmix_alpha) > 0)
        self.mixup = (
            Mixup(
                mixup_alpha=float(mix.mixup_alpha), cutmix_alpha=float(mix.cutmix_alpha),
                prob=float(mix.prob), switch_prob=float(mix.switch_prob), mode=str(mix.mode),
                label_smoothing=float(cfg.recipe.label_smoothing),
                num_classes=int(cfg.data.num_classes),
            )
            if use_mixup
            else None
        )
        self.train_loss = (
            SoftTargetCrossEntropy()
            if self.mixup
            else nn.CrossEntropyLoss(label_smoothing=float(cfg.recipe.label_smoothing))
        )
        self.val_loss = nn.CrossEntropyLoss()

        C = int(cfg.data.num_classes)
        self.acc1 = torchmetrics.Accuracy(task="multiclass", num_classes=C, top_k=1)
        self.acc5 = torchmetrics.Accuracy(task="multiclass", num_classes=C, top_k=5)
        self.best_acc1 = torchmetrics.MaxMetric()

    # ------------------------------------------------------------------ setup
    def setup(self, stage: str) -> None:
        if bool(self.cfg.compile.enabled) and torch.cuda.is_available():
            self._fwd = torch.compile(self.net, mode=str(self.cfg.compile.mode))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fwd(x)

    # ------------------------------------------------------------------ steps
    def training_step(self, batch, batch_idx: int):
        x, y = batch
        if self.mixup is not None:
            x, y = self.mixup(x, y)  # on-device; y becomes soft targets
        logits = self._fwd(x)
        loss = self.train_loss(logits, y)

        aux = collect_aux_losses(self.net)
        if aux is not None:
            if self.current_epoch >= int(self.cfg.recipe.aux_loss_start_epoch):
                loss = loss + aux
            self.log("train/aux_loss", aux, on_step=True, on_epoch=False)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        logits = self._fwd(x)
        self.acc1.update(logits, y)
        self.acc5.update(logits, y)
        self.log("val/loss", self.val_loss(logits, y))

    def on_validation_epoch_end(self) -> None:
        a1 = self.acc1.compute()
        a5 = self.acc5.compute()
        self.best_acc1.update(a1)
        self.log_dict(
            {"val/acc1": a1, "val/acc5": a5, "val/acc1_best": self.best_acc1.compute()},
            prog_bar=True,
        )
        self.acc1.reset()
        self.acc5.reset()

    # -------------------------------------------------------------- optimizer
    def configure_optimizers(self):
        opt_cfg = self.cfg.recipe.optimizer
        groups = build_param_groups(self.net, float(opt_cfg.weight_decay))
        opt = torch.optim.AdamW(
            groups, lr=float(opt_cfg.lr), betas=tuple(opt_cfg.betas), eps=float(opt_cfg.eps)
        )
        total = int(self.trainer.estimated_stepping_batches)
        epochs = max(1, int(self.cfg.recipe.epochs))
        warmup = int(float(self.cfg.recipe.schedule.warmup_epochs) * total / epochs)
        sched = warmup_cosine(
            opt, warmup_steps=warmup, total_steps=total,
            min_lr_ratio=float(self.cfg.recipe.schedule.min_lr) / float(opt_cfg.lr),
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

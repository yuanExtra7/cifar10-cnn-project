from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import MulticlassAccuracy


class CifarLitModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 10,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        max_epochs: int = 200,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.num_classes = num_classes
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.max_epochs = max_epochs
        self.label_smoothing = label_smoothing

        self.train_acc = MulticlassAccuracy(num_classes=num_classes, top_k=1)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, top_k=1)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes, top_k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch: Any, stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=float(self.label_smoothing))

        if stage == "train":
            acc = self.train_acc(logits, y)
        elif stage == "val":
            acc = self.val_acc(logits, y)
        else:
            acc = self.test_acc(logits, y)

        # Use "flat" metric names to make checkpoint filename formatting and CLI grepping easier.
        # (Avoid "/" in keys like "val/acc".)
        self.log(f"{stage}_loss", loss, prog_bar=(stage != "train"), on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._step(batch, "test")

    def configure_optimizers(self) -> dict[str, Any]:
        opt = SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=True,
        )
        # T_max uses number of epochs (Lightning steps the scheduler each epoch by default here)
        sch = CosineAnnealingLR(opt, T_max=int(self.max_epochs))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}


from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import lightning as L
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10


@dataclass(frozen=True)
class Cifar10Stats:
    mean: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    std: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)


class Cifar10DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 256,
        num_workers: int = 8,
        download: bool = True,
        val_size: int = 5000,
        seed: int = 42,
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.val_size = val_size
        self.seed = seed
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self._train: Optional[torch.utils.data.Dataset] = None
        self._val: Optional[torch.utils.data.Dataset] = None
        self._test: Optional[torch.utils.data.Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        stats = Cifar10Stats()
        train_tfms = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(stats.mean, stats.std),
            ]
        )
        test_tfms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(stats.mean, stats.std),
            ]
        )

        if stage in (None, "fit"):
            full_train = CIFAR10(root=self.data_dir, train=True, transform=train_tfms, download=self.download)

            # Create a validation split from the training set to avoid leaking test set into model selection.
            val_size = int(self.val_size)
            if val_size <= 0 or val_size >= len(full_train):
                raise ValueError(f"val_size must be in [1, {len(full_train) - 1}], got {val_size}")
            train_size = len(full_train) - val_size

            g = torch.Generator().manual_seed(int(self.seed))
            train_subset, val_subset = random_split(full_train, [train_size, val_size], generator=g)

            # Validation should not use train-time augmentations.
            # random_split returns Subset, so we need a second dataset for val with test transforms.
            full_train_no_aug = CIFAR10(root=self.data_dir, train=True, transform=test_tfms, download=False)
            val_subset = torch.utils.data.Subset(full_train_no_aug, indices=val_subset.indices)

            self._train = train_subset
            self._val = val_subset

        if stage in (None, "test"):
            self._test = CIFAR10(root=self.data_dir, train=False, transform=test_tfms, download=self.download)

    def train_dataloader(self) -> DataLoader:
        assert self._train is not None
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self._persistent_workers(),
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val is not None
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self._persistent_workers(),
        )

    def test_dataloader(self) -> DataLoader:
        assert self._test is not None
        return DataLoader(
            self._test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self._persistent_workers(),
        )

    def _persistent_workers(self) -> bool:
        if self.persistent_workers is not None:
            return self.persistent_workers
        return self.num_workers > 0


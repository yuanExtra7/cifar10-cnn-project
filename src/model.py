from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


def cifar_resnet18(num_classes: int = 10) -> nn.Module:
    """
    ResNet-18 adapted for CIFAR-style 32x32 images:
    - 3x3 conv, stride=1, padding=1
    - remove the initial maxpool
    """
    model = resnet18(num_classes=num_classes)

    # Replace the ImageNet stem with CIFAR stem
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    return model


@torch.no_grad()
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


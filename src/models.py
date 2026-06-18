from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn
from torchvision import models as tv_models


class SimpleCNN(nn.Module):
    """CNN cơ sở gọn nhẹ cho CIFAR-10.

    Thiết kế xấp xỉ 0.55M tham số:
    Conv-BN-ReLU blocks + MaxPool + FC nhỏ.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class VGG11TinyBN(nn.Module):
    """Phiên bản VGG-11 rút gọn cho CIFAR-10."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        cfg = [32, "M", 64, "M", 128, 128, "M", 256, 256, "M", 256, 256, "M"]
        layers = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([
                    nn.Conv2d(in_channels, int(v), kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(int(v)),
                    nn.ReLU(inplace=True),
                ])
                in_channels = int(v)
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def _set_first_conv_stride_1(model: nn.Module) -> None:
    """Chỉnh stem stride về 1 để phù hợp ảnh 32x32 khi có thể."""
    # MobileNetV2: features[0][0]
    try:
        if hasattr(model, "features") and isinstance(model.features[0], nn.Sequential):
            conv = model.features[0][0]
            if isinstance(conv, nn.Conv2d):
                conv.stride = (1, 1)
    except Exception:
        pass


def build_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    try:
        model = tv_models.resnet18(weights=None)
    except TypeError:
        model = tv_models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_mobilenetv2_cifar(num_classes: int = 10) -> nn.Module:
    try:
        model = tv_models.mobilenet_v2(weights=None, num_classes=num_classes)
    except TypeError:
        model = tv_models.mobilenet_v2(pretrained=False, num_classes=num_classes)
    _set_first_conv_stride_1(model)
    return model


def build_efficientnetb0_cifar(num_classes: int = 10) -> nn.Module:
    try:
        model = tv_models.efficientnet_b0(weights=None, num_classes=num_classes)
    except TypeError:
        model = tv_models.efficientnet_b0(pretrained=False, num_classes=num_classes)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    _set_first_conv_stride_1(model)
    return model


MODEL_REGISTRY: Dict[str, Callable[[int], nn.Module]] = {
    "SimpleCNN": lambda num_classes: SimpleCNN(num_classes=num_classes),
    "VGG11TinyBN": lambda num_classes: VGG11TinyBN(num_classes=num_classes),
    "ResNet18CIFAR": build_resnet18_cifar,
    "MobileNetV2CIFAR": build_mobilenetv2_cifar,
    "EfficientNetB0CIFAR": build_efficientnetb0_cifar,
}


def create_model(name: str, num_classes: int = 10) -> nn.Module:
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available models: {available}")
    return MODEL_REGISTRY[name](num_classes)

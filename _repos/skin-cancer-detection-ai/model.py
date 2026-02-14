"""
Model factory — returns a pre-trained backbone with a custom classifier head.
"""

import torch.nn as nn
from torchvision import models

import config


def build_model(
    arch: str = "mobilenet_v2",
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build and return a classification model.

    Supported architectures:
        - mobilenet_v2
        - efficientnet_b0
        - resnet18
    """
    if arch == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown architecture: {arch}")

    return model

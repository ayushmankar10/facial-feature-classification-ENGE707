"""Transfer-learning models with a shared eight-class output."""

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_V2_S_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    efficientnet_v2_s,
    resnet18,
)

from config import CLASSES


HEAD_PREFIXES = {
    "efficientnet_b0": "classifier.",
    "efficientnet_v2_s": "classifier.",
    "resnet18": "fc.",
}


def build_model(
    architecture="efficientnet_b0",
    pretrained=True,
):
    if architecture == "efficientnet_b0":
        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )
        model = efficientnet_b0(weights=weights)

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            len(CLASSES),
        )

    elif architecture == "efficientnet_v2_s":
        weights = (
            EfficientNet_V2_S_Weights.DEFAULT
            if pretrained
            else None
        )
        model = efficientnet_v2_s(weights=weights)

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            len(CLASSES),
        )

    elif architecture == "resnet18":
        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )
        model = resnet18(weights=weights)

        model.fc = nn.Linear(
            model.fc.in_features,
            len(CLASSES),
        )

    else:
        raise ValueError(
            f"Unsupported architecture: {architecture}"
        )

    return model


def is_head(name, architecture):
    if architecture not in HEAD_PREFIXES:
        raise ValueError(
            f"Unsupported architecture: {architecture}"
        )

    return name.startswith(
        HEAD_PREFIXES[architecture]
    )


def set_trainable(
    model,
    architecture,
    freeze_backbone,
):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            not freeze_backbone
            or is_head(name, architecture)
        )


def parameter_groups(model, config):
    backbone = []
    head = []

    for name, parameter in model.named_parameters():
        if is_head(name, config.architecture):
            head.append(parameter)
        else:
            backbone.append(parameter)

    return [
        {
            "params": backbone,
            "lr": config.learning_rate,
        },
        {
            "params": head,
            "lr": config.head_learning_rate,
        },
    ]
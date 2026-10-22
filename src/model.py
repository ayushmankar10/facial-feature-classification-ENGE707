import torch.nn as nn

from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)

from config import CLASSES


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
    prefix = (
        "classifier."
        if architecture == "efficientnet_b0"
        else "fc."
    )

    return name.startswith(prefix)


def set_trainable(
    model,
    architecture,
    freeze_backbone,
):
    for name, parameter in model.named_parameters():
        trainable = (
            not freeze_backbone
            or is_head(name, architecture)
        )

        parameter.requires_grad_(trainable)


def parameter_groups(model, config):
    backbone_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not is_head(name, config.architecture)
    ]

    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if is_head(name, config.architecture)
    ]

    return [
        {
            "params": backbone_parameters,
            "lr": config.learning_rate,
        },
        {
            "params": head_parameters,
            "lr": config.head_learning_rate,
        },
    ]
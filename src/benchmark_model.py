"""Benchmark full-network training before a long experiment."""

import argparse
import multiprocessing

import torch
from torch import nn

from config import SPLIT_DIR
from dataset import (
    FacialDataset,
    build_transforms,
    make_loader,
    read_split,
)
from model import build_model
from train import run_epoch
from utils import choose_device, seed_everything


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--architecture",
        choices=(
            "efficientnet_b0",
            "efficientnet_v2_s",
            "resnet18",
        ),
        default="efficientnet_v2_s",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )
    args = parser.parse_args()

    if args.batch_size < 2 or args.workers < 0:
        parser.error(
            "batch-size must be >= 2 and workers >= 0."
        )

    seed_everything(42)
    torch.set_num_threads(4)
    device = choose_device()

    train_frame = read_split(
        SPLIT_DIR / "train.csv"
    )
    validation_frame = read_split(
        SPLIT_DIR / "validation.csv"
    )

    train_loader = make_loader(
        FacialDataset(
            train_frame,
            build_transforms(
                224,
                training=True,
                resize_mode="stretch",
            ),
        ),
        args.batch_size,
        args.workers,
        shuffle=True,
        cuda=True,
    )

    validation_loader = make_loader(
        FacialDataset(
            validation_frame,
            build_transforms(
                224,
                resize_mode="stretch",
            ),
        ),
        args.batch_size,
        args.workers,
        cuda=True,
    )

    # Random weights avoid a download for this speed check.
    model = build_model(
        args.architecture,
        pretrained=False,
    ).to(
        device,
        memory_format=torch.channels_last,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0003,
        weight_decay=0.0001,
    )
    scaler = torch.amp.GradScaler("cuda")
    training_loss = nn.CrossEntropyLoss(
        label_smoothing=0.05
    )
    validation_loss = nn.CrossEntropyLoss()

    print(
        f"Model: {args.architecture}; "
        f"batch size: {args.batch_size}; "
        f"CPU workers: {args.workers}",
        flush=True,
    )

    print("\nWarming up: 10 training batches...", flush=True)

    run_epoch(
        model,
        train_loader,
        training_loss,
        device,
        optimizer=optimizer,
        scaler=scaler,
        max_batches=10,
    )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    print("Measuring: 60 training batches...", flush=True)

    training = run_epoch(
        model,
        train_loader,
        training_loss,
        device,
        optimizer=optimizer,
        scaler=scaler,
        max_batches=60,
    )

    print("Measuring: 30 validation batches...", flush=True)

    validation = run_epoch(
        model,
        validation_loader,
        validation_loss,
        device,
        max_batches=30,
    )

    train_seconds = (
        len(train_frame)
        / training["images_per_second"]
    )
    validation_seconds = (
        len(validation_frame)
        / validation["images_per_second"]
    )
    epoch_minutes = (
        train_seconds + validation_seconds
    ) / 60

    allocated = (
        torch.cuda.max_memory_allocated() / 1024**3
    )
    reserved = (
        torch.cuda.max_memory_reserved() / 1024**3
    )

    print("\nBENCHMARK RESULTS")
    print(
        "Training speed: "
        f"{training['images_per_second']:.1f} images/s"
    )
    print(
        "Validation speed: "
        f"{validation['images_per_second']:.1f} images/s"
    )
    print(
        f"Peak PyTorch allocated memory: {allocated:.2f} GiB"
    )
    print(
        f"Peak PyTorch reserved memory: {reserved:.2f} GiB"
    )
    print(
        "Estimated full fine-tuning epoch: "
        f"{epoch_minutes:.1f} minutes"
    )
    print(
        "\nTiming is approximate. This short benchmark "
        "cannot capture overnight thermal changes."
    )
    print(
        "Memory figures exclude other applications "
        "and some CUDA overhead."
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
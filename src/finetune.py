"""Fine-tune an existing checkpoint at a new image resolution."""

import argparse
import multiprocessing
import time
from dataclasses import fields
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from config import (
    CLASSES,
    PROJECT_ROOT,
    SPLIT_DIR,
    TrainConfig,
)
from dataset import (
    FacialDataset,
    build_transforms,
    make_loader,
    read_split,
)
from model import (
    build_model,
    parameter_groups,
    set_trainable,
)
from train import run_epoch
from utils import (
    choose_device,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    split_fingerprint,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.00001,
    )
    parser.add_argument(
        "--head-learning-rate",
        type=float,
        default=0.00003,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Check loading and training using two batches.",
    )

    args = parser.parse_args()

    allowed = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789_-"
    )

    if not args.run or any(c not in allowed for c in args.run):
        parser.error(
            "--run may only contain letters, numbers, "
            "underscores and hyphens."
        )

    if (
        args.epochs < 1
        or args.image_size < 32
        or args.batch_size < 2
        or args.workers < 0
        or args.learning_rate <= 0
        or args.head_learning_rate <= 0
    ):
        parser.error("Invalid training settings.")

    run_dir = PROJECT_ROOT / "outputs" / "runs" / args.run
    model_dir = PROJECT_ROOT / "models" / args.run

    if run_dir.exists() or model_dir.exists():
        raise FileExistsError(
            "Run already exists. Choose a new --run name."
        )

    parent = load_checkpoint(args.checkpoint)

    if parent.get("smoke"):
        raise ValueError(
            "Start from a real trained checkpoint, "
            "not a smoke-test checkpoint."
        )

    fingerprint = split_fingerprint()

    if parent["split_fingerprint"] != fingerprint:
        raise ValueError(
            "Current splits differ from the original "
            "checkpoint's training splits."
        )

    # Inherit the original experiment's settings.
    known_fields = {field.name for field in fields(TrainConfig)}
    inherited = {
        key: value
        for key, value in parent["config"].items()
        if key in known_fields
    }

    config = TrainConfig(**inherited)
    config.image_size = args.image_size
    config.batch_size = args.batch_size
    config.workers = args.workers
    config.epochs = 1 if args.smoke else args.epochs
    config.learning_rate = args.learning_rate
    config.head_learning_rate = args.head_learning_rate
    config.freeze_epochs = 0

    # We load trained weights below instead of ImageNet weights.
    config.pretrained = False

    seed_everything(config.seed)
    torch.set_num_threads(config.cpu_threads)
    device = choose_device()

    train_frame = read_split(SPLIT_DIR / "train.csv")
    validation_frame = read_split(
        SPLIT_DIR / "validation.csv"
    )

    if (
        set(train_frame["image_id"])
        & set(validation_frame["image_id"])
    ) or (
        set(train_frame["pixel_hash"])
        & set(validation_frame["pixel_hash"])
    ):
        raise ValueError("Training and validation overlap.")

    minimum_side = train_frame[
        ["width", "height"]
    ].min(axis=1)

    train_frame = train_frame.loc[
        minimum_side >= config.min_train_side
    ]

    if (
        set(train_frame["class"]) != set(CLASSES)
        or set(validation_frame["class"]) != set(CLASSES)
    ):
        raise ValueError(
            "All eight classes must be present in both splits."
        )

    if args.smoke:
        train_frame = train_frame.groupby(
            "class", group_keys=False
        ).head(4)

        validation_frame = validation_frame.groupby(
            "class", group_keys=False
        ).head(2)

    train_loader = make_loader(
        FacialDataset(
            train_frame,
            build_transforms(
                config.image_size,
                training=True,
                resize_mode=config.resize_mode,
            ),
        ),
        config.batch_size,
        config.workers,
        shuffle=True,
        cuda=True,
        seed=config.seed,
    )

    validation_loader = make_loader(
        FacialDataset(
            validation_frame,
            build_transforms(
                config.image_size,
                resize_mode=config.resize_mode,
            ),
        ),
        config.batch_size,
        config.workers,
        cuda=True,
        seed=config.seed,
    )

    model = build_model(
        config.architecture,
        pretrained=False,
    )
    model.load_state_dict(parent["model_state"])

    parent_epoch = int(parent["epoch"])
    del parent

    model.to(
        device,
        memory_format=torch.channels_last,
    )

    set_trainable(
        model,
        config.architecture,
        freeze_backbone=False,
    )

    print(
        f"Loaded trained weights: {args.checkpoint}",
        flush=True,
    )
    print(f"Source epoch: {parent_epoch}", flush=True)
    print(f"Architecture: {config.architecture}", flush=True)
    print(
        f"Model device: {next(model.parameters()).device}",
        flush=True,
    )
    print(
        f"Image size: {config.image_size}; "
        f"resize mode: {config.resize_mode}",
        flush=True,
    )
    print(
        f"Batch size: {config.batch_size}; "
        f"CPU workers: {config.workers}",
        flush=True,
    )
    print(
        f"Backbone LR: {config.learning_rate}; "
        f"classifier LR: {config.head_learning_rate}",
        flush=True,
    )

    # A fresh optimizer is intentional for this new fine-tuning stage.
    optimizer = torch.optim.AdamW(
        parameter_groups(model, config),
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.amp,
    )

    training_loss = nn.CrossEntropyLoss(
        label_smoothing=config.label_smoothing
    )
    validation_loss = nn.CrossEntropyLoss()

    run_dir.mkdir(parents=True)

    provenance = {
        "initial_checkpoint": str(args.checkpoint),
        "initial_checkpoint_epoch": parent_epoch,
        "stage": "Higher-resolution fine-tuning",
    }

    write_json(
        run_dir / "config.json",
        {
            **config.to_dict(),
            **provenance,
            "smoke": args.smoke,
            "split_fingerprint": fingerprint,
            "torch_version": str(torch.__version__),
        },
    )

    def save_model(filename, epoch, stats):
        save_checkpoint(
            model_dir / filename,
            {
                "model_state": model.state_dict(),
                "config": config.to_dict(),
                "classes": list(CLASSES),
                "epoch": epoch,
                "smoke": args.smoke,
                "split_fingerprint": fingerprint,
                "val_accuracy": stats["accuracy"],
                "val_loss": stats["loss"],
                **provenance,
            },
        )

    # Measure the loaded model at the new resolution first.
    print(
        "\nEvaluating before fine-tuning at the new resolution...",
        flush=True,
    )

    baseline = run_epoch(
        model,
        validation_loader,
        validation_loss,
        device,
        amp=config.amp,
        max_batches=2 if args.smoke else 0,
    )

    write_json(
        run_dir / "baseline_metrics.json",
        baseline,
    )

    best_accuracy = baseline["accuracy"]
    best_loss = baseline["loss"]
    best_epoch = 0

    # Keep the starting weights if fine-tuning makes results worse.
    save_model("best.pt", 0, baseline)

    print(
        f"Starting validation accuracy: {best_accuracy:.6f}; "
        f"loss: {best_loss:.6f}",
        flush=True,
    )

    history = []

    for epoch in range(1, config.epochs + 1):
        print(
            f"\nFine-tuning epoch {epoch}/{config.epochs}",
            flush=True,
        )
        started = time.perf_counter()

        training = run_epoch(
            model,
            train_loader,
            training_loss,
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp=config.amp,
            max_batches=2 if args.smoke else 0,
        )

        validation = run_epoch(
            model,
            validation_loader,
            validation_loss,
            device,
            amp=config.amp,
            max_batches=2 if args.smoke else 0,
        )

        row = {
            "epoch": epoch,
            "train_loss": training["loss"],
            "train_accuracy": training["accuracy"],
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
            "train_images_per_second": (
                training["images_per_second"]
            ),
        }

        history.append(row)
        pd.DataFrame(history).to_csv(
            run_dir / "history.csv",
            index=False,
        )

        improved = (
            validation["accuracy"] > best_accuracy
            or (
                validation["accuracy"] == best_accuracy
                and validation["loss"] < best_loss
            )
        )

        if improved:
            best_accuracy = validation["accuracy"]
            best_loss = validation["loss"]
            best_epoch = epoch
            save_model("best.pt", epoch, validation)

        save_model("last.pt", epoch, validation)
        scheduler.step()

        print(
            f"Train acc {training['accuracy']:.6f}, "
            f"val acc {validation['accuracy']:.6f}, "
            f"val loss {validation['loss']:.6f}, "
            f"{row['seconds']:.1f}s",
            flush=True,
        )

    print(
        f"\nBest validation accuracy: {best_accuracy:.6f}"
    )
    print(
        f"Best fine-tuning epoch: {best_epoch} "
        "(0 means before fine-tuning)"
    )
    print(f"Checkpoint: {model_dir / 'best.pt'}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
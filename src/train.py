import argparse
import multiprocessing
import time
from dataclasses import fields

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
from utils import (
    choose_device,
    save_checkpoint,
    seed_everything,
    split_fingerprint,
    write_json,
)


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    amp=True,
    freeze_backbone=False,
    max_batches=0,
):
    training = optimizer is not None

    model.train(training)

    if training and freeze_backbone:
        # Keep frozen backbone BatchNorm statistics fixed.
        for module in model.modules():
            if isinstance(
                module,
                nn.modules.batchnorm._BatchNorm,
            ):
                module.eval()

    totals = torch.zeros(
        3,
        device=device,
        dtype=torch.float64,
    )

    started = time.perf_counter()

    for step, batch in enumerate(loader, start=1):
        images, labels, _ = batch

        images = images.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                enabled=amp and device.type == "cuda",
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                scaler.step(optimizer)
                scaler.update()

        totals[0] += (
            loss.detach().double()
            * labels.size(0)
        )
        totals[1] += (
            logits.detach().argmax(dim=1) == labels
        ).sum()
        totals[2] += labels.size(0)

        if step % 100 == 0:
            phase = (
                "train" if training else "validation"
            )

            print(
                f"  {phase} batch {step}/{len(loader)}",
                flush=True,
            )

        if max_batches and step >= max_batches:
            break

    loss_sum, correct, count = totals.cpu().tolist()

    return {
        "loss": loss_sum / count,
        "accuracy": correct / count,
        "images_per_second": (
            count / (time.perf_counter() - started)
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train the facial-feature classifier."
    )

    parser.add_argument(
        "--run",
        required=True,
        help="Unique experiment folder name.",
    )

    parser.add_argument(
        "--architecture",
        choices=["efficientnet_b0", "resnet18"],
        default="efficientnet_b0",
    )

    defaults = TrainConfig()

    integer_arguments = [
        "epochs",
        "batch_size",
        "workers",
        "cpu_threads",
        "image_size",
        "freeze_epochs",
        "patience",
        "seed",
        "min_train_side",
    ]

    for name in integer_arguments:
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=int,
            default=getattr(defaults, name),
        )

    float_arguments = [
        "learning_rate",
        "head_learning_rate",
        "weight_decay",
        "label_smoothing",
    ]

    for name in float_arguments:
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=float,
            default=getattr(defaults, name),
        )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Two batches per phase for one epoch. "
            "Only checks the pipeline."
        ),
    )

    args = parser.parse_args()

    allowed_characters = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789_-"
    )

    if (
        not args.run
        or any(
            character not in allowed_characters
            for character in args.run
        )
    ):
        parser.error(
            "--run may only contain letters, numbers, "
            "underscores and hyphens."
        )

    values = {
        field.name: getattr(args, field.name)
        for field in fields(TrainConfig)
        if hasattr(args, field.name)
    }

    config = TrainConfig(
        **values,
        pretrained=not args.no_pretrained,
        amp=not args.no_amp,
    )

    if (
        config.epochs < 1
        or config.workers < 0
        or config.batch_size < 2
        or config.cpu_threads < 1
    ):
        parser.error(
            "epochs/threads must be positive, "
            "workers >= 0 and batch-size >= 2."
        )

    if (
        config.image_size < 32
        or not 0 <= config.label_smoothing < 1
        or config.freeze_epochs < 0
        or config.patience < 1
    ):
        parser.error(
            "Invalid image size, smoothing, "
            "freeze epochs or patience."
        )

    if (
        config.min_train_side < 0
        or min(
            config.learning_rate,
            config.head_learning_rate,
        ) <= 0
        or config.weight_decay < 0
    ):
        parser.error(
            "Invalid size threshold, learning rate "
            "or weight decay."
        )

    if args.smoke:
        config.epochs = 1
        config.freeze_epochs = 0

    seed_everything(config.seed)
    torch.set_num_threads(config.cpu_threads)

    device = choose_device(args.allow_cpu)

    train_frame = read_split(
        SPLIT_DIR / "train.csv"
    )
    validation_frame = read_split(
        SPLIT_DIR / "validation.csv"
    )

    hash_overlap = (
        set(train_frame["pixel_hash"])
        & set(validation_frame["pixel_hash"])
    )

    id_overlap = (
        set(train_frame["image_id"])
        & set(validation_frame["image_id"])
    )

    if hash_overlap or id_overlap:
        raise ValueError(
            "Training and validation data overlap."
        )

    # Any size filtering affects training only.
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
            "All eight classes must be present "
            "in both splits."
        )

    if args.smoke:
        train_frame = train_frame.groupby(
            "class",
            group_keys=False,
        ).head(4)

        validation_frame = validation_frame.groupby(
            "class",
            group_keys=False,
        ).head(2)

    run_dir = (
        PROJECT_ROOT
        / "outputs"
        / "runs"
        / args.run
    )

    if run_dir.exists():
        raise FileExistsError(
            "This run already exists. Use a new "
            "--run name to preserve its results."
        )

    train_dataset = FacialDataset(
        train_frame,
        build_transforms(
            config.image_size,
            training=True,
        ),
    )

    validation_dataset = FacialDataset(
        validation_frame,
        build_transforms(config.image_size),
    )

    train_loader = make_loader(
        train_dataset,
        config.batch_size,
        config.workers,
        shuffle=True,
        cuda=device.type == "cuda",
        seed=config.seed,
    )

    validation_loader = make_loader(
        validation_dataset,
        config.batch_size,
        config.workers,
        cuda=device.type == "cuda",
        seed=config.seed,
    )

    model = build_model(
        config.architecture,
        config.pretrained,
    )

    model = model.to(
        device,
        memory_format=torch.channels_last,
    )

    print(
        f"Model device: {next(model.parameters()).device}",
        flush=True,
    )
    print(
        f"CPU workers per loader: {config.workers}",
        flush=True,
    )
    print(
        f"Training rows: {len(train_frame):,}; "
        f"validation rows: {len(validation_frame):,}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        parameter_groups(model, config),
        weight_decay=config.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            config.amp and device.type == "cuda"
        ),
    )

    train_criterion = nn.CrossEntropyLoss(
        label_smoothing=config.label_smoothing
    )
    validation_criterion = nn.CrossEntropyLoss()

    run_dir.mkdir(parents=True)

    fingerprint = split_fingerprint()

    write_json(
        run_dir / "config.json",
        {
            **config.to_dict(),
            "smoke": args.smoke,
            "device": str(device),
            "torch_version": str(torch.__version__),
            "split_fingerprint": fingerprint,
            "note": (
                "Training loss includes label smoothing; "
                "validation loss is plain cross entropy."
            ),
        },
    )

    best_accuracy = -1.0
    best_loss = float("inf")
    bad_epochs = 0
    history = []

    checkpoint_dir = (
        PROJECT_ROOT / "models" / args.run
    )

    for epoch in range(config.epochs):
        frozen = (
            config.pretrained
            and epoch < config.freeze_epochs
        )

        set_trainable(
            model,
            config.architecture,
            freeze_backbone=frozen,
        )

        print(
            f"\nEpoch {epoch + 1}/{config.epochs}; "
            f"frozen backbone: {frozen}",
            flush=True,
        )

        started = time.perf_counter()

        train_stats = run_epoch(
            model,
            train_loader,
            train_criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp=config.amp,
            freeze_backbone=frozen,
            max_batches=2 if args.smoke else 0,
        )

        validation_stats = run_epoch(
            model,
            validation_loader,
            validation_criterion,
            device,
            amp=config.amp,
            max_batches=2 if args.smoke else 0,
        )

        row = {
            "epoch": epoch + 1,
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "val_loss": validation_stats["loss"],
            "val_accuracy": validation_stats["accuracy"],
            "learning_rate": (
                optimizer.param_groups[0]["lr"]
            ),
            "seconds": (
                time.perf_counter() - started
            ),
            "train_images_per_second": (
                train_stats["images_per_second"]
            ),
        }

        history.append(row)

        pd.DataFrame(history).to_csv(
            run_dir / "history.csv",
            index=False,
        )

        scheduler.step()

        improved = (
            validation_stats["accuracy"] > best_accuracy
            or (
                validation_stats["accuracy"]
                == best_accuracy
                and validation_stats["loss"] < best_loss
            )
        )

        if improved:
            best_accuracy = validation_stats["accuracy"]
            best_loss = validation_stats["loss"]
            bad_epochs = 0
        else:
            bad_epochs += 1

        checkpoint = {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "classes": list(CLASSES),
            "epoch": epoch + 1,
            "smoke": args.smoke,
            "split_fingerprint": fingerprint,
            "val_accuracy": validation_stats["accuracy"],
            "val_loss": validation_stats["loss"],
        }

        save_checkpoint(
            checkpoint_dir / "last.pt",
            checkpoint,
        )

        if improved:
            save_checkpoint(
                checkpoint_dir / "best.pt",
                checkpoint,
            )

        print(
            f"Train acc {row['train_accuracy']:.4f}, "
            f"val acc {row['val_accuracy']:.4f}, "
            f"val loss {row['val_loss']:.4f}, "
            f"{row['seconds']:.1f}s",
            flush=True,
        )

        if bad_epochs >= config.patience:
            print("Early stopping.", flush=True)
            break

    print(
        f"\nBest validation accuracy: "
        f"{best_accuracy:.4f}"
    )
    print(
        f"Checkpoint: {checkpoint_dir / 'best.pt'}"
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
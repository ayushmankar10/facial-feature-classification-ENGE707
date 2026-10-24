"""Train B0 on a separate split, with restartable epoch checkpoints."""

import argparse
import hashlib
import multiprocessing
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from config import (
    AUDIT_PATH,
    CLASSES,
    LABELS_PATH,
    PROJECT_ROOT,
    TrainConfig,
)
from dataset import (
    FacialDataset,
    build_transforms,
    make_loader,
    read_split,
)
from model import build_model, parameter_groups, set_trainable
from prepare_splits import make_splits
from train import run_epoch
from utils import (
    choose_device,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    write_json,
)


def fingerprint(directory):
    return {
        name: hashlib.sha256(
            (directory / name).read_bytes()
        ).hexdigest()
        for name in ("train.csv", "validation.csv")
    }


def create_frames(seed):
    audit = pd.read_csv(
        AUDIT_PATH,
        dtype={"image_id": "string"},
    )
    labels = pd.read_csv(
        LABELS_PATH,
        dtype="string",
    )

    if (
        labels.isna().any().any()
        or labels["image_id"].duplicated().any()
    ):
        raise ValueError(
            "Original labels contain missing values "
            "or duplicate IDs."
        )

    expected = (
        labels.set_index("image_id")["class"].sort_index()
    )
    actual = (
        audit.set_index("image_id")["class"]
        .astype("string")
        .sort_index()
    )

    if not expected.equals(actual):
        raise ValueError(
            "Audit does not match the original labels."
        )

    return make_splits(
        audit,
        validation_fraction=0.2,
        seed=seed,
    )


def capture_rng(train_loader, validation_loader):
    state = np.random.get_state()

    return {
        "python": random.getstate(),
        "numpy": [
            state[0],
            state[1].tolist(),
            int(state[2]),
            int(state[3]),
            float(state[4]),
        ],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "train_generator": (
            train_loader.generator.get_state()
        ),
        "validation_generator": (
            validation_loader.generator.get_state()
        ),
    }


def restore_rng(saved, train_loader, validation_loader):
    random.setstate(saved["python"])

    state = saved["numpy"]
    np.random.set_state((
        state[0],
        np.asarray(state[1], dtype=np.uint32),
        state[2],
        state[3],
        state[4],
    ))

    torch.set_rng_state(saved["torch"])
    torch.cuda.set_rng_state_all(saved["cuda"])

    train_loader.generator.set_state(
        saved["train_generator"]
    )
    validation_loader.generator.set_state(
        saved["validation_generator"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--run", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    allowed = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789_-"
    )

    if not args.run or any(c not in allowed for c in args.run):
        parser.error(
            "Use letters, numbers, underscores "
            "or hyphens in --run."
        )

    if (
        args.epochs < 1
        or args.batch_size < 2
        or args.workers < 0
    ):
        parser.error(
            "epochs >= 1, batch-size >= 2 "
            "and workers >= 0 required."
        )

    run_dir = (
        PROJECT_ROOT / "outputs" / "runs" / args.run
    )
    split_dir = run_dir / "splits"
    model_dir = PROJECT_ROOT / "models" / args.run

    saved = None

    if args.resume:
        saved = load_checkpoint(model_dir / "last.pt")
        config = TrainConfig(**saved["config"])
        smoke = bool(saved["smoke"])
        split_seed = saved["split_seed"]

        expected_directory = (
            split_dir.relative_to(PROJECT_ROOT).as_posix()
        )

        if saved["split_dir"] != expected_directory:
            raise ValueError(
                "Checkpoint belongs to a different run folder."
            )

        if saved["split_fingerprint"] != fingerprint(split_dir):
            raise ValueError(
                "Saved split files changed. Refusing to resume."
            )

        train_frame = read_split(split_dir / "train.csv")
        validation_frame = read_split(
            split_dir / "validation.csv"
        )

        start_epoch = saved["epoch"]
        history = saved["history"]
        best_accuracy = saved["best_accuracy"]
        best_loss = saved["best_loss"]
        bad_epochs = saved["bad_epochs"]

        if (
            start_epoch >= config.epochs
            or bad_epochs >= config.patience
        ):
            print("This run has already finished.")
            return

    else:
        if run_dir.exists() or model_dir.exists():
            raise FileExistsError(
                "Run exists. Use --resume "
                "or a new --run name."
            )

        config = TrainConfig(
            architecture="efficientnet_b0",
            image_size=224,
            resize_mode="stretch",
            epochs=1 if args.smoke else args.epochs,
            batch_size=args.batch_size,
            workers=args.workers,
            freeze_epochs=0 if args.smoke else 2,
            pretrained=not args.smoke,
        )

        smoke = args.smoke
        split_seed = args.split_seed

        if split_seed == 42:
            parser.error(
                "Use a different split seed "
                "from the original seed 42."
            )

        train_frame, validation_frame = create_frames(
            split_seed
        )

        start_epoch = 0
        history = []
        best_accuracy = -1.0
        best_loss = float("inf")
        bad_epochs = 0

    id_overlap = (
        set(train_frame["image_id"])
        & set(validation_frame["image_id"])
    )
    hash_overlap = (
        set(train_frame["pixel_hash"])
        & set(validation_frame["pixel_hash"])
    )

    if id_overlap or hash_overlap:
        raise ValueError(
            "Training and validation overlap."
        )

    seed_everything(config.seed)
    torch.set_num_threads(config.cpu_threads)
    device = choose_device()

    train_used = train_frame
    validation_used = validation_frame

    if smoke:
        train_used = train_frame.groupby(
            "class",
            group_keys=False,
        ).head(4)

        validation_used = validation_frame.groupby(
            "class",
            group_keys=False,
        ).head(2)

    train_loader = make_loader(
        FacialDataset(
            train_used,
            build_transforms(
                224,
                training=True,
                resize_mode="stretch",
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
            validation_used,
            build_transforms(
                224,
                resize_mode="stretch",
            ),
        ),
        config.batch_size,
        config.workers,
        cuda=True,
        seed=config.seed,
    )

    model = build_model(
        "efficientnet_b0",
        pretrained=config.pretrained and not args.resume,
    )

    if saved is not None:
        model.load_state_dict(saved["model_state"])

    model.to(
        device,
        memory_format=torch.channels_last,
    )

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

    if saved is not None:
        optimizer.load_state_dict(
            saved["optimizer_state"]
        )
        scheduler.load_state_dict(
            saved["scheduler_state"]
        )
        scaler.load_state_dict(
            saved["scaler_state"]
        )

        restore_rng(
            saved["rng_state"],
            train_loader,
            validation_loader,
        )

        del saved

        print(
            f"Resuming after epoch {start_epoch}; "
            "keeping saved settings.",
            flush=True,
        )

    else:
        split_dir.mkdir(parents=True)

        for name, frame in (
            ("train.csv", train_frame),
            ("validation.csv", validation_frame),
        ):
            (split_dir / name).write_text(
                frame.to_csv(index=False),
                encoding="utf-8",
                newline="",
            )

        write_json(
            run_dir / "config.json",
            {
                **config.to_dict(),
                "smoke": smoke,
                "split_seed": split_seed,
                "split_dir": (
                    split_dir.relative_to(PROJECT_ROOT)
                    .as_posix()
                ),
                "split_fingerprint": fingerprint(split_dir),
                "audit_sha256": hashlib.sha256(
                    AUDIT_PATH.read_bytes()
                ).hexdigest(),
                "torch_version": str(torch.__version__),
                "limitation": (
                    "Exact duplicates grouped; "
                    "related crops not grouped."
                ),
            },
        )

    split_hashes = fingerprint(split_dir)

    print(
        "Model: efficientnet_b0; "
        f"device: {next(model.parameters()).device}",
        flush=True,
    )
    print(
        f"Split seed: {split_seed}; "
        f"training seed: {config.seed}",
        flush=True,
    )
    print(
        f"Training rows: {len(train_used):,}; "
        f"validation rows: {len(validation_used):,}",
        flush=True,
    )
    print(f"Split folder: {split_dir}", flush=True)

    training_loss = nn.CrossEntropyLoss(
        label_smoothing=config.label_smoothing
    )
    validation_loss = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, config.epochs):
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

        training = run_epoch(
            model,
            train_loader,
            training_loss,
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp=config.amp,
            freeze_backbone=frozen,
            max_batches=2 if smoke else 0,
        )

        validation = run_epoch(
            model,
            validation_loader,
            validation_loss,
            device,
            amp=config.amp,
            max_batches=2 if smoke else 0,
        )

        row = {
            "epoch": epoch + 1,
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
        scheduler.step()

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
            bad_epochs = 0
        else:
            bad_epochs += 1

        checkpoint = {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "classes": list(CLASSES),
            "epoch": epoch + 1,
            "smoke": smoke,
            "split_seed": split_seed,
            "split_dir": (
                split_dir.relative_to(PROJECT_ROOT)
                .as_posix()
            ),
            "split_fingerprint": split_hashes,
            "val_accuracy": validation["accuracy"],
            "val_loss": validation["loss"],
            "best_accuracy": best_accuracy,
            "best_loss": best_loss,
            "bad_epochs": bad_epochs,
            "history": history,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "rng_state": capture_rng(
                train_loader,
                validation_loader,
            ),
        }

        if improved:
            save_checkpoint(
                model_dir / "best.pt",
                checkpoint,
            )

        save_checkpoint(
            model_dir / "last.pt",
            checkpoint,
        )

        pd.DataFrame(history).to_csv(
            run_dir / "history.csv",
            index=False,
        )

        print(
            f"Train acc {row['train_accuracy']:.6f}, "
            f"val acc {row['val_accuracy']:.6f}, "
            f"val loss {row['val_loss']:.6f}, "
            f"{row['seconds']:.1f}s",
            flush=True,
        )

        if bad_epochs >= config.patience:
            print("Early stopping.", flush=True)
            break

    print(
        f"\nBest validation accuracy: {best_accuracy:.6f}"
    )
    print(f"Checkpoint: {model_dir / 'best.pt'}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
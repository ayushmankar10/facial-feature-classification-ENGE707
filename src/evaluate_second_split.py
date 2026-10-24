"""Evaluate a second-split checkpoint on its saved validation data."""

import argparse
import hashlib
import multiprocessing
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from config import CLASSES, PROJECT_ROOT
from dataset import (
    FacialDataset,
    build_transforms,
    make_loader,
    read_split,
)
from model import build_model
from utils import (
    choose_device,
    load_checkpoint,
    predict_batches,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--flip-tta", action="store_true")

    args = parser.parse_args()

    if args.workers < 0 or args.batch_size < 1:
        parser.error(
            "workers >= 0 and batch-size >= 1 required."
        )

    evaluation_name = (
        "evaluation_flip"
        if args.flip_tta
        else "evaluation"
    )
    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "runs"
        / args.checkpoint.parent.name
        / evaluation_name
    )

    if output_dir.exists():
        raise FileExistsError(
            f"Evaluation already exists: {output_dir}"
        )

    checkpoint = load_checkpoint(args.checkpoint)

    if checkpoint.get("smoke"):
        raise ValueError(
            "Smoke checkpoints cannot be used "
            "for real evaluation."
        )

    if "split_dir" not in checkpoint:
        raise ValueError(
            "Use evaluate.py for original-split checkpoints."
        )

    split_dir = PROJECT_ROOT / checkpoint["split_dir"]

    actual = {
        name: hashlib.sha256(
            (split_dir / name).read_bytes()
        ).hexdigest()
        for name in ("train.csv", "validation.csv")
    }

    if actual != checkpoint["split_fingerprint"]:
        raise ValueError(
            "The checkpoint's saved split files changed."
        )

    config = checkpoint["config"]
    torch.set_num_threads(config.get("cpu_threads", 4))
    device = choose_device()

    frame = read_split(split_dir / "validation.csv")

    loader = make_loader(
        FacialDataset(
            frame,
            build_transforms(
                config["image_size"],
                resize_mode=config.get(
                    "resize_mode", "stretch"
                ),
            ),
        ),
        args.batch_size,
        args.workers,
        cuda=True,
    )

    model = build_model(
        config["architecture"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(
        device,
        memory_format=torch.channels_last,
    )

    print(
        f"Validation split: {split_dir}",
        flush=True,
    )

    ids, targets, probabilities = predict_batches(
        model,
        loader,
        device,
        amp=config["amp"],
        flip_tta=args.flip_tta,
    )

    if ids != frame["image_id"].tolist():
        raise RuntimeError("Prediction order changed.")

    predicted = probabilities.argmax(axis=1)

    metrics = {
        "accuracy": float(
            accuracy_score(targets, predicted)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(targets, predicted)
        ),
        "macro_f1": float(
            f1_score(targets, predicted, average="macro")
        ),
        "loss": float(
            log_loss(
                targets,
                probabilities,
                labels=range(len(CLASSES)),
            )
        ),
        "errors": int((targets != predicted).sum()),
        "checkpoint": str(args.checkpoint),
        "split_dir": checkpoint["split_dir"],
        "split_fingerprint": actual,
        "flip_tta": args.flip_tta,
        "note": (
            "Own-split score; not directly comparable "
            "with original-split accuracy."
        ),
    }

    output_dir.mkdir(parents=True)
    write_json(output_dir / "metrics.json", metrics)

    report = classification_report(
        targets,
        predicted,
        labels=range(len(CLASSES)),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).T.to_csv(
        output_dir / "classification_report.csv"
    )

    predictions = frame.copy()
    predictions["predicted_class"] = [
        CLASSES[index] for index in predicted
    ]
    predictions["confidence"] = probabilities.max(axis=1)

    predictions.to_csv(
        output_dir / "predictions.csv",
        index=False,
    )

    predictions.loc[
        targets != predicted
    ].to_csv(
        output_dir / "misclassified.csv",
        index=False,
    )

    np.savez_compressed(
        output_dir / "probabilities.npz",
        image_id=np.asarray(ids),
        probabilities=probabilities,
        classes=np.asarray(CLASSES),
        targets=targets,
    )

    matrix = confusion_matrix(
        targets,
        predicted,
        labels=range(len(CLASSES)),
    )

    pd.DataFrame(
        matrix,
        index=CLASSES,
        columns=CLASSES,
    ).to_csv(
        output_dir / "confusion_matrix.csv"
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(
        matrix,
        display_labels=CLASSES,
    ).plot(
        ax=ax,
        xticks_rotation=45,
        colorbar=False,
    )

    fig.tight_layout()
    fig.savefig(
        output_dir / "confusion_matrix.png",
        dpi=150,
    )
    plt.close(fig)

    print(metrics)
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
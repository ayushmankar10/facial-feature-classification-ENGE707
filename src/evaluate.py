import argparse
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

from config import (
    CLASSES,
    PROJECT_ROOT,
    SPLIT_DIR,
)
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
    split_fingerprint,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on validation data."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
    )
    parser.add_argument(
        "--flip-tta",
        action="store_true",
    )

    args = parser.parse_args()

    torch.set_num_threads(4)

    checkpoint = load_checkpoint(args.checkpoint)

    if checkpoint.get("smoke"):
        raise ValueError(
            "Smoke-test checkpoints are not "
            "real evaluation experiments."
        )

    if (
        checkpoint["split_fingerprint"]
        != split_fingerprint()
    ):
        raise ValueError(
            "Current splits differ from the "
            "checkpoint's training splits."
        )

    config = checkpoint["config"]
    device = choose_device(args.allow_cpu)

    frame = read_split(
        SPLIT_DIR / "validation.csv"
    )

    dataset = FacialDataset(
        frame,
        build_transforms(config["image_size"]),
    )

    loader = make_loader(
        dataset,
        args.batch_size,
        args.workers,
        cuda=device.type == "cuda",
    )

    model = build_model(
        config["architecture"],
        pretrained=False,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.to(
        device,
        memory_format=torch.channels_last,
    )

    ids, targets, probabilities = predict_batches(
        model,
        loader,
        device,
        amp=config["amp"],
        flip_tta=args.flip_tta,
    )

    predicted = probabilities.argmax(axis=1)

    metrics = {
        "accuracy": float(
            accuracy_score(targets, predicted)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                targets, predicted
            )
        ),
        "macro_f1": float(
            f1_score(
                targets,
                predicted,
                average="macro",
            )
        ),
        "loss": float(
            log_loss(
                targets,
                probabilities,
                labels=range(len(CLASSES)),
            )
        ),
        "checkpoint": str(args.checkpoint),
        "flip_tta": args.flip_tta,
    }

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
            f"Evaluation already exists: {output_dir}. "
            "Archive it before rerunning."
        )

    output_dir.mkdir(parents=True)

    write_json(
        output_dir / "metrics.json",
        metrics,
    )

    report = classification_report(
        targets,
        predicted,
        labels=range(len(CLASSES)),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        output_dir / "classification_report.csv"
    )

    predictions = frame.copy()

    if predictions["image_id"].tolist() != ids:
        raise RuntimeError(
            "Validation prediction order changed."
        )

    predictions["predicted_class"] = [
        CLASSES[index] for index in predicted
    ]
    predictions["confidence"] = (
        probabilities.max(axis=1)
    )

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

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

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
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
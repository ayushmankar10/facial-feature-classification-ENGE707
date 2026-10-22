import argparse
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import (
    CLASSES,
    IMAGE_EXTENSIONS,
    PROJECT_ROOT,
    SAMPLE_PATH,
    TEST_DIR,
)
from dataset import (
    FacialDataset,
    build_transforms,
    make_loader,
)
from model import build_model
from utils import (
    choose_device,
    load_checkpoint,
    predict_batches,
    validate_submission,
    write_json,
)


def make_test_frame(sample, test_dir=TEST_DIR):
    if (
        list(sample.columns) != ["image_id", "class"]
        or sample.empty
    ):
        raise ValueError(
            "Expected a nonempty sample template "
            "with image_id,class columns."
        )

    if (
        sample["image_id"].isna().any()
        or sample["image_id"].duplicated().any()
    ):
        raise ValueError(
            "Template IDs must be present and unique."
        )

    if not test_dir.is_dir():
        raise FileNotFoundError(test_dir)

    lookup = {}

    # Locate test inputs solely for final prediction.
    for path in sorted(test_dir.rglob("*")):
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            if path.stem in lookup:
                raise ValueError(
                    f"Ambiguous test filename: {path.stem}"
                )

            lookup[path.stem] = path

    rows = []

    for image_id in sample["image_id"]:
        key = Path(str(image_id)).stem

        if key not in lookup:
            raise FileNotFoundError(
                f"No test file for template ID {image_id}"
            )

        rows.append({
            "image_id": str(image_id),
            "path": (
                lookup[key]
                .relative_to(PROJECT_ROOT)
                .as_posix()
            ),
        })

    if len(lookup) != len(rows):
        raise ValueError(
            "Test image count does not match "
            "the submission template."
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Kaggle submission CSV."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--output",
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
        "--flip-tta",
        action="store_true",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
    )

    args = parser.parse_args()

    metadata_path = args.output.with_suffix(".json")

    if (
        args.output.exists()
        or metadata_path.exists()
    ):
        raise FileExistsError(
            "Submission or metadata already exists. "
            "Choose a new output name."
        )

    torch.set_num_threads(4)

    device = choose_device(args.allow_cpu)

    sample = pd.read_csv(
        SAMPLE_PATH,
        dtype={"image_id": "string"},
    )

    frame = make_test_frame(sample)

    average_probabilities = None

    for checkpoint_path in args.checkpoint:
        print(
            f"Predicting with: {checkpoint_path}",
            flush=True,
        )

        checkpoint = load_checkpoint(
            checkpoint_path
        )

        if checkpoint.get("smoke"):
            raise ValueError(
                "Cannot submit predictions from "
                "a smoke-test checkpoint."
            )

        config = checkpoint["config"]

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

        dataset = FacialDataset(
            frame,
            build_transforms(config["image_size"]),
            labelled=False,
        )

        loader = make_loader(
            dataset,
            args.batch_size,
            args.workers,
            cuda=device.type == "cuda",
        )

        ids, _, probabilities = predict_batches(
            model,
            loader,
            device,
            amp=config["amp"],
            flip_tta=args.flip_tta,
        )

        if ids != sample["image_id"].tolist():
            raise RuntimeError(
                "Prediction order does not "
                "match the template."
            )

        if average_probabilities is None:
            average_probabilities = probabilities
        else:
            average_probabilities += probabilities

        del loader, dataset, model, checkpoint

        if device.type == "cuda":
            torch.cuda.empty_cache()

    average_probabilities /= len(args.checkpoint)

    predicted_indices = (
        average_probabilities.argmax(axis=1)
    )

    submission = pd.DataFrame({
        "image_id": sample["image_id"],
        "class": [
            CLASSES[index]
            for index in predicted_indices
        ],
    })

    validate_submission(submission, sample)

    args.output.parent.mkdir(
        parents=True, exist_ok=True
    )

    submission.to_csv(
        args.output,
        index=False,
    )

    write_json(
        metadata_path,
        {
            "checkpoints": [
                str(path)
                for path in args.checkpoint
            ],
            "flip_tta": args.flip_tta,
            "ensemble": (
                "Equal arithmetic mean of probabilities"
            ),
            "rows": len(submission),
            "classes": list(CLASSES),
        },
    )

    print(
        f"\nValidated {len(submission):,} predictions."
    )
    print(f"Saved to: {args.output}")
    print(
        "Upload this CSV through Kaggle "
        "Submit Prediction."
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
"""Generate Kaggle submissions with optional selective flip TTA."""

import argparse
import multiprocessing
from pathlib import Path

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


def matching_key(value):
    """Match numeric filenames with or without leading zeros."""
    stem = Path(str(value)).stem

    if stem.isdecimal():
        return str(int(stem))

    return stem


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

    test_dir = Path(test_dir)

    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"Test folder not found: {test_dir}"
        )

    lookup = {}

    for path in sorted(test_dir.rglob("*")):
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            key = matching_key(path.name)

            if key in lookup:
                raise ValueError(
                    f"Multiple test files match ID {key}: "
                    f"{lookup[key].name} and {path.name}"
                )

            lookup[key] = path

    if not lookup:
        raise FileNotFoundError(
            f"No test images found inside: {test_dir}"
        )

    template_keys = sample["image_id"].map(matching_key)

    if template_keys.duplicated().any():
        raise ValueError(
            "Multiple template IDs refer to the same "
            "numeric image ID."
        )

    rows = []

    for image_id in sample["image_id"]:
        key = matching_key(image_id)

        if key not in lookup:
            examples = [
                path.name
                for path in list(lookup.values())[:5]
            ]
            raise FileNotFoundError(
                f"No test file for template ID {image_id}. "
                f"Example test filenames: {examples}"
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
            f"Found {len(lookup):,} test images, but the "
            f"template contains {len(rows):,} rows."
        )

    print(
        f"Matched all {len(rows):,} template IDs "
        "to test images.",
        flush=True,
    )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        required=True,
        help="One or more trained model checkpoints.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output submission CSV path.",
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

    tta_options = parser.add_mutually_exclusive_group()

    tta_options.add_argument(
        "--flip-tta",
        action="store_true",
        help="Apply horizontal-flip TTA to every model.",
    )
    tta_options.add_argument(
        "--v2s-flip-tta",
        action="store_true",
        help=(
            "Apply horizontal-flip TTA only to "
            "EfficientNetV2-S checkpoints at image size 224."
        ),
    )
    tta_options.add_argument(
        "--flip-tta-checkpoint",
        type=Path,
        nargs="+",
        default=[],
        help=(
            "Apply horizontal-flip TTA only to these "
            "checkpoints. Each must also appear in --checkpoint."
        ),
    )

    parser.add_argument(
        "--allow-cpu",
        action="store_true",
    )

    args = parser.parse_args()

    if args.batch_size < 1 or args.workers < 0:
        parser.error(
            "batch-size must be positive and "
            "workers must be zero or greater."
        )

    if args.output.suffix.lower() != ".csv":
        parser.error("--output must end in .csv")

    resolved_checkpoints = [
        path.resolve() for path in args.checkpoint
    ]

    if len(set(resolved_checkpoints)) != len(resolved_checkpoints):
        parser.error(
            "The same checkpoint was supplied more than once."
        )

    selected_tta = {
        path.resolve()
        for path in args.flip_tta_checkpoint
    }

    if not selected_tta.issubset(set(resolved_checkpoints)):
        parser.error(
            "TTA checkpoints must also appear in --checkpoint."
        )

    metadata_path = args.output.with_suffix(".json")

    if args.output.exists() or metadata_path.exists():
        raise FileExistsError(
            "Submission or metadata already exists. "
            "Choose a new output filename."
        )

    for checkpoint_path in args.checkpoint:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

    torch.set_num_threads(4)
    device = choose_device(args.allow_cpu)

    sample = pd.read_csv(
        SAMPLE_PATH,
        dtype={"image_id": "string"},
    )

    frame = make_test_frame(sample)

    average_probabilities = None
    model_details = []

    for checkpoint_path in args.checkpoint:
        print(
            f"\nPredicting with: {checkpoint_path}",
            flush=True,
        )

        checkpoint = load_checkpoint(checkpoint_path)

        if checkpoint.get("smoke"):
            raise ValueError(
                "Cannot submit predictions from "
                "a smoke-test checkpoint."
            )

        config = checkpoint["config"]
        architecture = config["architecture"]
        resize_mode = config.get("resize_mode", "stretch")

        use_flip_tta = (
            args.flip_tta
            or checkpoint_path.resolve() in selected_tta
            or (
                args.v2s_flip_tta
                and architecture == "efficientnet_v2_s"
                and config["image_size"] == 224
            )
        )

        model = build_model(
            architecture,
            pretrained=False,
        )

        model.load_state_dict(
            checkpoint["model_state"]
        )

        model.to(
            device,
            memory_format=torch.channels_last,
        )

        print(
            f"Architecture: {architecture}",
            flush=True,
        )
        print(
            f"Model device: {next(model.parameters()).device}",
            flush=True,
        )
        print(
            f"Resize mode: {resize_mode}",
            flush=True,
        )
        print(
            f"Flip TTA: {use_flip_tta}",
            flush=True,
        )

        dataset = FacialDataset(
            frame,
            build_transforms(
                config["image_size"],
                resize_mode=resize_mode,
            ),
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
            flip_tta=use_flip_tta,
        )

        if ids != sample["image_id"].tolist():
            raise RuntimeError(
                "Prediction order does not match "
                "the sample template."
            )

        if average_probabilities is None:
            average_probabilities = probabilities
        else:
            if probabilities.shape != average_probabilities.shape:
                raise RuntimeError(
                    "Models produced different prediction shapes."
                )

            average_probabilities += probabilities

        model_details.append({
            "checkpoint": str(checkpoint_path),
            "architecture": architecture,
            "image_size": config["image_size"],
            "resize_mode": resize_mode,
            "flip_tta": bool(use_flip_tta),
        })

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
        parents=True,
        exist_ok=True,
    )

    submission.to_csv(
        args.output,
        index=False,
    )

    write_json(
        metadata_path,
        {
            "checkpoints": [
                str(path) for path in args.checkpoint
            ],
            "models": model_details,
            "flip_tta": args.flip_tta,
            "v2s_flip_tta": args.v2s_flip_tta,
            "flip_tta_checkpoints": [
                str(path)
                for path in args.flip_tta_checkpoint
            ],
            "ensemble": (
                "Equal arithmetic mean of model probabilities. "
                "For models using TTA, original and flipped "
                "probabilities are averaged first."
            ),
            "rows": len(submission),
            "classes": list(CLASSES),
        },
    )

    print(
        f"\nValidated {len(submission):,} predictions."
    )
    print(
        f"Submission saved to: {args.output}"
    )
    print(
        "Upload this CSV through Kaggle Submit Prediction."
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
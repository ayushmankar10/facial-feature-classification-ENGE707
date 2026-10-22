import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from config import CLASSES, SPLIT_DIR


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Helps choose efficient CUDA convolution algorithms.
    # GPU runs can still have small nondeterministic differences.
    torch.backends.cudnn.benchmark = (
        torch.cuda.is_available()
    )


def choose_device(allow_cpu=False):
    if torch.cuda.is_available():
        device = torch.device("cuda")

        print(
            f"GPU: {torch.cuda.get_device_name(0)}",
            flush=True,
        )

    elif allow_cpu:
        device = torch.device("cpu")
        print(
            "CPU mode explicitly enabled.",
            flush=True,
        )

    else:
        raise RuntimeError(
            "CUDA is unavailable. Check the selected "
            ".venv. CPU mode requires --allow-cpu."
        )

    return device


def write_json(path, payload):
    path = Path(path)

    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def split_fingerprint():
    return {
        name: hashlib.sha256(
            (SPLIT_DIR / name).read_bytes()
        ).hexdigest()
        for name in [
            "train.csv",
            "validation.csv",
        ]
    }


def save_checkpoint(path, payload):
    path = Path(path)

    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    temporary = path.with_suffix(".tmp")

    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["classes"] != list(CLASSES):
        raise ValueError(
            "Checkpoint class order does not match "
            "this project."
        )

    return checkpoint


@torch.inference_mode()
def predict_batches(
    model,
    loader,
    device,
    amp=True,
    flip_tta=False,
):
    model.eval()

    ids = []
    targets = []
    probabilities = []

    for batch_index, batch in enumerate(
        loader, start=1
    ):
        images, labels, image_ids = batch

        images = images.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )

        with torch.autocast(
            device_type=device.type,
            enabled=amp and device.type == "cuda",
        ):
            logits = model(images)
            probs = logits.float().softmax(dim=1)

            if flip_tta:
                flipped_logits = model(
                    images.flip(-1)
                )

                flipped_probs = (
                    flipped_logits.float().softmax(dim=1)
                )

                probs = (
                    probs + flipped_probs
                ) / 2

        probabilities.append(
            probs.cpu().numpy()
        )

        ids.extend(image_ids)
        targets.extend(labels.tolist())

        if batch_index % 100 == 0:
            print(
                f"Predicted {len(ids):,}/"
                f"{len(loader.dataset):,}",
                flush=True,
            )

    return (
        ids,
        np.asarray(targets),
        np.concatenate(probabilities),
    )


def validate_submission(submission, sample):
    if list(submission.columns) != [
        "image_id", "class"
    ]:
        raise ValueError(
            "Submission columns must be "
            "image_id,class in that order."
        )

    if (
        len(submission) != len(sample)
        or submission.isna().any().any()
    ):
        raise ValueError(
            "Submission has missing predictions "
            "or the wrong row count."
        )

    if submission["image_id"].duplicated().any():
        raise ValueError(
            "Submission contains duplicate image IDs."
        )

    actual_ids = (
        submission["image_id"].astype(str).tolist()
    )
    expected_ids = (
        sample["image_id"].astype(str).tolist()
    )

    if actual_ids != expected_ids:
        raise ValueError(
            "Submission IDs or order differ "
            "from the sample template."
        )

    if not submission["class"].isin(CLASSES).all():
        raise ValueError(
            "Submission contains unknown classes."
        )
"""Image transforms, datasets and CPU data loaders."""

import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from config import (
    CLASS_TO_INDEX,
    MEAN,
    PROJECT_ROOT,
    STD,
)


class ResizeWithPadding:
    """Preserve image proportions and centre on a square."""

    def __init__(self, image_size):
        self.image_size = int(image_size)

        if self.image_size < 1:
            raise ValueError("image_size must be positive.")

        self.fill = tuple(round(value * 255) for value in MEAN)

    def __call__(self, image):
        image = image.convert("RGB")
        width, height = image.size
        scale = self.image_size / max(width, height)

        new_width = max(
            1,
            min(self.image_size, round(width * scale)),
        )
        new_height = max(
            1,
            min(self.image_size, round(height * scale)),
        )

        resized = image.resize(
            (new_width, new_height),
            resample=Image.Resampling.BILINEAR,
        )

        canvas = Image.new(
            "RGB",
            (self.image_size, self.image_size),
            color=self.fill,
        )

        canvas.paste(
            resized,
            (
                (self.image_size - new_width) // 2,
                (self.image_size - new_height) // 2,
            ),
        )

        return canvas


def build_transforms(
    image_size,
    training=False,
    resize_mode="stretch",
):
    if resize_mode == "stretch":
        resize = transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
    elif resize_mode == "pad":
        resize = ResizeWithPadding(image_size)
    else:
        raise ValueError(
            f"Unknown resize mode: {resize_mode}"
        )

    operations = [resize]

    if training:
        operations.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.04, 0.04),
                scale=(0.95, 1.05),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.1,
            ),
        ])

    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    return transforms.Compose(operations)


class FacialDataset(Dataset):
    def __init__(
        self,
        frame,
        transform,
        root=PROJECT_ROOT,
        labelled=True,
    ):
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform
        self.root = root
        self.labelled = labelled

        required = ["image_id", "path"]

        if labelled:
            required.append("class")

        missing = set(required) - set(self.frame.columns)

        if missing:
            raise ValueError(
                f"Dataset is missing columns: {missing}"
            )

        if (
            self.frame.empty
            or self.frame[required].isna().any().any()
        ):
            raise ValueError(
                "Dataset is empty or has missing fields."
            )

        if self.frame["image_id"].duplicated().any():
            raise ValueError(
                "Dataset contains duplicate image IDs."
            )

        if labelled and not self.frame["class"].isin(
            CLASS_TO_INDEX
        ).all():
            raise ValueError(
                "Dataset contains an unknown class."
            )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image_path = self.root / row["path"]

        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))

        target = (
            CLASS_TO_INDEX[row["class"]]
            if self.labelled
            else -1
        )

        return tensor, target, str(row["image_id"])


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)

    # Prevent every worker starting many CPU threads.
    torch.set_num_threads(1)


def make_loader(
    dataset,
    batch_size,
    workers,
    shuffle=False,
    cuda=False,
    seed=42,
):
    if workers < 0 or batch_size < 1:
        raise ValueError(
            "workers must be >= 0 and batch_size "
            "must be positive."
        )

    options = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": cuda,
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(seed),
        "drop_last": False,
    }

    if workers:
        options.update({
            "persistent_workers": True,
            "prefetch_factor": 2,
            "multiprocessing_context": "spawn",
        })

    return DataLoader(dataset, **options)


def read_split(path):
    return pd.read_csv(
        path,
        dtype={"image_id": "string"},
    )
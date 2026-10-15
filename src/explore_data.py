from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# Resolve paths relative to the project, regardless of where Python is run.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "enge-707-2026-s-2-cat-and-dog-facial-feature-classification-ch"
)
TRAIN_DIR = DATA_DIR / "train"
LABELS_PATH = DATA_DIR / "train_labels.csv"

EXPECTED_CLASSES = {
    "CatEye", "DogEye", "CatFace", "DogFace",
    "CatMouth", "DogMouth", "CatNose", "DogNose",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    # Read IDs as strings to preserve leading zeros.
    labels = pd.read_csv(LABELS_PATH, dtype="string")

    if set(labels.columns) != {"image_id", "class"}:
        raise ValueError("Expected CSV columns: image_id,class")

    print(f"Label rows: {len(labels):,}")
    print("\nMissing values:")
    print(labels.isna().sum())
    print(f"\nDuplicate image IDs: {labels['image_id'].duplicated().sum()}")

    if labels.isna().any().any():
        raise ValueError("Resolve missing labels or IDs before continuing.")

    if labels["image_id"].duplicated().any():
        raise ValueError("Resolve duplicate image IDs before continuing.")

    if set(labels["class"]) != EXPECTED_CLASSES:
        raise ValueError("Labels do not match the eight expected classes.")

    counts = labels["class"].value_counts().sort_index()
    summary = counts.to_frame(name="images")
    summary["percentage"] = (counts / len(labels) * 100).round(2)
    print("\nClass distribution:")
    print(summary.to_string())

    if not TRAIN_DIR.is_dir():
        raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")

    # Search recursively to support train/train nesting.
    image_paths = sorted(
        path for path in TRAIN_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    print(f"\nTraining image files found: {len(image_paths):,}")

    # Match IDs with filenames, allowing IDs with or without extensions.
    images_by_id = defaultdict(list)
    for path in image_paths:
        images_by_id[path.stem].append(path)

    ambiguous_ids = {
        image_id for image_id, paths in images_by_id.items()
        if len(paths) > 1
    }
    if ambiguous_ids:
        raise ValueError(
            "Multiple images share a filename stem. Examples: "
            f"{sorted(ambiguous_ids)[:10]}"
        )

    label_ids = labels["image_id"].map(lambda value: Path(value).stem)
    missing_ids = sorted(set(label_ids) - set(images_by_id))
    extra_ids = sorted(set(images_by_id) - set(label_ids))

    print(f"Labels without a matching image: {len(missing_ids):,}")
    print(f"Images without a matching label: {len(extra_ids):,}")

    if missing_ids:
        print("Missing ID examples:", missing_ids[:10])
        raise ValueError("Some labelled images could not be found.")

    if extra_ids:
        print("Unlabelled ID examples:", extra_ids[:10])

    # Show one reproducibly selected image per class.
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    for ax, class_name in zip(axes.flat, sorted(EXPECTED_CLASSES)):
        row = labels.loc[labels["class"] == class_name].sample(
            n=1, random_state=42
        ).iloc[0]
        image_id = Path(row["image_id"]).stem
        image_path = images_by_id[image_id][0]

        with Image.open(image_path) as image:
            width, height = image.size
            ax.imshow(image.convert("RGB"))

        ax.set_title(f"{class_name}\n{image_id}: {width} × {height}")
        ax.axis("off")

    fig.suptitle("Training samples: one image per class")
    plt.tight_layout()
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "training_samples.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSample grid saved to: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
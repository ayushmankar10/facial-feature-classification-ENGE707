from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs/runs/effnet_b0_20epochs/evaluation"
TRAIN_CSV = ROOT / "outputs/splits/train.csv"


def image_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def difference_hash(path):
    """Describe brightness changes in a small grayscale image."""
    with Image.open(path) as image:
        image = ImageOps.grayscale(image)
        image = image.resize(
            (9, 8),
            resample=Image.Resampling.LANCZOS,
        )
        pixels = np.asarray(image)

    return (pixels[:, 1:] > pixels[:, :-1]).reshape(-1)


def main():
    train = pd.read_csv(TRAIN_CSV, dtype={"image_id": str})
    errors = pd.read_csv(
        RUN_DIR / "misclassified.csv",
        dtype={"image_id": str},
    )

    hashes = np.empty((len(train), 64), dtype=bool)

    print(f"Checking {len(train):,} training images...")

    for index, row in enumerate(train.itertuples(index=False)):
        hashes[index] = difference_hash(image_path(row.path))

        if (index + 1) % 5000 == 0:
            print(f"Processed {index + 1:,}/{len(train):,}")

    matches = []

    for row in errors.itertuples(index=False):
        query = difference_hash(image_path(row.path))
        distances = np.count_nonzero(hashes != query, axis=1)

        # Select candidates across all classes.
        nearest = np.argsort(distances, kind="stable")[:3]

        for rank, index in enumerate(nearest, start=1):
            candidate = train.iloc[index]

            matches.append({
                "validation_id": row.image_id,
                "validation_class": getattr(row, "_1"),
                "predicted_class": row.predicted_class,
                "validation_path": row.path,
                "rank": rank,
                "training_id": candidate["image_id"],
                "training_class": candidate["class"],
                "training_path": candidate["path"],
                "hash_distance": int(distances[index]),
            })

    output = RUN_DIR / "similar_training_candidates.csv"
    pd.DataFrame(matches).to_csv(output, index=False)

    print(f"\nSaved: {output}")
    print(
        "Lower distances indicate more similar brightness patterns."
        "\nThese are candidates for visual review, not confirmed duplicates."
    )


if __name__ == "__main__":
    main()
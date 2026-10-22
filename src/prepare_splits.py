import argparse
import hashlib
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    AUDIT_PATH,
    CLASSES,
    LABELS_PATH,
    SPLIT_DIR,
)


def make_splits(
    audit,
    validation_fraction=0.2,
    seed=42,
):
    required = {
        "image_id",
        "class",
        "path",
        "width",
        "height",
        "pixel_hash",
        "error",
    }

    missing_columns = required - set(audit.columns)

    if missing_columns:
        raise ValueError(
            f"Audit is missing columns: {missing_columns}"
        )

    if not 0 < validation_fraction < 1:
        raise ValueError(
            "Validation fraction must be between 0 and 1."
        )

    if audit.empty:
        raise ValueError("Audit contains no images.")

    if (
        audit["image_id"].isna().any()
        or audit["image_id"].duplicated().any()
    ):
        raise ValueError(
            "Audit image IDs must be present and unique."
        )

    if audit["error"].fillna("").ne("").any():
        raise ValueError(
            "Resolve audit failures before splitting."
        )

    metadata_columns = [
        "class",
        "path",
        "pixel_hash",
        "width",
        "height",
    ]

    if audit[metadata_columns].isna().any().any():
        raise ValueError(
            "Audit contains missing image metadata."
        )

    if not (
        audit[["width", "height"]] > 0
    ).all().all():
        raise ValueError(
            "Image dimensions must be positive."
        )

    if set(audit["class"]) != set(CLASSES):
        raise ValueError(
            "Expected all eight classes."
        )

    conflicting_labels = (
        audit.groupby("pixel_hash")["class"]
        .nunique()
        .gt(1)
    )

    if conflicting_labels.any():
        raise ValueError(
            "Identical images have conflicting labels."
        )

    ordered = audit.sort_values(
        "image_id"
    ).copy()

    # Select one representative per exact duplicate group.
    groups = ordered.drop_duplicates("pixel_hash")

    _, validation_groups = train_test_split(
        groups,
        test_size=validation_fraction,
        random_state=seed,
        stratify=groups["class"],
    )

    is_validation = ordered["pixel_hash"].isin(
        validation_groups["pixel_hash"]
    )

    columns = [
        "image_id",
        "class",
        "path",
        "width",
        "height",
        "pixel_hash",
    ]

    train = ordered.loc[
        ~is_validation, columns
    ].reset_index(drop=True)

    validation = ordered.loc[
        is_validation, columns
    ].reset_index(drop=True)

    overlap = (
        set(train["pixel_hash"])
        & set(validation["pixel_hash"])
    )

    if overlap:
        raise RuntimeError(
            "Exact duplicate content crosses the split."
        )

    if len(train) + len(validation) != len(audit):
        raise RuntimeError(
            "Some images were lost during splitting."
        )

    return train, validation


def main():
    parser = argparse.ArgumentParser(
        description="Create fixed training/validation splits."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
    )
    args = parser.parse_args()

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
            "Original labels have missing values "
            "or duplicate IDs."
        )

    expected = (
        labels.set_index("image_id")["class"]
        .sort_index()
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

    train, validation = make_splits(
        audit,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )

    metadata = {
        "method": (
            "Stratified split of exact RGB-pixel hash groups"
        ),
        "seed": args.seed,
        "validation_fraction_of_groups": (
            args.validation_fraction
        ),
        "audit_sha256": hashlib.sha256(
            AUDIT_PATH.read_bytes()
        ).hexdigest(),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "classes": list(CLASSES),
        "limitation": (
            "Near-duplicates and related crops "
            "are not grouped by this audit."
        ),
    }

    payloads = {
        "train.csv": train.to_csv(index=False),
        "validation.csv": validation.to_csv(index=False),
        "split_metadata.json": (
            json.dumps(metadata, indent=2) + "\n"
        ),
    }

    # Prevent accidentally changing the established split.
    for name, contents in payloads.items():
        target = SPLIT_DIR / name

        if (
            target.exists()
            and target.read_text(encoding="utf-8")
            != contents
        ):
            raise FileExistsError(
                f"{target} already contains a different "
                "split. Archive it explicitly before "
                "creating another."
            )

    SPLIT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    for name, contents in payloads.items():
        (SPLIT_DIR / name).write_text(
            contents,
            encoding="utf-8",
            newline="",
        )

    print(f"Training images: {len(train):,}")
    print(f"Validation images: {len(validation):,}")
    print("Overlapping exact content hashes: 0")

    counts = pd.DataFrame({
        "train": train["class"].value_counts(),
        "validation": validation["class"].value_counts(),
    }).sort_index()

    print("\nClass counts:")
    print(counts.to_string())
    print(f"\nSaved to: {SPLIT_DIR}")


if __name__ == "__main__":
    main()
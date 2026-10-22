import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

from explore_data import (
    IMAGE_EXTENSIONS,
    LABELS_PATH,
    PROJECT_ROOT,
    TRAIN_DIR,
)


def main():
    labels = pd.read_csv(LABELS_PATH, dtype="string")
    label_map = dict(zip(labels["image_id"], labels["class"]))

    if not TRAIN_DIR.is_dir():
        raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")

    paths = sorted(
        path for path in TRAIN_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not paths:
        raise ValueError("No training images found.")

    records = []
    print(f"Checking {len(paths):,} training images...", flush=True)

    for number, path in enumerate(paths, start=1):
        record = {
            "image_id": path.stem,
            "class": label_map.get(path.stem, ""),
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "width": None,
            "height": None,
            "mode": "",
            "pixel_hash": "",
            "error": "",
        }

        try:
            # Check file integrity, then reopen to fully decode the image.
            with Image.open(path) as image:
                image.verify()

            with Image.open(path) as image:
                image.load()
                record["width"], record["height"] = image.size
                record["mode"] = image.mode

                # Hash original-size RGB pixels, including dimensions.
                # This detects identical decoded content across files.
                rgb = image.convert("RGB")
                digest = hashlib.sha256()
                digest.update(f"{rgb.width},{rgb.height}:".encode())
                digest.update(rgb.tobytes())
                record["pixel_hash"] = digest.hexdigest()

        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
            record["error"] = f"{type(error).__name__}: {error}"

        records.append(record)

        if number % 5000 == 0 or number == len(paths):
            print(f"Checked {number:,}/{len(paths):,}", flush=True)

    audit = pd.DataFrame(records)
    valid = audit.loc[audit["error"] == ""].copy()
    failed = audit.loc[audit["error"] != ""].copy()

    duplicates = valid.loc[
        valid.duplicated("pixel_hash", keep=False)
    ].sort_values(["pixel_hash", "image_id"])

    output_dir = PROJECT_ROOT / "outputs" / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    audit.to_csv(output_dir / "image_audit.csv", index=False)
    failed.to_csv(output_dir / "failed_images.csv", index=False)
    duplicates.to_csv(output_dir / "exact_duplicates.csv", index=False)

    print("\nAUDIT SUMMARY")
    print(f"Images checked: {len(audit):,}")
    print(f"Successfully decoded: {len(valid):,}")
    print(f"Failed integrity or decoding checks: {len(failed):,}")
    print(f"Images in exact duplicate groups: {len(duplicates):,}")
    print(f"Exact duplicate groups: {duplicates['pixel_hash'].nunique():,}")

    if not duplicates.empty:
        conflicting_groups = (
            duplicates.groupby("pixel_hash")["class"].nunique() > 1
        ).sum()
        print(f"Duplicate groups with conflicting labels: {conflicting_groups:,}")

    if not valid.empty:
        print("\nImage dimensions:")
        print(
            valid[["width", "height"]]
            .astype(int)
            .describe()
            .round(1)
            .to_string()
        )

        print("\nOriginal colour modes:")
        print(valid["mode"].value_counts().to_string())

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
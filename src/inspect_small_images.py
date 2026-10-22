import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from config import AUDIT_PATH, PROJECT_ROOT


def main():
    audit = pd.read_csv(
        AUDIT_PATH,
        dtype={"image_id": "string"},
    )

    valid = audit.loc[
        audit["error"].fillna("") == ""
    ].copy()

    valid["short_side"] = valid[
        ["width", "height"]
    ].min(axis=1)

    smallest = valid.sort_values(
        ["short_side", "image_id"]
    ).head(24)

    fig, axes = plt.subplots(
        4, 6, figsize=(15, 10)
    )

    for ax in axes.flat:
        ax.axis("off")

    for ax, (_, row) in zip(
        axes.flat, smallest.iterrows()
    ):
        image_path = PROJECT_ROOT / row["path"]

        with Image.open(image_path) as image:
            ax.imshow(
                image.convert("RGB"),
                interpolation="nearest",
            )

        ax.set_title(
            f"{row['image_id']} | {row['class']}\n"
            f"{int(row['width'])} × {int(row['height'])}",
            fontsize=9,
        )

    fig.suptitle("24 smallest training images")
    fig.tight_layout()

    output_path = (
        PROJECT_ROOT / "outputs" / "smallest_images.png"
    )
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
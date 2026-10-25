from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps

from explore_data import PROJECT_ROOT


def main():
    audit_path = PROJECT_ROOT / "outputs" / "audit" / "image_audit.csv"
    audit = pd.read_csv(audit_path)
    valid = audit.loc[audit["error"].fillna("").eq("")]

    classes = [
        "CatEye", "CatFace", "CatMouth", "CatNose",
        "DogEye", "DogFace", "DogMouth", "DogNose",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    for ax, class_name in zip(axes.flat, classes):
        candidates = valid.loc[valid["class"].eq(class_name)]
        row = candidates.sample(n=1, random_state=42).iloc[0]
        image_path = PROJECT_ROOT / Path(row["path"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            # Preserve the image shape and add white padding.
            display = ImageOps.pad(
                image,
                (256, 256),
                method=Image.Resampling.LANCZOS,
                color="white",
            )
            ax.imshow(display)

        ax.set_title(class_name, fontsize=13)
        ax.axis("off")

    fig.tight_layout()
    output_dir = PROJECT_ROOT / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sample_grid.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
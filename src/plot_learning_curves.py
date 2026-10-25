import matplotlib.pyplot as plt
import pandas as pd

from config import PROJECT_ROOT


def main():
    runs = {
        "EfficientNet-B0": "effnet_b0_20epochs",
        "EfficientNetV2-S": "effnet_v2s_20epochs",
        "EfficientNet-B0 (split 123)": "effnet_b0_split123_20epochs",
    }

    fig, axes = plt.subplots(
        3, 2, figsize=(12, 11), constrained_layout=True
    )

    for row, (name, folder) in enumerate(runs.items()):
        history_path = (
            PROJECT_ROOT
            / "outputs"
            / "runs"
            / folder
            / "history.csv"
        )
        history = pd.read_csv(history_path)

        best = history.sort_values(
            ["val_accuracy", "val_loss"],
            ascending=[False, True],
        ).iloc[0]

        accuracy_ax = axes[row, 0]
        accuracy_ax.plot(
            history["epoch"],
            history["train_accuracy"] * 100,
            label="Training",
        )
        accuracy_ax.plot(
            history["epoch"],
            history["val_accuracy"] * 100,
            label="Validation",
        )
        accuracy_ax.scatter(
            best["epoch"],
            best["val_accuracy"] * 100,
            color="black",
            marker="*",
            s=90,
            zorder=5,
            label="Selected checkpoint",
        )
        accuracy_ax.set_title(f"{name}: accuracy")
        accuracy_ax.set_ylabel("Accuracy (%)")
        accuracy_ax.set_ylim(80, 100.5)

        loss_ax = axes[row, 1]
        loss_ax.plot(
            history["epoch"],
            history["train_loss"],
            label="Training (label smoothing)",
        )
        loss_ax.plot(
            history["epoch"],
            history["val_loss"],
            label="Validation (plain cross-entropy)",
        )
        loss_ax.set_title(f"{name}: loss")
        loss_ax.set_ylabel("Loss")
        loss_ax.set_ylim(bottom=0)

        for ax in (accuracy_ax, loss_ax):
            ax.set_xlabel("Epoch")
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)

    output_dir = PROJECT_ROOT / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    for extension in ("png", "pdf"):
        output_path = output_dir / f"learning_curves.{extension}"
        fig.savefig(output_path, dpi=300)
        print(f"Saved: {output_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
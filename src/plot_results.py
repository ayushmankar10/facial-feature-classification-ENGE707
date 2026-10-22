import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from config import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(
        description="Plot an experiment's learning curves."
    )

    parser.add_argument(
        "--run",
        required=True,
    )

    args = parser.parse_args()

    runs_root = (
        PROJECT_ROOT / "outputs" / "runs"
    )
    run_dir = runs_root / args.run

    if (
        run_dir.resolve().parent
        != runs_root.resolve()
    ):
        raise ValueError(
            "Use a run folder name, not a path."
        )

    history = pd.read_csv(
        run_dir / "history.csv"
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4)
    )

    for ax, metric in zip(
        axes, ["loss", "accuracy"]
    ):
        ax.plot(
            history["epoch"],
            history[f"train_{metric}"],
            label="Training",
        )

        ax.plot(
            history["epoch"],
            history[f"val_{metric}"],
            label="Validation",
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(alpha=0.3)

    axes[0].set_title(
        "Training uses label smoothing; "
        "validation uses plain cross entropy",
        fontsize=9,
    )

    fig.tight_layout()

    output_path = (
        run_dir / "learning_curves.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
    )
    plt.close(fig)

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
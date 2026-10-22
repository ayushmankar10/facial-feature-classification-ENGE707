import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import CLASSES
from prepare_splits import make_splits


class SplitTests(unittest.TestCase):
    def make_frame(self):
        rows = []

        for class_name in CLASSES:
            for index in range(20):
                rows.append({
                    "image_id": f"{class_name}_{index}",
                    "class": class_name,
                    "path": "unused.png",
                    "width": 8,
                    "height": 8,
                    "pixel_hash": (
                        f"{class_name}_{index // 2}"
                    ),
                    "error": "",
                })

        return pd.DataFrame(rows)

    def test_duplicate_groups_stay_together(self):
        frame = self.make_frame()
        train, validation = make_splits(frame)

        overlap = (
            set(train["pixel_hash"])
            & set(validation["pixel_hash"])
        )

        self.assertFalse(overlap)

        combined_ids = (
            set(train["image_id"])
            | set(validation["image_id"])
        )

        self.assertEqual(
            combined_ids,
            set(frame["image_id"]),
        )

        self.assertEqual(
            set(validation["class"]),
            set(CLASSES),
        )

    def test_split_is_reproducible(self):
        frame = self.make_frame()

        first = make_splits(frame)
        second = make_splits(
            frame.sample(
                frac=1,
                random_state=7,
            )
        )

        for first_frame, second_frame in zip(
            first, second
        ):
            pd.testing.assert_frame_equal(
                first_frame,
                second_frame,
            )

    def test_conflicting_labels_rejected(self):
        frame = self.make_frame()

        frame.loc[0, "pixel_hash"] = (
            frame.loc[20, "pixel_hash"]
        )

        with self.assertRaises(ValueError):
            make_splits(frame)

    def test_corrupted_images_rejected(self):
        frame = self.make_frame()
        frame.loc[0, "error"] = "decode error"

        with self.assertRaises(ValueError):
            make_splits(frame)


class PipelineTests(unittest.TestCase):
    def test_training_step_and_prediction_shapes(self):
        import torch
        from PIL import Image

        from dataset import (
            FacialDataset,
            build_transforms,
            make_loader,
        )
        from model import build_model
        from train import run_epoch
        from utils import predict_batches

        torch.set_num_threads(1)

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            rows = []

            for index, class_name in enumerate(CLASSES):
                filename = f"{index}.png"

                Image.new(
                    "RGB",
                    (12 + index, 18),
                    (index * 25, 30, 50),
                ).save(root / filename)

                rows.append({
                    "image_id": f"{index:05}",
                    "path": filename,
                    "class": class_name,
                })

            frame = pd.DataFrame(rows)

            dataset = FacialDataset(
                frame,
                build_transforms(32),
                root=root,
            )

            loader = make_loader(
                dataset,
                batch_size=4,
                workers=0,
            )

            model = build_model(
                "efficientnet_b0",
                pretrained=False,
            )

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=0.001,
            )

            scaler = torch.amp.GradScaler(
                "cuda",
                enabled=False,
            )

            statistics = run_epoch(
                model,
                loader,
                torch.nn.CrossEntropyLoss(),
                torch.device("cpu"),
                optimizer=optimizer,
                scaler=scaler,
                amp=False,
                max_batches=1,
            )

            self.assertGreater(
                statistics["loss"], 0
            )

            ids, _, probabilities = predict_batches(
                model,
                loader,
                torch.device("cpu"),
                amp=False,
            )

            self.assertEqual(
                ids,
                frame["image_id"].tolist(),
            )

            self.assertEqual(
                probabilities.shape,
                (8, 8),
            )

            probability_sums = torch.from_numpy(
                probabilities.sum(axis=1)
            )

            self.assertTrue(
                torch.allclose(
                    probability_sums,
                    torch.ones(8),
                    atol=1e-5,
                )
            )

    def test_submission_validation(self):
        from utils import validate_submission

        sample = pd.DataFrame({
            "image_id": ["00001", "00002"],
            "class": ["placeholder", "placeholder"],
        })

        submission = pd.DataFrame({
            "image_id": ["00001", "00002"],
            "class": ["CatEye", "DogEye"],
        })

        validate_submission(
            submission,
            sample,
        )

        with self.assertRaises(ValueError):
            validate_submission(
                submission.iloc[::-1],
                sample,
            )


if __name__ == "__main__":
    unittest.main()
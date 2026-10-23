"""Shared paths and saved training settings."""

from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "enge-707-2026-s-2-cat-and-dog-facial-feature-classification-ch"
)

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
LABELS_PATH = DATA_DIR / "train_labels.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"

AUDIT_PATH = (
    PROJECT_ROOT / "outputs" / "audit" / "image_audit.csv"
)
SPLIT_DIR = PROJECT_ROOT / "outputs" / "splits"

CLASSES = (
    "CatEye",
    "CatFace",
    "CatMouth",
    "CatNose",
    "DogEye",
    "DogFace",
    "DogMouth",
    "DogNose",
)

CLASS_TO_INDEX = {
    name: index for index, name in enumerate(CLASSES)
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    architecture: str = "efficientnet_b0"
    image_size: int = 224
    resize_mode: str = "stretch"
    batch_size: int = 16
    workers: int = 2
    cpu_threads: int = 4
    epochs: int = 20
    freeze_epochs: int = 2
    learning_rate: float = 0.0003
    head_learning_rate: float = 0.001
    weight_decay: float = 0.0001
    label_smoothing: float = 0.05
    patience: int = 5
    seed: int = 42
    pretrained: bool = True
    amp: bool = True
    min_train_side: int = 0

    def to_dict(self):
        return asdict(self)
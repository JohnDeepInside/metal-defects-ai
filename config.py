"""
Centralized configuration for the Metal Surface Defect Detection project.
"""

import argparse
from pathlib import Path


# ============================================================
# Default paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_DIR = OUTPUT_DIR / "results"

# ============================================================
# Dataset
# ============================================================
CLASS_NAMES = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted_Surface",
    "Rolled-in_Scale",
    "Scratches",
]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224

# ============================================================
# ImageNet normalization
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================
# Training defaults
# ============================================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
WEIGHT_DECAY = 1e-4
PATIENCE = 5          # early-stopping patience
SCHEDULER_PATIENCE = 2
SCHEDULER_FACTOR = 0.5
NUM_WORKERS = 2
SEED = 42

# ============================================================
# Data split ratios
# ============================================================
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with sensible defaults."""
    parser = argparse.ArgumentParser(
        description="Train / evaluate a metal-defect classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Root directory of the dataset")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Directory for checkpoints and results")

    # Training
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help="Early-stopping patience (0 = disabled)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)

    # Model
    parser.add_argument("--model", type=str, default="efficientnet_b0",
                        choices=["efficientnet_b0", "mobilenet_v2", "resnet18"],
                        help="Backbone architecture")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use ImageNet-pretrained weights")

    # Misc
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to checkpoint to resume training from")

    return parser.parse_args()

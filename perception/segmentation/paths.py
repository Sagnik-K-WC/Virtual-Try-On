from pathlib import Path
import os


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# LIP dataset location
#
# Set the LIP_DATASET_ROOT environment variable on each machine.
# Example on Windows:
#
# $env:LIP_DATASET_ROOT="C:\path\to\LIP"
#
# If it is not set, the code expects the dataset at:
# <project-root>/datasets/LIP

LIP_DATASET_ROOT = Path(
    os.environ.get(
        "LIP_DATASET_ROOT",
        PROJECT_ROOT / "datasets" / "LIP"
    )
)


# Common LIP directories

LIP_IMAGES_ROOT = (
    LIP_DATASET_ROOT
    / "TrainVal_images"
    / "TrainVal_images"
)

LIP_MASKS_ROOT = (
    LIP_DATASET_ROOT
    / "TrainVal_parsing_annotations"
    / "TrainVal_parsing_annotations"
    / "TrainVal_parsing_annotations"
)


LIP_TRAIN_IMAGES = LIP_IMAGES_ROOT / "train_images"
LIP_VAL_IMAGES = LIP_IMAGES_ROOT / "val_images"

LIP_TRAIN_MASKS = LIP_MASKS_ROOT / "train_segmentations"
LIP_VAL_MASKS = LIP_MASKS_ROOT / "val_segmentations"


# Local experiment outputs
#
# These stay outside GitHub because checkpoints and generated files
# are ignored by .gitignore.

EXPERIMENTS_ROOT = PROJECT_ROOT / "local_experiments"
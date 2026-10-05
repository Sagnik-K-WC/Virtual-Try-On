from pathlib import Path

import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
)

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_ID = "pirocheto/schp-lip-20"

IMAGE_DIR = (
    ROOT
    / "datasets/LIP/TrainVal_images/"
    / "TrainVal_images/val_images"
)

MASK_DIR = (
    ROOT
    / "datasets/LIP/TrainVal_parsing_annotations/"
    / "TrainVal_parsing_annotations/"
    / "TrainVal_parsing_annotations/"
    / "val_segmentations"
)

NUM_CLASSES = 20

CLASS_NAMES = [
    "Background",
    "Hat",
    "Hair",
    "Glove",
    "Sunglasses",
    "Upper-clothes",
    "Dress",
    "Coat",
    "Socks",
    "Pants",
    "Jumpsuits",
    "Scarf",
    "Skirt",
    "Face",
    "Left-arm",
    "Right-arm",
    "Left-leg",
    "Right-leg",
    "Left-shoe",
    "Right-shoe",
]

# Start with 100 images.
# Change to None later for the complete 10,000-image validation set.
MAX_IMAGES = None

# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("SCHP LIP EVALUATION")
print("===================")
print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading processor...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

print("Loading model...")

model = AutoModelForSemanticSegmentation.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
).to(device)

model.eval()

print("Model loaded.")


# ============================================================
# FIND VALIDATION IMAGES
# ============================================================

image_files = sorted(IMAGE_DIR.glob("*.jpg"))

if MAX_IMAGES is not None:
    image_files = image_files[:MAX_IMAGES]

print("\nImages to evaluate:", len(image_files))


# ============================================================
# CONFUSION MATRIX
# ============================================================

# Rows = ground truth
# Columns = prediction

confusion_matrix = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=np.int64,
)


# ============================================================
# INFERENCE
# ============================================================

for index, image_path in enumerate(image_files, start=1):

    mask_path = MASK_DIR / f"{image_path.stem}.png"

    if not mask_path.exists():
        print("WARNING: Missing mask:", mask_path)
        continue

    image = Image.open(image_path).convert("RGB")
    ground_truth = np.array(
        Image.open(mask_path),
        dtype=np.int64,
    )

    # Prepare image for SCHP
    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Run inference
    with torch.inference_mode():
        outputs = model(**inputs)

    # SCHP refined parsing output
    logits = outputs.parsing_logits

    # Resize prediction to original mask resolution
    resized_logits = torch.nn.functional.interpolate(
        logits,
        size=ground_truth.shape,
        mode="bilinear",
        align_corners=False,
    )

    prediction = (
        resized_logits
        .argmax(dim=1)
        .squeeze()
        .cpu()
        .numpy()
    )

    # Update confusion matrix
    valid = (
        (ground_truth >= 0)
        & (ground_truth < NUM_CLASSES)
        & (prediction >= 0)
        & (prediction < NUM_CLASSES)
    )

    gt = ground_truth[valid].flatten()
    pred = prediction[valid].flatten()

    combined = gt * NUM_CLASSES + pred

    counts = np.bincount(
        combined,
        minlength=NUM_CLASSES * NUM_CLASSES,
    )

    confusion_matrix += counts.reshape(
        NUM_CLASSES,
        NUM_CLASSES,
    )

    if index % 10 == 0 or index == 1:
        print(
            f"Processed {index}/{len(image_files)}"
        )


# ============================================================
# CALCULATE IoU
# ============================================================

intersection = np.diag(confusion_matrix)

ground_truth_pixels = confusion_matrix.sum(axis=1)

predicted_pixels = confusion_matrix.sum(axis=0)

union = (
    ground_truth_pixels
    + predicted_pixels
    - intersection
)

iou = np.divide(
    intersection,
    union,
    out=np.zeros(NUM_CLASSES, dtype=float),
    where=union != 0,
)

mIoU = iou.mean()


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("SEGMENTATION RESULTS")
print("====================")

for class_id, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_id:2d} "
        f"{class_name:<15} "
        f"IoU: {iou[class_id]:.4f}"
    )

print("--------------------")
print(f"mIoU: {mIoU:.4f}")
print(f"mIoU: {mIoU * 100:.2f}%")


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = ROOT / "schp_evaluation.txt"

with open(results_file, "w", encoding="utf-8") as f:

    f.write("SCHP LIP Evaluation\n")
    f.write("===================\n\n")

    f.write(
        f"Images evaluated: {len(image_files)}\n"
    )

    f.write(
        f"mIoU: {mIoU:.4f} "
        f"({mIoU * 100:.2f}%)\n\n"
    )

    for class_id, class_name in enumerate(CLASS_NAMES):

        f.write(
            f"{class_id:2d} "
            f"{class_name:<15} "
            f"IoU: {iou[class_id]:.4f}\n"
        )

print("\nResults saved to:")
print(results_file)
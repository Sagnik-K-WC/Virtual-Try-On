import os
import numpy as np
from PIL import Image

import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "pirocheto/schp-lip-20"

CHECKPOINT_PATH = (
    r"C:\Users\iamsa\VirtualTryOn\schp_500_training"
    r"\schp_lip_20_finetuned_500.pt"
)

IMAGE_DIR = (
    r"C:\Users\iamsa\VirtualTryOn\datasets\LIP"
    r"\TrainVal_images\TrainVal_images\val_images"
)

MASK_DIR = (
    r"C:\Users\iamsa\VirtualTryOn\datasets\LIP"
    r"\TrainVal_parsing_annotations"
    r"\TrainVal_parsing_annotations"
    r"\TrainVal_parsing_annotations"
    r"\val_segmentations"
)

IMAGE_SIZE = 473
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
    "Right-shoe"
]


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD PROCESSOR
# ============================================================

print("\nLoading processor...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)


# ============================================================
# LOAD PRETRAINED MODEL
# ============================================================

print("Loading pretrained SCHP model...")

model = AutoModelForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)


# ============================================================
# LOAD OUR FINE-TUNED CHECKPOINT
# ============================================================

print("Loading fine-tuned checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


# ============================================================
# FIND VALID VALIDATION PAIRS
# ============================================================

all_images = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
])


valid_images = []

for filename in all_images:

    mask_name = (
        os.path.splitext(filename)[0]
        + ".png"
    )

    mask_path = os.path.join(
        MASK_DIR,
        mask_name
    )

    if os.path.exists(mask_path):
        valid_images.append(filename)


print(
    "\nValidation image/mask pairs:",
    len(valid_images)
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion_matrix = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=np.int64
)


# ============================================================
# EVALUATION
# ============================================================

print("\nStarting validation...")
print("------------------------------------------")


with torch.no_grad():

    for index, filename in enumerate(valid_images):

        image_path = os.path.join(
            IMAGE_DIR,
            filename
        )

        mask_path = os.path.join(
            MASK_DIR,
            os.path.splitext(filename)[0] + ".png"
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        # ----------------------------------------------------
        # Load ground-truth mask
        # ----------------------------------------------------

        mask = Image.open(
            mask_path
        )

        mask = mask.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST
        )

        mask = np.array(
            mask,
            dtype=np.int64
        )

        # ----------------------------------------------------
        # Process image
        # ----------------------------------------------------

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        pixel_values = inputs[
            "pixel_values"
        ].to(device)

        # ----------------------------------------------------
        # Model inference
        # ----------------------------------------------------

        outputs = model(
            pixel_values=pixel_values
        )

        logits = outputs.parsing_logits

        # ----------------------------------------------------
        # Resize logits if necessary
        # ----------------------------------------------------

        if logits.shape[-2:] != mask.shape:

            logits = torch.nn.functional.interpolate(
                logits,
                size=mask.shape,
                mode="bilinear",
                align_corners=False
            )

        # ----------------------------------------------------
        # Convert logits to predicted classes
        # ----------------------------------------------------

        prediction = torch.argmax(
            logits,
            dim=1
        )[0].cpu().numpy()

        # ----------------------------------------------------
        # Update confusion matrix
        # ----------------------------------------------------

        valid = (
            (mask >= 0)
            & (mask < NUM_CLASSES)
        )

        true_labels = mask[valid]
        pred_labels = prediction[valid]

        indices = (
            NUM_CLASSES * true_labels
            + pred_labels
        )

        counts = np.bincount(
            indices,
            minlength=NUM_CLASSES ** 2
        )

        confusion_matrix += counts.reshape(
            NUM_CLASSES,
            NUM_CLASSES
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (index + 1) % 500 == 0
            or index == 0
        ):

            print(
                f"Processed "
                f"{index + 1}/{len(valid_images)}"
            )


# ============================================================
# CALCULATE IoU
# ============================================================

ious = []

for class_id in range(NUM_CLASSES):

    intersection = confusion_matrix[
        class_id,
        class_id
    ]

    ground_truth = confusion_matrix[
        class_id,
        :
    ].sum()

    predicted = confusion_matrix[
        :,
        class_id
    ].sum()

    union = (
        ground_truth
        + predicted
        - intersection
    )

    if union == 0:

        iou = float("nan")

    else:

        iou = (
            intersection
            / union
        )

    ious.append(iou)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n==========================================")
print("FINE-TUNED SCHP VALIDATION RESULTS")
print("==========================================")

for class_id, iou in enumerate(ious):

    if np.isnan(iou):

        print(
            f"{class_id:2d} "
            f"{CLASS_NAMES[class_id]:15s} "
            f"IoU: N/A"
        )

    else:

        print(
            f"{class_id:2d} "
            f"{CLASS_NAMES[class_id]:15s} "
            f"IoU: {iou:.4f}"
        )


valid_ious = [
    x for x in ious
    if not np.isnan(x)
]

miou = np.mean(valid_ious)

print("\n------------------------------------------")

print(
    f"mIoU: {miou:.4f}"
)

print(
    f"mIoU: {miou * 100:.2f}%"
)

print("==========================================")
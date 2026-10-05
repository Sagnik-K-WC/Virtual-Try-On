import os
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "pirocheto/schp-lip-20"

IMAGE_DIR = (
    r"C:\Users\iamsa\VirtualTryOn\datasets\LIP"
    r"\TrainVal_images\TrainVal_images\val_images"
)

MASK_DIR = (
    r"C:\Users\iamsa\VirtualTryOn\datasets\LIP"
    r"\TrainVal_parsing_annotations"
    r"\TrainVal_parsing_annotations"
    r"\TrainVal_parsing_annotations\val_segmentations"
)

VAL_ID_FILE = (
    r"C:\Users\iamsa\VirtualTryOn\datasets\LIP"
    r"\TrainVal_images\val_id.txt"
)

IMAGE_SIZE = (473, 473)
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
# SCHP CENTER / SCALE
# ============================================================

def get_center_scale(width, height):

    aspect_ratio = (
        IMAGE_SIZE[1] / IMAGE_SIZE[0]
    )

    x = 0
    y = 0
    w = width - 1
    h = height - 1

    center = np.zeros(
        2,
        dtype=np.float32
    )

    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:

        h = w / aspect_ratio

    elif w < aspect_ratio * h:

        w = h * aspect_ratio

    scale = np.array(
        [w, h],
        dtype=np.float32
    )

    return center, scale


# ============================================================
# SCHP AFFINE TRANSFORM
# ============================================================

def get_dir(src_point, rot_rad):

    sn = np.sin(rot_rad)
    cs = np.cos(rot_rad)

    result = [0, 0]

    result[0] = (
        src_point[0] * cs
        - src_point[1] * sn
    )

    result[1] = (
        src_point[0] * sn
        + src_point[1] * cs
    )

    return result


def get_3rd_point(a, b):

    direct = a - b

    return (
        b
        + np.array(
            [-direct[1], direct[0]],
            dtype=np.float32
        )
    )


def get_affine_transform(
    center,
    scale,
    rot,
    output_size,
    inv=0
):

    scale_tmp = scale

    src_w = scale_tmp[0]

    dst_w = output_size[1]
    dst_h = output_size[0]

    rot_rad = (
        np.pi * rot / 180
    )

    src_dir = get_dir(
        [0, src_w * -0.5],
        rot_rad
    )

    dst_dir = np.array(
        [
            0,
            (dst_w - 1) * -0.5
        ],
        np.float32
    )

    src = np.zeros(
        (3, 2),
        dtype=np.float32
    )

    dst = np.zeros(
        (3, 2),
        dtype=np.float32
    )

    src[0, :] = center

    src[1, :] = (
        center
        + src_dir
    )

    dst[0, :] = [
        (dst_w - 1) * 0.5,
        (dst_h - 1) * 0.5
    ]

    dst[1, :] = (
        np.array(
            [
                (dst_w - 1) * 0.5,
                (dst_h - 1) * 0.5
            ]
        )
        + dst_dir
    )

    src[2:, :] = get_3rd_point(
        src[0, :],
        src[1, :]
    )

    dst[2:, :] = get_3rd_point(
        dst[0, :],
        dst[1, :]
    )

    if inv:

        trans = cv2.getAffineTransform(
            np.float32(dst),
            np.float32(src)
        )

    else:

        trans = cv2.getAffineTransform(
            np.float32(src),
            np.float32(dst)
        )

    return trans


# ============================================================
# LOAD SCHP PROCESSOR
# ============================================================

print("\nLoading SCHP processor...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)


# ============================================================
# LOAD SCHP MODEL
# ============================================================

print("Loading SCHP model...")

model = AutoModelForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = model.to(device)

model.eval()


# ============================================================
# LOAD VALIDATION IDS
# ============================================================

with open(
    VAL_ID_FILE,
    "r"
) as f:

    val_ids = [
        line.strip()
        for line in f
        if line.strip()
    ]


print(
    "\nValidation images:",
    len(val_ids)
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

print("\nStarting SCHP official-style evaluation")
print("with horizontal flip test-time augmentation...")
print("------------------------------------------")


with torch.no_grad():

    for index, image_id in enumerate(val_ids):

        # ----------------------------------------------------
        # File paths
        # ----------------------------------------------------

        image_path = os.path.join(
            IMAGE_DIR,
            image_id + ".jpg"
        )

        mask_path = os.path.join(
            MASK_DIR,
            image_id + ".png"
        )

        # ----------------------------------------------------
        # Load original image
        # ----------------------------------------------------

        image_bgr = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )

        if image_bgr is None:

            print(
                "WARNING: Could not read:",
                image_path
            )

            continue

        height, width = (
            image_bgr.shape[:2]
        )

        # ----------------------------------------------------
        # Calculate center and scale
        # ----------------------------------------------------

        center, scale = (
            get_center_scale(
                width,
                height
            )
        )

        # ----------------------------------------------------
        # Create SCHP affine transformation
        # ----------------------------------------------------

        transform = get_affine_transform(
            center,
            scale,
            0,
            IMAGE_SIZE
        )

        # ----------------------------------------------------
        # Crop original image to 473x473
        # ----------------------------------------------------

        cropped = cv2.warpAffine(
            image_bgr,
            transform,
            (
                IMAGE_SIZE[1],
                IMAGE_SIZE[0]
            ),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # ----------------------------------------------------
        # Convert BGR → RGB
        # ----------------------------------------------------

        cropped_rgb = cv2.cvtColor(
            cropped,
            cv2.COLOR_BGR2RGB
        )

        # ====================================================
        # ORIGINAL IMAGE INFERENCE
        # ====================================================

        original_inputs = processor(
            images=cropped_rgb,
            return_tensors="pt"
        )

        original_pixel_values = (
            original_inputs["pixel_values"]
            .to(device)
        )

        original_outputs = model(
            pixel_values=original_pixel_values
        )

        original_logits = (
            original_outputs.logits
        )

        if original_logits.shape[-2:] != IMAGE_SIZE:

            original_logits = F.interpolate(
                original_logits,
                size=IMAGE_SIZE,
                mode="bilinear",
                align_corners=True
            )


        # ====================================================
        # HORIZONTAL FLIP INFERENCE
        # ====================================================

        flipped_image = cv2.flip(
            cropped_rgb,
            1
        )

        flipped_inputs = processor(
            images=flipped_image,
            return_tensors="pt"
        )

        flipped_pixel_values = (
            flipped_inputs["pixel_values"]
            .to(device)
        )

        flipped_outputs = model(
            pixel_values=flipped_pixel_values
        )

        flipped_logits = (
            flipped_outputs.logits
        )

        if flipped_logits.shape[-2:] != IMAGE_SIZE:

            flipped_logits = F.interpolate(
                flipped_logits,
                size=IMAGE_SIZE,
                mode="bilinear",
                align_corners=True
            )


        # ----------------------------------------------------
        # Flip prediction back to original orientation
        # ----------------------------------------------------

        flipped_logits = torch.flip(
            flipped_logits,
            dims=[3]
        )


        # ====================================================
        # COMBINE ORIGINAL + FLIPPED PREDICTIONS
        # ====================================================

        logits = (
            original_logits
            + flipped_logits
        ) / 2.0


        # ====================================================
        # CONVERT LOGITS TO NUMPY
        # ====================================================

        logits = (
            logits[0]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )


        # ====================================================
        # MAP LOGITS BACK TO ORIGINAL IMAGE SIZE
        # ====================================================

        inverse_transform = (
            get_affine_transform(
                center,
                scale,
                0,
                IMAGE_SIZE,
                inv=1
            )
        )

        restored_logits = np.zeros(
            (
                height,
                width,
                NUM_CLASSES
            ),
            dtype=np.float32
        )

        for class_id in range(NUM_CLASSES):

            restored_logits[:, :, class_id] = (
                cv2.warpAffine(
                    logits[:, :, class_id],
                    inverse_transform,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
            )


        # ====================================================
        # FINAL CLASS PREDICTION
        # ====================================================

        prediction = np.argmax(
            restored_logits,
            axis=2
        ).astype(np.int64)


        # ====================================================
        # LOAD ORIGINAL GROUND-TRUTH MASK
        # ====================================================

        ground_truth = np.array(
            Image.open(mask_path),
            dtype=np.int64
        )


        # ====================================================
        # REMOVE IGNORE LABEL
        # ====================================================

        valid = (
            ground_truth != 255
        )

        true_labels = (
            ground_truth[valid]
        )

        predicted_labels = (
            prediction[valid]
        )


        # ====================================================
        # UPDATE CONFUSION MATRIX
        # ====================================================

        encoded = (
            NUM_CLASSES * true_labels
            + predicted_labels
        )

        counts = np.bincount(
            encoded,
            minlength=NUM_CLASSES ** 2
        )

        confusion_matrix += (
            counts.reshape(
                NUM_CLASSES,
                NUM_CLASSES
            )
        )


        # ====================================================
        # PROGRESS
        # ====================================================

        if (
            index == 0
            or (index + 1) % 500 == 0
        ):

            print(
                f"Processed "
                f"{index + 1}/{len(val_ids)}"
            )


# ============================================================
# CALCULATE IoU
# ============================================================

ious = []

for class_id in range(NUM_CLASSES):

    intersection = (
        confusion_matrix[
            class_id,
            class_id
        ]
    )

    ground_truth_total = (
        confusion_matrix[
            class_id,
            :
        ].sum()
    )

    prediction_total = (
        confusion_matrix[
            :,
            class_id
        ].sum()
    )

    union = (
        ground_truth_total
        + prediction_total
        - intersection
    )

    if union == 0:

        iou = float("nan")

    else:

        iou = (
            intersection / union
        )

    ious.append(iou)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n==========================================")
print("SCHP OFFICIAL-STYLE + FLIP VALIDATION")
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

miou = np.mean(
    valid_ious
)

print("\n------------------------------------------")

print(
    f"mIoU: {miou:.4f}"
)

print(
    f"mIoU: {miou * 100:.2f}%"
)

print("==========================================")
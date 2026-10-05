import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
)

from perception.segmentation.paths import (
    LIP_TRAIN_IMAGES,
    LIP_TRAIN_MASKS,
    EXPERIMENTS_ROOT,
)


# ============================================================
# 1. MODEL CONFIGURATION
# ============================================================

MODEL_ID = "pirocheto/schp-lip-20"


# ============================================================
# 2. INPUT IMAGE AND GROUND-TRUTH MASK
# ============================================================

IMAGE_PATH = LIP_TRAIN_IMAGES / "1000_1234574.jpg"
MASK_PATH = LIP_TRAIN_MASKS / "1000_1234574.png"


# ============================================================
# 3. OUTPUT CONFIGURATION
# ============================================================

OUTPUT_DIR = EXPERIMENTS_ROOT / "schp_inference"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "schp_comparison.png"


# ============================================================
# 4. CHECK INPUT FILES
# ============================================================

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Input image not found:\n{IMAGE_PATH}"
    )

if not MASK_PATH.exists():
    raise FileNotFoundError(
        f"Ground-truth mask not found:\n{MASK_PATH}"
    )


# ============================================================
# 5. SELECT DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("SCHP HUMAN PARSING INFERENCE")
print("=" * 60)

print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 6. LOAD IMAGE AND GROUND TRUTH
# ============================================================

print("\nLoading image...")

image = Image.open(IMAGE_PATH).convert("RGB")

# IMPORTANT:
# The segmentation mask contains class IDs from 0 to 19.
# It must not be converted to RGB.
ground_truth = np.array(Image.open(MASK_PATH))


# ============================================================
# 7. LOAD SCHP PROCESSOR
# ============================================================

print("Loading processor...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)


# ============================================================
# 8. LOAD SCHP MODEL
# ============================================================

print("Loading model...")

model = AutoModelForSemanticSegmentation.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = model.to(device)
model.eval()


# ============================================================
# 9. PREPROCESS IMAGE
# ============================================================

print("Preprocessing image...")

inputs = processor(
    images=image,
    return_tensors="pt",
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


# ============================================================
# 10. RUN INFERENCE
# ============================================================

print("Running SCHP inference...")

with torch.inference_mode():
    outputs = model(**inputs)


# ============================================================
# 11. GET PARSING OUTPUT
# ============================================================

# SCHP produces parsing logits for 20 LIP classes.
logits = outputs.parsing_logits

print("Parsing logits:", logits.shape)


# ============================================================
# 12. RESIZE PREDICTION TO ORIGINAL IMAGE SIZE
# ============================================================

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


# ============================================================
# 13. CREATE COMPARISON VISUALIZATION
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 6),
)

axes[0].imshow(image)
axes[0].set_title("Original Image")

axes[1].imshow(
    ground_truth,
    cmap="tab20",
    vmin=0,
    vmax=19,
)
axes[1].set_title("LIP Ground Truth")

axes[2].imshow(
    prediction,
    cmap="tab20",
    vmin=0,
    vmax=19,
)
axes[2].set_title("SCHP Prediction")


for ax in axes:
    ax.axis("off")


plt.tight_layout()


# ============================================================
# 14. SAVE COMPARISON
# ============================================================

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight",
)

print("\nComparison saved to:")
print(OUTPUT_PATH)


# ============================================================
# 15. DISPLAY COMPARISON
# ============================================================

plt.show()


# ============================================================
# 16. REPORT CLASSES
# ============================================================

print("\nGround-truth classes:")
print(np.unique(ground_truth))

print("\nPredicted classes:")
print(np.unique(prediction))

print("\nInference complete.")
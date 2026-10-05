import torch
from PIL import Image

from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
)

from perception.segmentation.paths import LIP_TRAIN_IMAGES


# ============================================================
# 1. MODEL CONFIGURATION
# ============================================================

MODEL_ID = "pirocheto/schp-lip-20"


# ============================================================
# 2. IMAGE CONFIGURATION
# ============================================================

IMAGE_PATH = LIP_TRAIN_IMAGES / "1000_1234574.jpg"


# ============================================================
# 3. LOAD PROCESSOR
# ============================================================

print("Loading processor...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)


# ============================================================
# 4. LOAD MODEL
# ============================================================

print("Loading model...")

model = AutoModelForSemanticSegmentation.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)


# ============================================================
# 5. SELECT DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(device)
model.eval()


# ============================================================
# 6. LOAD IMAGE
# ============================================================

print("Loading image...")

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )

image = Image.open(IMAGE_PATH).convert("RGB")


# ============================================================
# 7. PREPROCESS IMAGE
# ============================================================

inputs = processor(
    images=image,
    return_tensors="pt",
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


# ============================================================
# 8. DISPLAY INPUT INFORMATION
# ============================================================

print()
print("INPUT INFORMATION")
print("==================")
print("Image:", IMAGE_PATH)
print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

print("Input shape:", inputs["pixel_values"].shape)
print("Input dtype:", inputs["pixel_values"].dtype)
print(
    "Input range:",
    inputs["pixel_values"].min().item(),
    "to",
    inputs["pixel_values"].max().item(),
)


# ============================================================
# 9. RUN MODEL
# ============================================================

print()
print("Running forward pass...")

with torch.inference_mode():
    outputs = model(**inputs)


# ============================================================
# 10. DISPLAY OUTPUT INFORMATION
# ============================================================

print()
print("OUTPUT INFORMATION")
print("===================")
print("Output type:", type(outputs))
print("Parsing logits:", outputs.parsing_logits.shape)
print("Edge logits:", outputs.edge_logits.shape)
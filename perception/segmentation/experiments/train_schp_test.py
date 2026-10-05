import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image

from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "pirocheto/schp-lip-20"

IMAGE_PATH = (
    "datasets/LIP/TrainVal_images/"
    "TrainVal_images/train_images/1000_1234574.jpg"
)

MASK_PATH = (
    "datasets/LIP/TrainVal_parsing_annotations/"
    "TrainVal_parsing_annotations/"
    "TrainVal_parsing_annotations/"
    "train_segmentations/1000_1234574.png"
)

IMAGE_SIZE = 473


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD PROCESSOR
# ============================================================

print("\nLoading processor...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)


# ============================================================
# LOAD PRETRAINED SCHP MODEL
# ============================================================

print("Loading model...")

model = AutoModelForSemanticSegmentation.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
).to(device)


# ============================================================
# TRAINING MODE
# ============================================================

model.train()

# SCHP contains BatchNorm layers.
#
# With batch size = 1, some deep feature maps become:
#
#     [1, 512, 1, 1]
#
# BatchNorm cannot calculate training statistics from
# only one value per channel.
#
# Therefore, keep BatchNorm layers in evaluation mode
# while the rest of the model remains trainable.

for module in model.modules():
    if isinstance(
        module,
        torch.nn.modules.batchnorm._BatchNorm
    ):
        module.eval()


# ============================================================
# LOAD IMAGE
# ============================================================

print("Loading image...")

image = Image.open(
    IMAGE_PATH
).convert("RGB")


# ============================================================
# LOAD GROUND-TRUTH SEGMENTATION MASK
# ============================================================

mask = Image.open(
    MASK_PATH
)

# Resize the ground-truth mask to the same spatial
# resolution expected by SCHP.
#
# IMPORTANT:
# NEAREST interpolation must be used for segmentation
# masks because the pixel values are class IDs.

mask = mask.resize(
    (IMAGE_SIZE, IMAGE_SIZE),
    Image.Resampling.NEAREST,
)

# Convert the mask into a PyTorch tensor.
#
# Shape:
#     [1, 473, 473]
#
# Values are integer class IDs from 0 to 19.

mask = torch.from_numpy(
    np.array(
        mask,
        dtype=np.int64,
    )
).unsqueeze(0)

mask = mask.to(device)


# ============================================================
# PREPARE IMAGE FOR SCHP
# ============================================================

inputs = processor(
    images=image,
    return_tensors="pt",
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


print()
print("INPUT INFORMATION")
print("==================")
print("Input:", inputs["pixel_values"].shape)
print("Mask:", mask.shape)


# ============================================================
# CREATE OPTIMIZER
# ============================================================

# We use a very small learning rate because we are
# fine-tuning an already pretrained SCHP model.

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-4,
    momentum=0.9,
    weight_decay=5e-4,
)


# ============================================================
# SAVE ONE PARAMETER BEFORE TRAINING
# ============================================================

parameter_before = (
    next(model.parameters())
    .detach()
    .clone()
)


# ============================================================
# FORWARD PASS
# ============================================================

print("\nRunning training step...")

optimizer.zero_grad(
    set_to_none=True
)

# Mixed precision reduces GPU memory usage.

with torch.amp.autocast(
    device_type="cuda",
    dtype=torch.float16,
):

    outputs = model(
        **inputs
    )

    parsing_logits = outputs.parsing_logits

    # --------------------------------------------------------
    # Segmentation loss
    # --------------------------------------------------------
    #
    # Cross entropy compares the model's predicted class
    # for every pixel with the ground-truth class.
    #
    # parsing_logits:
    #     [1, 20, 473, 473]
    #
    # mask:
    #     [1, 473, 473]

    loss = F.cross_entropy(
        parsing_logits,
        mask,
        ignore_index=255,
    )


print(
    "Parsing logits:",
    parsing_logits.shape,
)

print(
    "Loss:",
    loss.item(),
)


# ============================================================
# BACKWARD PASS
# ============================================================

print("Running backward pass...")

loss.backward()

print("Backward pass completed.")


# ============================================================
# OPTIMIZER UPDATE
# ============================================================

optimizer.step()

print("Optimizer step completed.")


# ============================================================
# CHECK WHETHER PARAMETERS CHANGED
# ============================================================

parameter_after = (
    next(model.parameters())
    .detach()
    .clone()
)

parameter_change = (
    parameter_after - parameter_before
).abs().sum().item()


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("TRAINING TEST RESULT")
print("====================")

print(
    "Initial loss:",
    loss.item(),
)

print(
    "Parameter change:",
    parameter_change,
)

if parameter_change > 0:

    print(
        "SUCCESS: model parameters changed."
    )

    print(
        "The SCHP model can be trained "
        "in this environment."
    )

else:

    print(
        "ERROR: model parameters did not change."
    )


# ============================================================
# GPU MEMORY
# ============================================================

if device == "cuda":

    peak_memory = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    print(
        "Peak GPU memory:",
        round(peak_memory, 2),
        "GB",
    )
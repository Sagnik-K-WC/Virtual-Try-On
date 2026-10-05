import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "pirocheto/schp-lip-20"

IMAGE_DIR = r"C:\Users\iamsa\VirtualTryOn\datasets\LIP\TrainVal_images\TrainVal_images\train_images"
MASK_DIR = r"C:\Users\iamsa\VirtualTryOn\datasets\LIP\TrainVal_parsing_annotations\TrainVal_parsing_annotations\TrainVal_parsing_annotations\train_segmentations"

OUTPUT_DIR = r"C:\Users\iamsa\VirtualTryOn\schp_500_training"

NUM_IMAGES = 500
IMAGE_SIZE = 473
BATCH_SIZE = 1
NUM_WORKERS = 0

LEARNING_RATE = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

NUM_CLASSES = 20
IGNORE_INDEX = 255

SEED = 42
USE_AMP = True


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# DATASET
# ============================================================

class LIPDataset(Dataset):

    def __init__(self, image_dir, mask_dir, image_names, processor):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = image_names
        self.processor = processor

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        filename = self.image_names[idx]

        image_path = os.path.join(
            self.image_dir,
            filename
        )

        mask_path = os.path.join(
            self.mask_dir,
            os.path.splitext(filename)[0] + ".png"
        )

        # ----------------------------------------------------
        # Load RGB image
        # ----------------------------------------------------

        image = Image.open(image_path).convert("RGB")

        # ----------------------------------------------------
        # Resize image to SCHP LIP input size
        # ----------------------------------------------------

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        # ----------------------------------------------------
        # Load segmentation mask
        #
        # IMPORTANT:
        # segmentation masks contain class IDs.
        # Therefore nearest-neighbor interpolation must be used.
        # ----------------------------------------------------

        mask = Image.open(mask_path)

        mask = mask.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST
        )

        mask = np.array(mask, dtype=np.int64)

        # ----------------------------------------------------
        # Process image using SCHP processor
        # ----------------------------------------------------

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        pixel_values = inputs["pixel_values"].squeeze(0)

        mask = torch.tensor(mask, dtype=torch.long)

        return pixel_values, mask, filename


# ============================================================
# FIND VALID IMAGE/MASK PAIRS
# ============================================================

all_images = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])


valid_images = []

for filename in all_images:

    mask_name = os.path.splitext(filename)[0] + ".png"
    mask_path = os.path.join(MASK_DIR, mask_name)

    if os.path.exists(mask_path):
        valid_images.append(filename)


print("Total valid image/mask pairs:", len(valid_images))


# ============================================================
# SELECT DETERMINISTIC 500-IMAGE SUBSET
# ============================================================

random.Random(SEED).shuffle(valid_images)

selected_images = valid_images[:NUM_IMAGES]

print("Selected images:", len(selected_images))

print("\nFirst 10 selected images:")

for filename in selected_images[:10]:
    print(" ", filename)


# ============================================================
# LOAD PROCESSOR + MODEL
# ============================================================

print("\nLoading SCHP model...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = model.to(device)


# ============================================================
# DATASET + DATALOADER
# ============================================================

dataset = LIPDataset(
    IMAGE_DIR,
    MASK_DIR,
    selected_images,
    processor
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print("\nDataset size:", len(dataset))
print("Number of batches:", len(loader))


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=IGNORE_INDEX
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# MIXED PRECISION
# ============================================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(USE_AMP and device.type == "cuda")
)


# ============================================================
# BATCH NORMALIZATION FIX
# ============================================================
#
# SCHP contains BatchNorm layers.
#
# Because we are using batch size = 1, BatchNorm statistics
# cannot be reliably estimated during training.
#
# We therefore keep BatchNorm layers in evaluation mode while
# allowing the rest of the network to train normally.
# ============================================================

model.train()

for module in model.modules():

    if isinstance(
        module,
        torch.nn.modules.batchnorm._BatchNorm
    ):
        module.eval()


# ============================================================
# TRAINING LOOP
# ============================================================

print("\nStarting 500-image fine-tuning...")
print("------------------------------------------")

running_loss = 0.0

for step, (pixel_values, masks, filenames) in enumerate(loader):

    pixel_values = pixel_values.to(
        device,
        non_blocking=True
    )

    masks = masks.to(
        device,
        non_blocking=True
    )

    optimizer.zero_grad(set_to_none=True)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.amp.autocast(
        device_type="cuda",
        enabled=(USE_AMP and device.type == "cuda")
    ):

        outputs = model(
            pixel_values=pixel_values
        )

        # SCHP exposes refined parsing logits.
        logits = outputs.parsing_logits

        # Ensure output resolution matches mask.
        if logits.shape[-2:] != masks.shape[-2:]:

            logits = torch.nn.functional.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        loss = criterion(
            logits,
            masks
        )

    # --------------------------------------------------------
    # Backward pass
    # --------------------------------------------------------

    scaler.scale(loss).backward()

    scaler.step(optimizer)

    scaler.update()

    running_loss += loss.item()

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (step + 1) % 25 == 0 or step == 0:

        average_loss = running_loss / (step + 1)

        print(
            f"Step {step + 1:3d}/{len(loader)} | "
            f"Loss: {loss.item():.6f} | "
            f"Average Loss: {average_loss:.6f} | "
            f"Image: {filenames[0]}"
        )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

checkpoint_path = os.path.join(
    OUTPUT_DIR,
    "schp_lip_20_finetuned_500.pt"
)

checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "model_name": MODEL_NAME,
    "num_images": NUM_IMAGES,
    "image_size": IMAGE_SIZE,
    "learning_rate": LEARNING_RATE,
    "momentum": MOMENTUM,
    "weight_decay": WEIGHT_DECAY,
    "seed": SEED
}

torch.save(
    checkpoint,
    checkpoint_path
)

print("\n==========================================")
print("TRAINING COMPLETE")
print("==========================================")

print("Checkpoint saved to:")
print(checkpoint_path)

print(
    "Final average training loss:",
    running_loss / len(loader)
)

if torch.cuda.is_available():

    print(
        "Peak GPU memory:",
        round(
            torch.cuda.max_memory_allocated() / (1024 ** 3),
            2
        ),
        "GB"
    )
import torch
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "pirocheto/schp-lip-20"

ROOT = Path(__file__).resolve().parent

IMAGE_DIR = (
    ROOT
    / "datasets/LIP/TrainVal_images"
    / "TrainVal_images/train_images"
)

MASK_DIR = (
    ROOT
    / "datasets/LIP/TrainVal_parsing_annotations"
    / "TrainVal_parsing_annotations"
    / "TrainVal_parsing_annotations/train_segmentations"
)

IMAGE_SIZE = 473

# We are deliberately testing only 8 images.
NUM_TEST_IMAGES = 8

# Batch size 1 keeps VRAM usage conservative.
BATCH_SIZE = 1


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LIP DATASET
# ============================================================

class LIPDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        processor,
        max_images=None,
    ):

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.processor = processor

        self.image_files = sorted(
            self.image_dir.glob("*.jpg")
        )

        if max_images is not None:
            self.image_files = self.image_files[:max_images]

        print(
            "Images found:",
            len(self.image_files)
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):

        image_path = self.image_files[index]

        mask_path = (
            self.mask_dir
            / f"{image_path.stem}.png"
        )

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask not found: {mask_path}"
            )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # Load segmentation mask
        # ----------------------------------------------------

        mask = Image.open(
            mask_path
        )

        # Segmentation masks contain integer class IDs.
        # Therefore nearest-neighbour interpolation must be used.

        mask = mask.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST,
        )

        mask = torch.from_numpy(
            np.array(
                mask,
                dtype=np.int64,
            )
        )

        # ----------------------------------------------------
        # Process image
        # ----------------------------------------------------

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        # Remove processor's batch dimension.
        pixel_values = inputs[
            "pixel_values"
        ].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "mask": mask,
            "image_name": image_path.name,
        }


# ============================================================
# LOAD PROCESSOR
# ============================================================

print("\nLoading processor...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)


# ============================================================
# CREATE DATASET
# ============================================================

print("\nCreating LIP dataset...")

dataset = LIPDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    processor=processor,
    max_images=NUM_TEST_IMAGES,
)


# ============================================================
# CREATE DATALOADER
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print(
    "Batches:",
    len(loader),
)


# ============================================================
# LOAD SCHP MODEL
# ============================================================

print("\nLoading SCHP model...")

model = AutoModelForSemanticSegmentation.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
).to(device)


# ============================================================
# PUT MODEL IN TRAINING MODE
# ============================================================

model.train()


# ============================================================
# FREEZE BATCHNORM STATISTICS
# ============================================================

# The model has BatchNorm layers.
#
# With batch size = 1, some deep layers produce:
#
#     [1, 512, 1, 1]
#
# BatchNorm cannot calculate training statistics
# from only one value per channel.
#
# Therefore, keep BatchNorm layers in evaluation mode
# while allowing the rest of the network to train.

for module in model.modules():

    if isinstance(
        module,
        torch.nn.modules.batchnorm._BatchNorm,
    ):
        module.eval()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-4,
    momentum=0.9,
    weight_decay=5e-4,
)


# ============================================================
# TRAINING LOOP
# ============================================================

print("\nStarting multi-image training test...")
print("--------------------------------------")

# IMPORTANT:
#
# Do NOT call model.train() again here.
#
# Doing so would put the BatchNorm layers back into
# training mode and recreate the [1, 512, 1, 1] error.

for batch_index, batch in enumerate(
    loader,
    start=1,
):

    pixel_values = batch[
        "pixel_values"
    ].to(device)

    masks = batch[
        "mask"
    ].to(device)

    image_name = batch[
        "image_name"
    ][0]

    # --------------------------------------------------------
    # Clear previous gradients
    # --------------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.amp.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):

        outputs = model(
            pixel_values=pixel_values
        )

        parsing_logits = (
            outputs.parsing_logits
        )

        # ----------------------------------------------------
        # Segmentation loss
        # ----------------------------------------------------

        loss = F.cross_entropy(
            parsing_logits,
            masks,
            ignore_index=255,
        )

    # --------------------------------------------------------
    # Backward pass
    # --------------------------------------------------------

    loss.backward()

    # --------------------------------------------------------
    # Update model parameters
    # --------------------------------------------------------

    optimizer.step()

    print(
        f"Step {batch_index}/{len(loader)} "
        f"| Image: {image_name} "
        f"| Loss: {loss.item():.6f}"
    )


# ============================================================
# FINISHED
# ============================================================

print()
print("MULTI-IMAGE TRAINING TEST")
print("=========================")

print(
    "Successfully processed:",
    len(dataset),
    "images",
)

print(
    "Successfully completed:",
    len(loader),
    "optimizer updates",
)

print(
    "SUCCESS: LIP Dataset + DataLoader + "
    "SCHP training pipeline works."
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
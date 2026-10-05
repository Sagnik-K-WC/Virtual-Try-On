from pathlib import Path

from perception.segmentation.paths import (
    LIP_TRAIN_IMAGES,
    LIP_VAL_IMAGES,
    LIP_TRAIN_MASKS,
    LIP_VAL_MASKS,
)


# LIP directories
train_images = LIP_TRAIN_IMAGES
val_images = LIP_VAL_IMAGES

train_masks = LIP_TRAIN_MASKS
val_masks = LIP_VAL_MASKS


def get_image_stems(folder):
    """Return filename IDs without extensions."""
    extensions = {".jpg", ".jpeg", ".png"}

    return {
        file.stem
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in extensions
    }


def get_mask_stems(folder):
    """Return mask filename IDs without extensions."""
    return {
        file.stem
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() == ".png"
    }


def check_split(name, image_dir, mask_dir, expected_images):
    print(f"\n{'=' * 50}")
    print(f"{name.upper()} SET")
    print(f"{'=' * 50}")

    if not image_dir.exists():
        print("ERROR: Image directory not found:")
        print(image_dir)
        return

    if not mask_dir.exists():
        print("ERROR: Mask directory not found:")
        print(mask_dir)
        return

    images = get_image_stems(image_dir)
    masks = get_mask_stems(mask_dir)

    missing_masks = images - masks
    extra_masks = masks - images

    print(f"Images found:          {len(images)}")
    print(f"Masks found:           {len(masks)}")
    print(f"Expected images:       {expected_images}")
    print(f"Missing masks:         {len(missing_masks)}")
    print(f"Extra masks:           {len(extra_masks)}")

    if missing_masks:
        print("\nFirst missing masks:")
        for item in sorted(missing_masks)[:10]:
            print(" ", item)

    if extra_masks:
        print("\nFirst extra masks:")
        for item in sorted(extra_masks)[:10]:
            print(" ", item)

    if (
        len(images) == expected_images
        and len(masks) == expected_images
        and not missing_masks
        and not extra_masks
    ):
        print("\n✓ DATASET SPLIT PASSED")
    else:
        print("\n✗ DATASET SPLIT NEEDS INVESTIGATION")


print("LIP DATASET VERIFICATION")
print("=========================")

check_split(
    "Training",
    train_images,
    train_masks,
    expected_images=30462,
)

check_split(
    "Validation",
    val_images,
    val_masks,
    expected_images=10000,
)

print("\nVerification complete.")
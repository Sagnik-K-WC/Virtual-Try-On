# Human Parsing / Segmentation

## Overview

The segmentation component of the Virtual Try-On system performs **human parsing** on a person image.

Human parsing is a pixel-level semantic segmentation task in which each pixel is assigned a semantic class representing a human body part, clothing region, or background.

For this project, SCHP (Self-Correction for Human Parsing) is used with the 20-class LIP human-parsing label space.

---

## Purpose in the Virtual Try-On System

Human parsing provides a pixel-level representation of the person's body and clothing.

This information can later be combined with pose estimation and other perception outputs to create a structured representation of the person for the Virtual Try-On pipeline.

Relevant regions include:

- Upper clothes
- Dress
- Coat
- Pants
- Jumpsuits
- Arms
- Legs
- Face
- Hair
- Shoes
- Other human and clothing regions

The segmentation output will eventually be consumed by the common human-perception layer of the system.

---

## Dataset

The project uses the **LIP (Look Into Person)** human-parsing dataset.

LIP provides 20 semantic classes covering human body parts and clothing regions.

### Dataset Statistics

| Split | Images | Masks |
|---|---:|---:|
| Training | 30,462 | 30,462 |
| Validation | 10,000 | 10,000 |
| Test | 10,000 | — |

The training and validation image/mask correspondence was programmatically verified.

Verification results:

- Training images: 30,462
- Training masks: 30,462
- Missing training masks: 0
- Extra training masks: 0
- Validation images: 10,000
- Validation masks: 10,000
- Missing validation masks: 0
- Extra validation masks: 0

---

## Model

### SCHP

The project uses **SCHP (Self-Correction for Human Parsing)** as the human-parsing model.

SCHP was selected because it is specifically designed for human parsing and supports the 20-class LIP semantic label space.

The model produces:

- **20-class parsing logits**
- **2-class edge logits**
- Input resolution of **473 × 473**

### Model References

Two SCHP model forms were used during development:

1. A packaged SCHP LIP model was used for modern inference and development.
2. The original SCHP repository and its original pretrained LIP checkpoint were used to reproduce the reference evaluation procedure.

The quantitative project baseline is based on the **original SCHP evaluation pipeline and original pretrained checkpoint**.

---

## Validated Baseline

The original SCHP evaluation procedure was reproduced and executed on the complete 10,000-image LIP validation set.

### Results

| Metric | Result |
|---|---:|
| Pixel Accuracy | 88.10% |
| Mean Accuracy | 72.76% |
| **Mean IoU** | **58.62%** |

The **58.62% mIoU** result is the official human-parsing baseline currently used by the project.

Detailed results are available in [`baseline.md`](baseline.md).

---

## Additional Experiments

Several additional evaluation and training experiments were performed during development.

These included:

- Initial custom SCHP evaluation
- Investigation of differences between custom evaluation and the original SCHP evaluation procedure
- Horizontal flip test-time augmentation investigation
- Single-batch training verification
- Small dataset training verification
- 500-image fine-tuning experiment

The experiments did not provide sufficient evidence to replace the validated pretrained baseline.

The pretrained SCHP configuration is therefore retained as the current project baseline.

Detailed experiment history is available in [`experiments.md`](experiments.md).

---

## Current Status

The SCHP human-parsing component has been successfully validated.

Current achievements:

- LIP dataset verified
- SCHP environment configured
- Original SCHP checkpoint loaded successfully
- Original evaluation procedure reproduced
- All 10,000 LIP validation images evaluated
- 58.62% mIoU baseline established
- Segmentation results visually inspected

### Current Project State

**Validated:** SCHP human parsing baseline

**Next development stage:** Build a reusable segmentation inference interface and connect its output to the common human-perception layer.

---

## Repository Structure

```text
perception/segmentation/
├── datasets/
├── evaluation/
├── experiments/
└── inference/
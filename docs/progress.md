# Project Progress

This document records the development history of the Virtual Try-On
project, including major decisions, experiments, problems, solutions,
and validated milestones.

---

## Phase 1 — Project Concept

### Objective

Develop a machine-learning-based Virtual Try-On system supporting both:

- Photo-based virtual try-on
- Real-time camera-based virtual try-on

The system is designed around a shared human-perception layer followed
by different processing paths for photo generation and real-time
interaction.

### Initial Architecture

The planned system consists of:

1. Person Detection
2. Pose Estimation
3. Human Parsing / Segmentation
4. Body Representation
5. Garment Processing
6. Garment Alignment
7. Virtual Try-On Generation
8. Tracking and Real-Time Fitting
9. Rendering

---

## Phase 2 — CatVTON Proof of Concept

### Objective

Establish that a pretrained Virtual Try-On generation model can be
executed locally and produce a successful try-on result.

### Work Completed

- Set up the CatVTON environment locally.
- Resolved dependency compatibility issues.
- Successfully launched the CatVTON Gradio application.
- Supplied a person image and an upper-body garment image.
- Successfully generated a virtual try-on result.

### Outcome

The CatVTON generation component was successfully validated as a
working pretrained inference component.

This established the VTO generation stage that will later be integrated
with the project's perception components.

---

## Phase 3 — Pose Estimation

### Objective

Develop and evaluate a pose-estimation component for identifying human
body keypoints.

### Work Completed

- Set up an Ultralytics YOLO pose environment.
- Verified GPU/CUDA acceleration.
- Evaluated YOLO26n and YOLO26s pose models.
- Trained models on the COCO Pose dataset.
- Evaluated pose precision, recall, and mAP metrics.
- Investigated model continuation and fine-tuning behavior.

### Current Status

Pose estimation is being developed as part of the human-perception
component of the system.

---

## Phase 4 — Human Parsing / Segmentation

### Objective

Develop a human-parsing component that identifies semantic human
regions at pixel level.

### Dataset

The LIP (Look into Person) dataset was selected for human parsing.

It provides 20 semantic classes including:

- Background
- Hair
- Upper-clothes
- Dress
- Coat
- Pants
- Face
- Arms
- Legs
- Shoes
- Other human/clothing regions

### Model Selection

SCHP (Self-Correction for Human Parsing) was selected because it is
specifically designed for human parsing and provides a pretrained model
for the LIP label set.

### Work Completed

- Acquired and verified the LIP dataset.
- Verified training and validation image/mask correspondence.
- Set up a modern SCHP environment.
- Successfully compiled the original SCHP InPlaceABNSync extension.
- Loaded the original pretrained SCHP LIP checkpoint.
- Reproduced the original repository evaluation pipeline.
- Evaluated all 10,000 LIP validation images.

### Baseline Result

Original SCHP evaluation:

**mIoU: 58.62%**

### Flip TTA Experiment

Horizontal flip test-time augmentation was evaluated using the original
SCHP implementation, including left/right semantic-class correction.

Result:

**mIoU: 59.00%**

### Current Decision

SCHP with horizontal flip TTA is currently selected as the human-parsing
configuration for the project.

---

## Current Project Status

The project currently has independently validated components for:

- Virtual Try-On generation using CatVTON
- Human pose estimation experiments
- Human parsing using SCHP

The next major objective is to integrate these components into the
common perception and Virtual Try-On pipeline.

---

## Next Steps

1. Organize validated code and experiments into the project repository.
2. Integrate human perception outputs.
3. Develop garment preprocessing and alignment.
4. Integrate the perception layer with CatVTON.
5. Develop the real-time camera pipeline.
6. Evaluate the complete system.
# Project Progress

This document records the major development stages, experiments, decisions, and validated milestones of the Virtual Try-On project.

---

## Phase 1 — Project Concept and Architecture

### Objective

Develop a unified machine-learning-based Virtual Try-On system supporting two operating modes:

- **Photo-Based Virtual Try-On** — a user provides a person image and a garment image, and the system generates a high-quality try-on result.
- **Real-Time Virtual Try-On** — a camera-based pipeline tracks the user's body and garment placement to provide an interactive try-on experience.

### Initial Architecture

The planned system consists of the following major components:

1. Person Detection
2. Pose Estimation
3. Human Parsing / Segmentation
4. Body Representation
5. Garment Processing
6. Garment Alignment
7. Virtual Try-On Generation
8. Tracking and Real-Time Fitting
9. Rendering

The system is designed around a shared human-perception layer. Photo-based and real-time modes then use the perception outputs differently according to their requirements.

### Development Strategy

The project does not aim to train a complete Virtual Try-On generation model from scratch. Instead:

- Pose estimation and human parsing/segmentation form the main machine-learning development components.
- A pretrained Virtual Try-On model is used for high-quality image generation.
- The final system will integrate these components into a unified pipeline.

---

## Phase 2 — CatVTON Proof of Concept

### Objective

Validate that a pretrained Virtual Try-On generation model could be executed locally and produce a successful try-on result.

### Work Completed

- Set up a dedicated CatVTON environment.
- Resolved PEFT and Accelerate compatibility issues.
- Resolved Gradio, FastAPI, and Starlette compatibility issues.
- Verified the environment using `pip check`.
- Successfully loaded the pretrained CatVTON model.
- Launched the CatVTON Gradio application locally.
- Provided a person image and an upper-body garment image.
- Successfully generated a virtual try-on result.

### Outcome

CatVTON inference was successfully demonstrated locally.

This established the Virtual Try-On generation stage as a working pretrained component that can later be integrated with the project's perception components.

### Status

**✓ Completed — CatVTON successfully tested**

---

## Phase 3 — Pose Estimation

### Objective

Develop and evaluate a pose-estimation component capable of identifying human body keypoints for the human-perception layer.

### Dataset and Environment

Pose experiments were conducted using the COCO Pose dataset with the Ultralytics YOLO pose framework.

The training environment was configured for GPU acceleration using the NVIDIA GeForce RTX 5060 Laptop GPU.

### Initial Sanity Experiment — YOLO26n

A one-epoch YOLO26n training run was performed as a sanity check to verify that the complete training pipeline was functioning correctly.

Approximate validation results:

| Metric | Result |
|---|---:|
| Precision | 0.662 |
| Recall | 0.497 |
| mAP50 | 0.504 |
| mAP50-95 | 0.197 |

This experiment was used primarily to validate the training pipeline and was not treated as the final pose result.

### YOLO26n — 5 Epochs

A longer five-epoch YOLO26n experiment was subsequently performed.

Approximate validation results:

| Metric | Result |
|---|---:|
| Precision | 0.807 |
| Recall | 0.695 |
| mAP50 | 0.745 |
| mAP50-95 | 0.459 |

The experiment demonstrated substantial improvement compared with the one-epoch sanity run.

### YOLO26s — 5 Epochs

A larger YOLO26s pose model was also trained for five epochs.

Validation results:

| Metric | Result |
|---|---:|
| Precision | 0.812 |
| Recall | 0.675 |
| mAP50 | 0.729 |
| mAP50-95 | 0.431 |

The larger model did not automatically produce higher validation mAP than the YOLO26n experiment at the same five-epoch training duration.

### Continuation and Fine-Tuning Investigation

Attempts were made to continue training from the five-epoch YOLO26s checkpoint.

The original Ultralytics resume mechanism reported that the configured five-epoch run had already completed and therefore could not simply be resumed by changing the epoch count.

A separate continuation-style experiment and a controlled low-learning-rate fine-tuning experiment were subsequently investigated. These experiments resulted in substantial degradation and were not adopted as project models.

### Current Decision

The five-epoch YOLO pose experiments are retained as experimental results. Further pose training and selection will be performed later if required by the integrated system.

### Status

**✓ Pose model trained and evaluated**

**Note:** Pose estimation is considered an experimental/validated component at this stage rather than a final production model.

---

## Phase 4 — LIP Dataset Preparation

### Objective

Prepare a suitable human-parsing dataset for the segmentation component.

### Dataset Selection

The **LIP (Look Into Person)** dataset was selected for human parsing.

The dataset provides 20 semantic classes representing human parts and clothing regions.

### Dataset Statistics

| Split | Images | Masks |
|---|---:|---:|
| Training | 30,462 | 30,462 |
| Validation | 10,000 | 10,000 |
| Test | 10,000 | — |

### Dataset Verification

Image and segmentation-mask correspondence was programmatically verified.

#### Training Set

- Images: 30,462
- Masks: 30,462
- Missing masks: 0
- Extra masks: 0

#### Validation Set

- Images: 10,000
- Masks: 10,000
- Missing masks: 0
- Extra masks: 0

### Outcome

The downloaded LIP training and validation data were successfully verified and found to have complete image/mask correspondence.

### Status

**✓ Completed — LIP dataset verified**

---

## Phase 5 — SCHP Human Parsing / Segmentation

### Objective

Develop a pixel-level human-parsing component that identifies semantic human and clothing regions in a person image.

Unlike pose estimation, which identifies body keypoints, human parsing provides a semantic label for each pixel.

This allows the Virtual Try-On system to distinguish regions such as upper clothes, dresses, coats, pants, arms, legs, face, hair, and shoes.

### Model Selection

**SCHP (Self-Correction for Human Parsing)** was selected because it is specifically designed for human parsing and provides a pretrained model for the LIP semantic label space.

The model operates with the 20-class LIP representation.

### Environment and Compatibility Work

The original SCHP repository uses an older PyTorch/CUDA toolchain and a custom InPlaceABNSync extension.

A separate environment was therefore configured to reproduce the original evaluation pipeline on the project's NVIDIA RTX 5060 Laptop GPU.

Compatibility work included:

- configuring a compatible PyTorch/CUDA environment,
- installing the required CUDA development tools,
- configuring the Visual Studio C++ build environment,
- compiling the custom InPlaceABNSync extension,
- applying required compatibility adjustments to the original extension source.

The original pretrained SCHP LIP checkpoint was subsequently loaded successfully.

### Official Baseline Evaluation

The original SCHP evaluation procedure was reproduced and executed on the complete 10,000-image LIP validation set.

Validated results:

| Metric | Result |
|---|---:|
| Pixel Accuracy | 88.10% |
| Mean Accuracy | 72.76% |
| Mean IoU | **58.62%** |

The **58.62% mIoU** result is currently used as the project's official SCHP human-parsing baseline.

### Additional Evaluation Investigation

Initial custom evaluation approaches produced lower results than expected. The evaluation procedure was therefore investigated rather than assuming that the pretrained model was performing poorly.

This led to reproduction of the original SCHP evaluation pipeline and the validated 58.62% mIoU baseline.

Horizontal flip test-time augmentation was also investigated as an additional experiment. Because the project requires a clearly defined and reproducible baseline, the original SCHP evaluation result of 58.62% mIoU is retained as the official baseline.

### Fine-Tuning Investigation

The SCHP training pipeline was tested progressively:

1. Single-batch training test
2. Small eight-image training test
3. 500-image fine-tuning experiment

These tests verified that the training pipeline could perform forward propagation, loss calculation, backpropagation, and parameter updates.

The 500-image experiment did not provide evidence that fine-tuning improved the pretrained baseline. Its evaluation procedure also differed from the definitive original SCHP evaluation protocol, so its result is not treated as a directly comparable official performance figure.

### Current Decision

The validated pretrained SCHP model is retained as the current human-parsing baseline.

Further fine-tuning is not currently required.

### Status

**✓ Completed — SCHP human parsing validated**

**✓ Segmentation baseline documented**

---

## Phase 6 — Project Repository Organization

### Objective

Organize the project into a shared GitHub repository containing source code, documentation, experiments, and reproducible project structure.

### Repository Structure

The repository separates major system components into:

- `perception/` — human perception components
- `garment/` — garment processing
- `vton/` — Virtual Try-On generation
- `realtime/` — real-time camera pipeline
- `integration/` — system integration
- `evaluation/` — evaluation utilities
- `docs/` — project documentation
- `requirements/` — environment requirements

### Segmentation Organization

The segmentation component is organized into:

```text
perception/segmentation/
├── datasets/
├── evaluation/
├── experiments/
└── inference/
---

## Current Project Status

```text
✓ Project architecture defined
✓ CatVTON successfully tested
✓ Pose model trained/evaluated
✓ LIP dataset verified
✓ SCHP human parsing validated
✓ Segmentation baseline documented

⏳ Segmentation integration
⏳ Garment processing
⏳ Real-time pipeline
⏳ Full system integration
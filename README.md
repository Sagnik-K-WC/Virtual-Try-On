# Virtual Try-On System

A machine-learning-based Virtual Try-On system combining human perception, garment processing, Virtual Try-On generation, and real-time camera interaction.

The project is being developed as a unified system with two operating modes:

- **Photo-Based Virtual Try-On** — generates a high-quality try-on image from a person image and a garment image.
- **Real-Time Virtual Try-On** — uses a camera-based perception and tracking pipeline to provide interactive clothing visualization.

---

## Project Objective

The objective of this project is to develop a complete Virtual Try-On pipeline that can understand a person's body and clothing regions, process a target garment, and generate or render the garment on the person.

The system is designed around a shared human-perception layer consisting primarily of:

- Person detection
- Pose estimation
- Human parsing / semantic segmentation
- Body representation

These outputs will subsequently be combined with garment processing and Virtual Try-On generation.

The project focuses on developing and integrating the machine-learning perception components while using a pretrained Virtual Try-On generation model for high-quality image synthesis.

---

## System Architecture

The planned system follows the architecture below:

```text
                         USER
                           │
              ┌────────────┴────────────┐
              │                         │
        Photo Upload               Live Camera
              │                         │
              └────────────┬────────────┘
                           │
                    Person Detection
                           │
                     Pose Estimation
                           │
               Human Parsing / Segmentation
                           │
                   Body Representation
                      /            \
                     /              \
              PHOTO MODE         LIVE MODE
                  │                  │
          Garment Processing      Tracking
                  │                  │
          Garment Alignment     Garment Fitting
                  │                  │
              CatVTON             Occlusion
                  │                  │
       High-Quality Image        Rendering
                                     │
                                  Real-Time
                                     AR
```

## Project Modes

### 1. Photo-Based Virtual Try-On

The photo-based pipeline is intended to produce a high-quality generated image.

```text
Person Image
     │
     ├── Person Detection
     ├── Pose Estimation
     └── Human Parsing
              │
              ↓
      Body Representation
              │
Garment Image → Garment Processing
              │
              ↓
       Garment Alignment
              │
              ↓
           CatVTON
              │
              ↓
      High-Quality VTO Image
```

CatVTON has already been successfully tested locally as the pretrained Virtual Try-On generation component.

The complete integration with the perception pipeline is still under development.

---

### 2. Real-Time Virtual Try-On

The real-time pipeline is intended to operate from a live camera feed.

```text
Camera
  │
  ↓
Person Detection
  │
  ↓
Pose / Segmentation
  │
  ↓
Tracking
  │
  ↓
Garment Fitting
  │
  ↓
Occlusion Handling
  │
  ↓
Real-Time Rendering
```

The real-time pipeline will use lightweight perception, tracking, fitting, and rendering rather than attempting to run a computationally expensive diffusion-based VTO generator independently on every camera frame.

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
```

The project is currently transitioning from independently validated components toward system integration.

---

# Completed Work

## 1. Virtual Try-On Generation — CatVTON

CatVTON was set up and executed locally as a proof of concept.

Completed work includes:

- CatVTON environment setup
- PEFT / Accelerate compatibility resolution
- Gradio / FastAPI / Starlette compatibility resolution
- Dependency verification using `pip check`
- Local Gradio application launch
- Person image and garment image input
- Successful upper-body Virtual Try-On generation

**Current status:** CatVTON successfully tested as a pretrained VTO generation component.

Detailed development history is available in [`docs/progress.md`](docs/progress.md).

---

## 2. Pose Estimation

Human pose estimation is being developed as part of the human-perception layer.

Experiments were performed using:

- Ultralytics YOLO pose models
- COCO Pose dataset
- NVIDIA RTX 5060 Laptop GPU

### YOLO26n — 5 Epochs

| Metric | Result |
|---|---:|
| Precision | 0.807 |
| Recall | 0.695 |
| mAP50 | 0.745 |
| mAP50-95 | 0.459 |

### YOLO26s — 5 Epochs

| Metric | Result |
|---|---:|
| Precision | 0.812 |
| Recall | 0.675 |
| mAP50 | 0.729 |
| mAP50-95 | 0.431 |

Additional continuation and fine-tuning experiments were investigated but were not adopted because they resulted in substantial performance degradation.

The pose component remains an experimental/validated component and can be further developed during system integration.

---

## 3. Human Parsing / Segmentation

Human parsing provides a pixel-level semantic representation of a person's body and clothing.

It allows the system to distinguish regions such as:

- Upper clothes
- Dress
- Coat
- Pants
- Arms
- Legs
- Face
- Hair
- Shoes

### Dataset

The project uses the **LIP (Look Into Person)** human-parsing dataset.

| Split | Images | Masks |
|---|---:|---:|
| Training | 30,462 | 30,462 |
| Validation | 10,000 | 10,000 |
| Test | 10,000 | — |

Training and validation image/mask correspondence was programmatically verified with no missing or extra masks.

### Model

The project uses **SCHP (Self-Correction for Human Parsing)** for the human-parsing component.

The model was selected because it is specifically designed for human parsing and supports the 20-class LIP semantic label space.

### Validated Baseline

The original SCHP evaluation pipeline was reproduced on the complete 10,000-image LIP validation set.

| Metric | Result |
|---|---:|
| Pixel Accuracy | 88.10% |
| Mean Accuracy | 72.76% |
| **Mean IoU** | **58.62%** |

**58.62% mIoU is the current official SCHP baseline for this project.**

Additional evaluation and fine-tuning experiments were also investigated. The validated pretrained baseline is currently retained rather than using the experimental fine-tuned models.

Detailed documentation:

- [`Segmentation Overview`](docs/perception/segmentation/README.md)
- [`Validated Baseline`](docs/perception/segmentation/baseline.md)
- [`Segmentation Experiments`](docs/perception/segmentation/experiments.md)

---

# Technology Stack

| Area | Technology |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| VTO Generation | CatVTON |
| Pose Estimation | Ultralytics YOLO Pose |
| Human Parsing | SCHP |
| Human Parsing Dataset | LIP |
| Pose Dataset | COCO Pose |
| VTO Interface | Gradio |
| GPU Acceleration | NVIDIA CUDA |
| Version Control | Git / GitHub |

---

# Repository Structure

```text
Virtual-Try-On/
│
├── README.md
│
├── docs/
│   ├── architecture/
│   ├── datasets/
│   ├── experiments/
│   ├── perception/
│   │   └── segmentation/
│   ├── setup/
│   └── progress.md
│
├── perception/
│   ├── pose/
│   └── segmentation/
│       ├── datasets/
│       ├── evaluation/
│       ├── experiments/
│       └── inference/
│
├── garment/
├── vton/
├── realtime/
├── integration/
├── evaluation/
│
└── requirements/
```

### Main directory responsibilities

| Directory | Purpose |
|---|---|
| `perception/` | Human perception components |
| `garment/` | Garment processing and representation |
| `vton/` | Virtual Try-On generation |
| `realtime/` | Live camera pipeline |
| `integration/` | Integration between system components |
| `evaluation/` | Evaluation utilities and metrics |
| `docs/` | Technical documentation and project history |
| `requirements/` | Environment and dependency specifications |

---

# Team Responsibilities

The project is divided into three major technical areas.

### Person 1 — Human Perception ML

Responsible for:

- Pose estimation
- Human parsing / segmentation
- Person representation
- Perception model evaluation
- Integration of perception outputs

### Person 2 — Garment and VTO Pipeline

Responsible for:

- Garment preprocessing
- Garment representation
- Garment alignment
- CatVTON integration
- Photo-based Virtual Try-On pipeline

### Person 3 — Real-Time Application

Responsible for:

- Camera input
- Tracking
- Real-time garment fitting
- Occlusion handling
- Rendering
- User-facing application pipeline

The final system will combine these components through shared interfaces.

---

# Development Roadmap

## Completed

- [x] Define overall system architecture
- [x] Establish photo and real-time VTO modes
- [x] Validate CatVTON locally
- [x] Set up and evaluate pose estimation experiments
- [x] Acquire LIP human-parsing dataset
- [x] Verify LIP image/mask correspondence
- [x] Set up SCHP human parsing
- [x] Reproduce original SCHP evaluation
- [x] Establish 58.62% mIoU segmentation baseline
- [x] Organize initial project repository and documentation

## In Progress / Planned

- [ ] Build reusable segmentation inference interface
- [ ] Integrate segmentation into the common perception layer
- [ ] Develop garment preprocessing
- [ ] Develop garment alignment
- [ ] Integrate perception outputs with CatVTON
- [ ] Develop the photo-based end-to-end pipeline
- [ ] Develop real-time camera processing
- [ ] Implement tracking and garment fitting
- [ ] Implement occlusion handling and rendering
- [ ] Evaluate the complete integrated system

---

# Project Documentation

### General Project

- [`Project Progress`](docs/progress.md) — chronological development history, experiments, decisions, and milestones.

### Human Parsing / Segmentation

- [`Segmentation Overview`](docs/perception/segmentation/README.md)
- [`Validated SCHP Baseline`](docs/perception/segmentation/baseline.md)
- [`Segmentation Experiments`](docs/perception/segmentation/experiments.md)

More documentation will be added as the remaining components are implemented.

---

# Current Direction

The project is currently moving from independently validated machine-learning components toward integration.

The development sequence is:

```text
Project Architecture
        ↓
CatVTON Proof of Concept
        ↓
Pose Estimation Development
        ↓
LIP Dataset Preparation
        ↓
SCHP Human Parsing
        ↓
Validated Perception Components
        ↓
Component Integration
        ↓
Photo-Based Virtual Try-On
        ↓
Real-Time Virtual Try-On
        ↓
Complete System Evaluation
```

The immediate technical priority is to create reusable interfaces between the validated perception components and the rest of the Virtual Try-On pipeline.
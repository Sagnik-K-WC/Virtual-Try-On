# Human Pose Estimation

## Overview

The pose-estimation component is part of the human-perception layer of the Virtual Try-On system.

Pose estimation identifies anatomical keypoints of a person and provides information about the person's body configuration and posture.

This information will later be combined with human parsing/segmentation and other perception outputs to create a structured representation of the person for the Virtual Try-On pipeline.

---

## Purpose in the Virtual Try-On System

Pose estimation provides the spatial configuration of the person's body.

The detected keypoints can be used to estimate the position and orientation of important body regions such as:

- Head
- Shoulders
- Elbows
- Wrists
- Hips
- Knees
- Ankles

This information is particularly useful for:

- understanding body posture,
- determining garment placement,
- supporting garment alignment,
- tracking body movement,
- and assisting the real-time Virtual Try-On pipeline.

Pose information will eventually be combined with human parsing/segmentation to provide complementary information about the person.

---

## Pose Representation

The current experiments use the **COCO 17-keypoint pose representation**.

| ID | Keypoint |
|---:|---|
| 0 | Nose |
| 1 | Left Eye |
| 2 | Right Eye |
| 3 | Left Ear |
| 4 | Right Ear |
| 5 | Left Shoulder |
| 6 | Right Shoulder |
| 7 | Left Elbow |
| 8 | Right Elbow |
| 9 | Left Wrist |
| 10 | Right Wrist |
| 11 | Left Hip |
| 12 | Right Hip |
| 13 | Left Knee |
| 14 | Right Knee |
| 15 | Left Ankle |
| 16 | Right Ankle |

Each keypoint represents a body location that can be used to describe the person's pose.

---

## Dataset

The pose experiments use the **COCO Pose** dataset through the Ultralytics pose-training framework.

The training experiments used:

- COCO Pose annotations
- 56,599 training images
- 2,346 validation images
- 17 keypoints
- 640 × 640 training resolution

---

## Model Framework

Pose estimation experiments were conducted using the **Ultralytics YOLO pose framework**.

The project investigated:

- YOLO26n Pose
- YOLO26s Pose

GPU-accelerated training was performed using the NVIDIA GeForce RTX 5060 Laptop GPU.

The training environment used PyTorch with CUDA acceleration and Ultralytics.

---

## Experimental Status

Pose estimation is currently considered an **experimental development component**.

Several models and training configurations have been evaluated, but no final production pose model has been selected.

The current experimental reference is the five-epoch YOLO26s experiment.

This is **not considered a final model**. Further development may involve additional training, model selection, hyperparameter experiments, or evaluation based on the requirements of the integrated Virtual Try-On system.

---

## Current Experimental Result

The five-epoch YOLO26s experiment produced the following validation results:

| Metric | Result |
|---|---:|
| Precision | 0.812 |
| Recall | 0.675 |
| mAP50 | 0.729 |
| mAP50-95 | 0.431 |

These results are retained as an experimental reference rather than a final project baseline.

Detailed experiments and model-development history are documented in [`experiments.md`](experiments.md).

---

## Integration Role

The eventual pose-estimation interface should provide structured keypoint information to the common human-perception layer.

A conceptual output may contain:

```python
{
    "keypoints": ...,
    "confidence": ...,
    "person_bbox": ...
}
```

The eventual pose-estimation interface will provide structured keypoint information to the common human-perception layer.

---

## Current Status

**Experimental development**

### Completed

- YOLO pose environment configured
- GPU/CUDA acceleration verified
- YOLO26n sanity experiment completed
- YOLO26n five-epoch experiment completed
- YOLO26s five-epoch experiment completed
- Model continuation/fine-tuning behavior investigated
- Pose metrics evaluated

### Still Open

- Final pose model selection
- Further training and optimization
- Integration with segmentation
- Integration with the Virtual Try-On pipeline
- Evaluation under project-specific VTO requirements

---

## Repository Structure

```text
perception/pose/
├── README.md
├── baseline.md
└── experiments.md
```

The pose source code and experiments will be organized here as development progresses.

---

## Related Documentation

- [`Pose Experiments`](experiments.md) — detailed model experiments and training investigations
- [`Pose Baseline`](baseline.md) — current experimental reference configuration
- [`Segmentation Documentation`](../segmentation/README.md) — complementary human-parsing component
- [`Project Progress`](../../progress.md) — overall Virtual Try-On development history
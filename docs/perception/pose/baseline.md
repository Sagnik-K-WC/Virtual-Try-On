# Current Experimental Pose Reference

## Purpose

This document records the current experimental reference configuration for the human pose-estimation component.

The pose-estimation work is still under development. Therefore, this document does **not** define a final production pose model.

Detailed experiments and training investigations are documented in [`experiments.md`](experiments.md).

---

## Current Experimental Reference

The current experimental reference is the five-epoch **YOLO26s Pose** experiment trained on the COCO Pose dataset.

### Configuration

| Parameter | Value |
|---|---|
| Model | YOLO26s Pose |
| Dataset | COCO Pose |
| Epochs | 5 |
| Batch size | 2 |
| Image size | 640 × 640 |
| Pretrained | Yes |
| AMP | Enabled |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |

---

## Validation Results

| Metric | Result |
|---|---:|
| Precision | 0.812 |
| Recall | 0.675 |
| mAP50 | 0.729 |
| mAP50-95 | 0.431 |

Training took approximately 4.9 hours under the recorded experimental configuration.

---

## Comparison with YOLO26n

The five-epoch YOLO26n experiment produced:

| Metric | YOLO26n | YOLO26s |
|---|---:|---:|
| Precision | 0.807 | **0.812** |
| Recall | **0.695** | 0.675 |
| mAP50 | **0.745** | 0.729 |
| mAP50-95 | **0.459** | 0.431 |

The YOLO26n experiment therefore achieved higher recall, mAP50, and mAP50-95, while YOLO26s achieved slightly higher precision.

This demonstrates that the larger YOLO26s model did not automatically provide better validation performance under the current five-epoch training conditions.

---

## Why YOLO26s Is Retained as the Current Reference

YOLO26s is retained as the current experimental reference because it represents the larger model configuration that was explicitly investigated as a potential improvement over YOLO26n.

However, this should not be interpreted as a conclusion that YOLO26s is definitively superior.

The model remains a reference point for further development and integration testing.

---

## Continuation and Fine-Tuning Results

Additional experiments attempted to continue training or fine-tune the YOLO26s checkpoint.

These experiments resulted in substantial degradation of validation performance and were not adopted.

The original five-epoch YOLO26s checkpoint therefore remains the current experimental reference.

The detailed continuation and fine-tuning investigations are documented in [`experiments.md`](experiments.md).

---

## Current Status

**Experimental — not final**

The pose-estimation component is functional and has been evaluated using multiple YOLO pose configurations.

However, final model selection remains open.

Future development may include:

- Additional training
- Hyperparameter tuning
- Alternative YOLO model sizes
- Alternative pose-estimation architectures
- Evaluation on project-specific Virtual Try-On requirements
- Integration testing with human parsing and garment processing

---

## Integration Considerations

The final pose model should not be selected solely on standalone validation metrics.

Its usefulness within the complete Virtual Try-On system must also be considered.

The eventual perception layer is expected to combine pose information with human parsing:

```text
Pose Estimation
      │
      │ body keypoints
      ↓
Human Perception Representation
      ↑
      │ semantic regions
      │
Human Parsing / Segmentation
```

The pose component will therefore be evaluated again during system integration.

---

## Reference Result

For the current stage of development:

```text
Model: YOLO26s Pose
Training: 5 epochs
Precision: 0.812
Recall: 0.675
mAP50: 0.729
mAP50-95: 0.431
Status: Experimental
```

This result is a development reference and **not a final project model**.
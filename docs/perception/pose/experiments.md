# Pose Estimation Experiments

## Purpose

This document records the pose-estimation experiments performed during development of the human-perception component.

The purpose of these experiments was to verify the pose-training pipeline, compare model sizes, evaluate training behavior, and investigate whether additional training could improve the observed results.

No final production pose model has been selected at this stage.

---

## 1. YOLO26n — One-Epoch Sanity Experiment

### Objective

Before committing to longer training runs, a one-epoch experiment was performed to verify that the complete YOLO pose-training pipeline was functioning correctly.

### Configuration

```text
Model: YOLO26n Pose
Dataset: COCO Pose
Epochs: 1
Batch size: 2
Image size: 640
Device: NVIDIA RTX 5060 Laptop GPU
Pretrained: Yes
AMP: Enabled
```

### Validation Results

| Metric | Result |
|---|---:|
| Precision | 0.662 |
| Recall | 0.497 |
| mAP50 | 0.504 |
| mAP50-95 | 0.197 |

### Interpretation

The experiment successfully demonstrated that the pose-training pipeline was working.

The relatively limited performance was expected from a one-epoch sanity run and was not treated as a final model result.

---

## 2. YOLO26n — Five-Epoch Experiment

### Objective

A longer training run was performed to observe how the YOLO26n pose model developed with additional training.

### Configuration

```text
Model: YOLO26n Pose
Dataset: COCO Pose
Epochs: 5
Batch size: 2
Image size: 640
Device: NVIDIA RTX 5060 Laptop GPU
Pretrained: Yes
AMP: Enabled
```

### Validation Results

| Metric | Result |
|---|---:|
| Precision | 0.807 |
| Recall | 0.695 |
| mAP50 | 0.745 |
| mAP50-95 | 0.459 |

Training took approximately 6.4 hours.

### Interpretation

The five-epoch experiment produced a substantial improvement compared with the one-epoch sanity run.

This confirmed that the model was learning effectively from the COCO Pose training data.

---

## 3. YOLO26s — Five-Epoch Experiment

### Objective

A larger YOLO26s pose model was evaluated to investigate whether the increased model capacity would provide better validation performance.

### Configuration

```text
Model: YOLO26s Pose
Dataset: COCO Pose
Epochs: 5
Batch size: 2
Image size: 640
Device: NVIDIA RTX 5060 Laptop GPU
Pretrained: Yes
AMP: Enabled
```

### Model Size

The YOLO26s experiment used approximately:

```text
11.8 million parameters
29.5 GFLOPs
```

### Validation Results

| Metric | Result |
|---|---:|
| Precision | 0.812 |
| Recall | 0.675 |
| mAP50 | 0.729 |
| mAP50-95 | 0.431 |

Training took approximately 4.9 hours.

### Comparison with YOLO26n

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLO26n — 5 epochs | 0.807 | 0.695 | 0.745 | **0.459** |
| YOLO26s — 5 epochs | **0.812** | 0.675 | 0.729 | 0.431 |

### Interpretation

The larger YOLO26s model did not automatically produce higher validation mAP than YOLO26n at the same five-epoch training duration.

YOLO26s achieved slightly higher precision, while YOLO26n achieved higher recall, mAP50, and mAP50-95.

This indicates that model size alone was not sufficient to determine the better configuration under the current experimental conditions.

---

## 4. Training Continuation Investigation

### Objective

After completing the five-epoch YOLO26s experiment, an attempt was made to continue training the existing checkpoint for additional epochs.

### Observation

The Ultralytics resume mechanism reported that the configured five-epoch training run had already finished.

Changing the epoch count during resume was therefore not accepted as a way to extend the completed run.

### Outcome

The original five-epoch run was retained.

This highlighted the distinction between:

- resuming an interrupted training run, and
- starting a new training run from an existing checkpoint.

---

## 5. Continuation-Style Experiment

A separate experiment was attempted using the five-epoch YOLO26s checkpoint as the starting model for another five-epoch training run.

The experiment was configured with:

```text
Starting checkpoint: YOLO26s five-epoch checkpoint
Epochs: 5
Batch size: 2
Image size: 640
Workers: 2
AMP: Enabled
```

The experiment did not preserve the performance of the original five-epoch checkpoint.

The resulting model showed substantially lower validation metrics.

### Outcome

This experiment was not adopted.

The original validated five-epoch YOLO26s result remains the more useful experimental reference.

---

## 6. Controlled Fine-Tuning Investigation

A further controlled test was performed using the YOLO26s checkpoint with a lower learning rate and no warmup.

### Configuration

```text
Starting checkpoint: YOLO26s five-epoch best checkpoint
Epochs: 1
Batch size: 2
Image size: 640
Workers: 2
Optimizer: AdamW
Learning rate: 0.0001
Warmup epochs: 0
AMP: Enabled
```

### Result

The one-epoch fine-tuning test produced substantially degraded validation performance:

| Metric | Result |
|---|---:|
| Precision | 0.127 |
| Recall | 0.074 |
| mAP50 | 0.021 |
| mAP50-95 | 0.003 |

### Interpretation

The controlled test did not preserve the performance of the original checkpoint.

The fine-tuned result was therefore not adopted.

The exact cause of the degradation was not conclusively established during the experiment, so no single cause is claimed here.

---

## 7. Experimental Conclusions

The pose experiments established several observations.

### Finding 1 — The training pipeline works

The one-epoch sanity run successfully verified the complete training pipeline.

### Finding 2 — Additional training substantially improves the initial model

YOLO26n improved considerably between one and five epochs.

### Finding 3 — A larger model does not automatically produce better validation metrics

At five epochs, YOLO26n achieved higher mAP50 and mAP50-95 than YOLO26s, while YOLO26s achieved slightly higher precision.

### Finding 4 — Additional training requires controlled experimentation

The continuation-style and controlled fine-tuning experiments did not preserve the performance of the original checkpoints.

These results indicate that simply starting another training run from an existing checkpoint does not guarantee improvement.

### Finding 5 — No final pose model has been selected

The current experiments are sufficient to establish that the pose component is functional, but they are not sufficient to declare a final production configuration.

Further development may involve:

- additional training,
- hyperparameter tuning,
- different model sizes,
- alternative pose architectures,
- or evaluation based on the requirements of the integrated Virtual Try-On pipeline.

---

## 8. Current Experimental Reference

For documentation purposes, the five-epoch YOLO26s experiment is currently retained as the **experimental reference configuration**.

```text
Model: YOLO26s Pose
Training: 5 epochs
Precision: 0.812
Recall: 0.675
mAP50: 0.729
mAP50-95: 0.431
```

This is **not a final project model**.

The pose-estimation component remains open for further development and model selection.

---

## 9. Next Pose Development

The next stage is to determine how pose estimation should be integrated with the human-parsing component.

The eventual perception layer should combine:

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

The final pose model should ultimately be selected not only according to standalone validation metrics, but also according to its usefulness within the complete Virtual Try-On system.
# SCHP Human Parsing Baseline

## 1. Objective

Evaluate the pretrained SCHP human parsing model on the LIP validation dataset and establish a reproducible segmentation baseline for the Virtual Try-On project.

The human parsing component provides pixel-level information about body parts and clothing regions. This information will later be used as part of the common human-perception layer of the Virtual Try-On system.

---

## 2. Model

**Model:** Self-Correction for Human Parsing (SCHP)

**Checkpoint:**

`exp-schp-201908261155-lip.pth`

**Dataset:** LIP (Look into Person)

**Number of classes:** 20

The 20 classes are:

1. Background
2. Hat
3. Hair
4. Glove
5. Sunglasses
6. Upper-clothes
7. Dress
8. Coat
9. Socks
10. Pants
11. Jumpsuits
12. Scarf
13. Skirt
14. Face
15. Left-arm
16. Right-arm
17. Left-leg
18. Right-leg
19. Left-shoe
20. Right-shoe

---

## 3. Dataset

The LIP validation set contains:

- 10,000 validation images
- 10,000 corresponding segmentation annotations
- 20 semantic classes

The dataset was verified before evaluation.

### Dataset verification

| Split | Images | Masks | Missing Masks | Extra Masks |
|---|---:|---:|---:|---:|
| Training | 30,462 | 30,462 | 0 | 0 |
| Validation | 10,000 | 10,000 | 0 | 0 |

Both dataset splits passed the verification checks.

---

## 4. Evaluation Setup

The original SCHP repository evaluation pipeline was used rather than a custom metric implementation.

The evaluation was performed using:

- Input size: 473 × 473
- Batch size: 1
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- Validation samples: 10,000
- Number of classes: 20
- Pretrained SCHP LIP checkpoint

The original SCHP evaluation code was adapted only where necessary to run the original implementation in the modern project environment.

---

## 5. Results

The complete LIP validation set was evaluated successfully.

### Overall metrics

| Metric | Result |
|---|---:|
| Pixel Accuracy | **88.10%** |
| Mean Accuracy | **72.76%** |
| Mean IoU (mIoU) | **58.62%** |

The complete evaluation processed all:

**10,000 / 10,000 validation images**

---

## 6. Per-Class Results

| Class | IoU / Accuracy (%) |
|---|---:|
| Background | 88.36 |
| Hat | 69.96 |
| Hair | 73.55 |
| Glove | 50.46 |
| Sunglasses | 40.74 |
| Upper-clothes | 69.93 |
| Dress | 39.01 |
| Coat | 57.45 |
| Socks | 54.29 |
| Pants | 76.00 |
| Jumpsuits | 32.86 |
| Scarf | 26.32 |
| Skirt | 31.70 |
| Face | 76.19 |
| Left-arm | 68.64 |
| Right-arm | 70.92 |
| Left-leg | 67.27 |
| Right-leg | 66.57 |
| Left-shoe | 55.75 |
| Right-shoe | 56.47 |

---

## 7. Interpretation

The pretrained SCHP model provides a strong human-parsing baseline for the project.

The overall mIoU of **58.62%** indicates that the model can produce meaningful pixel-level segmentation across the 20 LIP semantic classes.

The model performs particularly well on large and visually distinctive regions such as:

- Background
- Pants
- Face
- Hair
- Upper-clothes
- Arms
- Legs

Smaller or visually ambiguous regions such as scarves, skirts, jumpsuits, sunglasses, and dresses are more difficult and have lower scores.

For the Virtual Try-On system, the most important aspect is that SCHP can provide structured information about clothing and body regions rather than simply detecting the person as one object.

---

## 8. Project Decision

The pretrained SCHP checkpoint will be used as the initial human-parsing component of the Virtual Try-On perception pipeline.

Further training of SCHP is not currently required because the pretrained model already provides a strong validated baseline.

The focus will therefore shift toward integrating the segmentation output with:

1. Pose estimation
2. Person representation
3. Garment processing
4. CatVTON
5. Real-time processing

Any future fine-tuning will be treated as an experimental improvement over this baseline rather than a requirement for the initial system.

---

## 9. Baseline Status

**Status: VALIDATED**

- Dataset verified
- Pretrained checkpoint verified
- Original evaluation pipeline executed
- 10,000 validation images evaluated
- mIoU: **58.62%**
- Pixel accuracy: **88.10%**

This baseline will be used for comparison against any future segmentation experiments.
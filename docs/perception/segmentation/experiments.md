# Segmentation Experiments

## Purpose

This document records experiments performed during development of the human parsing component.

The validated SCHP pretrained checkpoint is documented separately in `baseline.md`.

---

## 1. Initial SCHP Evaluation

An initial evaluation pipeline was developed to test the pretrained SCHP model.

The first direct single-scale evaluation produced a lower mIoU than the published SCHP result.

This led to investigation of the original SCHP evaluation procedure.

The original repository evaluation code was subsequently reproduced using the original checkpoint and LIP validation structure.

---

## 2. Original SCHP Evaluation Reproduction

The original SCHP repository evaluation pipeline was successfully executed in a modern Windows environment.

Compatibility work was required because the original repository uses an older PyTorch/CUDA toolchain.

The model's custom InPlaceABNSync extension was successfully compiled after compatibility adjustments.

The original checkpoint was then evaluated on all 10,000 LIP validation images.

Result:

```text
mIoU = 58.62%
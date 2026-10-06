# Segmentation Experiments

## Purpose

This document records the experiments and investigations performed during development of the human-parsing component.

The validated SCHP pretrained baseline is documented separately in [`baseline.md`](baseline.md).

The experiments recorded here include evaluation investigations, test-time augmentation, training-pipeline verification, and fine-tuning experiments.

---

## 1. Initial SCHP Evaluation

An initial evaluation pipeline was developed to test the pretrained SCHP model.

The first direct single-scale evaluation produced a substantially lower mIoU than the published SCHP result.

Rather than assuming that the pretrained model was underperforming, the evaluation procedure was investigated to determine whether differences in preprocessing, model output handling, or evaluation methodology were responsible.

This led to reproduction of the original SCHP evaluation procedure.

### Outcome

The initial custom evaluation was not used as the project's official quantitative baseline.

---

## 2. Original SCHP Evaluation Reproduction

The original SCHP repository evaluation pipeline was successfully reproduced in a modern Windows environment.

The original repository uses an older PyTorch/CUDA toolchain and a custom `InPlaceABNSync` extension.

Compatibility work was therefore required to:

- configure a compatible PyTorch/CUDA environment,
- configure the CUDA development toolchain,
- configure the Visual Studio C++ build environment,
- compile the custom InPlaceABNSync extension,
- apply compatibility adjustments to the original extension source.

The original pretrained SCHP LIP checkpoint was then evaluated using the original evaluation procedure.

### Evaluation Set

The complete LIP validation set was used:

```text
Validation images: 10,000
Semantic classes: 20
Input resolution: 473 × 473
Batch size: 1
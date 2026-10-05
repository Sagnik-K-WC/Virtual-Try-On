# Human Parsing / Segmentation

## Overview

The segmentation component of the Virtual Try-On system performs human parsing on a person image.

Human parsing is a pixel-level semantic segmentation task in which each pixel is assigned a semantic human-part or clothing class.

For this project, the segmentation component is based on the SCHP (Self-Correction for Human Parsing) model using the LIP human parsing label space.

## Purpose in the Virtual Try-On System

Segmentation provides a pixel-level representation of the person's body and clothing.

The resulting human parsing mask can later be used by the Virtual Try-On pipeline to determine regions such as:

- Upper clothes
- Dress
- Coat
- Pants
- Arms
- Legs
- Shoes
- Face
- Hair

This information will be combined with pose estimation and other perception outputs before integration with the Virtual Try-On pipeline.

## Dataset

The project uses the LIP (Look into Person) human parsing dataset.

Dataset statistics used in this project:

- Training images: 30,462
- Validation images: 10,000
- Semantic classes: 20
- Training masks: 30,462
- Validation masks: 10,000

The dataset was verified programmatically before model evaluation.

## Model

The current baseline model is:

`pirocheto/schp-lip-20`

SCHP was selected because it is specifically designed for human parsing and directly uses the 20-class LIP label space.

The pretrained model produces:

- Parsing logits: 20 classes
- Edge logits: 2 classes
- Input resolution: 473 × 473

## Current Status

The pretrained SCHP checkpoint has been successfully validated on the LIP validation set.

The model:

- Loads successfully on the NVIDIA GeForce RTX 5060 Laptop GPU
- Produces 20-class human parsing predictions
- Produces visually meaningful segmentation masks
- Has been evaluated on all 10,000 LIP validation images

See [`baseline.md`](baseline.md) for quantitative results.

## Repository Structure

```text
perception/
└── segmentation/
    ├── datasets/
    ├── evaluation/
    ├── experiments/
    └── inference/
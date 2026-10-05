# SCHP Segmentation Baseline

## Checkpoint

**Checkpoint:** SCHP Human Parsing Baseline Established

**Status:** Validated

The pretrained SCHP human parsing model was successfully integrated into the project and evaluated on the complete LIP validation set.

---

## 1. Dataset Verification

The LIP dataset was verified before model evaluation.

### Training split

- Images: 30,462
- Masks: 30,462
- Missing masks: 0
- Extra masks: 0

### Validation split

- Images: 10,000
- Masks: 10,000
- Missing masks: 0
- Extra masks: 0

Both dataset splits passed verification.

---

## 2. Human Parsing Classes

The LIP label space contains 20 semantic classes:

| ID | Class |
|---:|---|
| 0 | Background |
| 1 | Hat |
| 2 | Hair |
| 3 | Glove |
| 4 | Sunglasses |
| 5 | Upper-clothes |
| 6 | Dress |
| 7 | Coat |
| 8 | Socks |
| 9 | Pants |
| 10 | Jumpsuits |
| 11 | Scarf |
| 12 | Skirt |
| 13 | Face |
| 14 | Left-arm |
| 15 | Right-arm |
| 16 | Left-leg |
| 17 | Right-leg |
| 18 | Left-shoe |
| 19 | Right-shoe |

---

## 3. Model

The baseline uses:

`pirocheto/schp-lip-20`

The model is loaded using the Hugging Face Transformers interface.

The model produces:

```text
Parsing logits: [1, 20, 473, 473]
Edge logits:    [1, 2, 473, 473]
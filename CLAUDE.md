# CLAUDE.md

## Project Overview

This project is a research-oriented semantic segmentation project based on **SegFormer-B0**.

The primary goal is to improve lightweight segmentation performance by:

* keeping the **SegFormer-B0 encoder architecture fixed**
* comparing **decoder variants** and **loss functions**
* conducting experiments under a **single-variable principle**
* separating **controlled experiments** and **performance-oriented experiments**

This is not a production system.

Priority order:

1. structural correctness
2. experiment fairness
3. reproducibility
4. minimal code modification

---

## Core Project Principles

### 1. Encoder must remain unchanged

The encoder must remain architecturally identical to SegFormer-B0.

Allowed:

* modular refactoring

Not allowed:

* changing architecture
* changing dimensions / depth / heads / sr_ratios

---

### 2. Decoder and loss are experiment variables

The encoder is fixed.

Only the following can change:

* decoder (MLP / FPN)
* loss function (CE / Focal / Dice / Boundary / Combined)

---

### 3. Single-variable experiment rule

Each experiment must change only one factor at a time.

Examples:

* E0 vs E1 → decoder only
* E0 vs E2 → loss only

---

## Experiment Strategy (Final Design)

The project uses a **two-stage experiment pipeline**.

---

## Step 1 — Internal Experiments (E0 ~ E4)

### Purpose

* Fair comparison of decoder and loss
* Isolate structural effects only

---

### Fixed Training Conditions (All E0 ~ E4)

* pretrained: false
* augmentation: **basic only**
* scheduler: simple (poly decay, already implemented)
* epochs: 40
* optimizer: AdamW (lr=6e-5)
* batch size: fixed

---

### Allowed Changes per Experiment

Only these can change:

* model (decoder)
* criterion (loss)

---

### Experiment Matrix

| Experiment | Decoder | Loss          |
| ---------- | ------- | ------------- |
| E0         | MLP     | CE            |
| E1         | FPN     | CE            |
| E2         | MLP     | Focal         |
| E3         | MLP     | CE + Dice     |
| E4         | MLP     | CE + Boundary |

---

### Important Notes

* Absolute performance is NOT the goal
* Relative performance comparison is the goal
* Small dataset instability is acceptable
* Best checkpoint must be saved

---

### Implementation Rule

Do NOT refactor training pipeline.

Only modify:

* model construction
* loss construction
* exp_name

---

## Step 2 — Paper-like Experiment (E5)

### Purpose

* Evaluate the **maximum achievable performance** of the best structure

---

### Procedure

1. Select best model from E0 ~ E4
2. Apply paper-like settings ONLY to that model

---

### E5 Configuration

* augmentation: **paperlike**
* pretrained: (to be added in next step)
* scheduler: (to be improved in next step)
* longer training (80~100 epochs recommended)

---

### Important Rule

Paper-like settings must NOT be used in E0~E4.

---

## Config System Usage

Two config types are used:

### Internal config

Used for E0 ~ E4:

* augmentation_type: basic
* pretrained: false

Example:

configs/e0_internal.yaml

---

### Paper-like config

Used ONLY for E5:

* augmentation_type: paperlike
* pretrained: true (future step)

Example:

configs/e0_paperlike.yaml
(Optionally rename to e5_best.yaml)

---

## Data & Transform Rules

### Basic Transform

* Resize
* Normalize
* ToTensor

---

### Paper-like Transform

Train:

* RandomResize (0.5 ~ 2.0)
* RandomCrop
* RandomHorizontalFlip
* ColorJitter
* Normalize
* ToTensor

Val:

* Resize only

---

### Critical Constraints

* image interpolation → bilinear
* mask interpolation → nearest
* mask class index must never be corrupted

---

## Repository Rules

* Do not import MMSeg / MMCV / MMEngine
* Do not rewrite project
* Maintain compatibility with existing code
* Use config instead of hardcoding where possible
* Keep changes minimal

---

## What Claude Should Do

When responding:

* modify only requested parts
* do not refactor entire files
* preserve working code
* ensure compatibility with tests
* follow experiment design strictly

---

## Final Summary

* E0~E4 → internal controlled comparison
* E5 → best structure with paper-like settings
* separation of concerns is critical

---

One-line summary:

Compare fairly first, then maximize performance.
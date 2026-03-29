# SegFormer Lab - Lightweight Semantic Segmentation Study

## 📌 Project Overview
This project aims to improve lightweight semantic segmentation performance based on **SegFormer-B0**.

### Key Directions
- Encoder: Keep original SegFormer-B0 structure (no modification)
- Decoder: Replace MLP decoder with FPN-based decoder
- Loss: Compare CE, Focal, CE+Dice, CE+Boundary
- Principle: Single-variable analysis for fair comparison

---

## 🧱 Project Structure

segformer-core/
├── models/
├── data/
├── configs/
├── utils/
├── scripts/
├── notebooks/
├── tests/
├── weights/
├── outputs/

---

## ⚙️ Setup

### 1. Clone repository
git clone https://github.com/iNES-Segmentation-Project/segformer-core
cd segformer-core

### 2. Create virtual environment
python -m venv .venv

### Activate

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

### 3. Install dependencies
pip install -r requirements.txt

---

## 🚀 Training
python scripts/train.py --config configs/b0_mlp_ce.yaml

---

## 📊 Evaluation
python scripts/evaluate.py --config configs/b0_mlp_ce.yaml

---

## 🧪 Experiments

ID    Encoder         Decoder    Loss
E0    SegFormer-B0    MLP        CE
E1    SegFormer-B0    FPN        CE
E2    SegFormer-B0    MLP        Focal
E3    SegFormer-B0    MLP        CE+Dice
E4    SegFormer-B0    MLP        CE+Boundary
E5    SegFormer-B0    FPN        Best

---

## 📌 Metrics
- mIoU
- Per-class IoU
- Boundary IoU
- FPS / Latency
- FLOPs / Parameters

---

## 👥 Team Roles

MJ
- Baseline reproduction
- Encoder modularization
- Training pipeline & metrics

찬호
- FPN decoder
- Loss functions
- Visualization & presentation

---

## 📎 Notes
- Encoder must remain identical to SegFormer-B0
- Ensure fair comparison across experiments
- CamVid → Cityscapes validation

---

## 📌 Status
🚧 In progress (Week 1: Baseline & Environment Setup)
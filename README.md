# SegFormer-B0 Semantic Segmentation Research Project

> **Lightweight SegFormer에서 decoder 구조와 multi-scale feature fusion이 segmentation 성능에 어떤 영향을 주는지 controlled experiment로 분석한 연구형 프로젝트입니다.**

---

## 1. Overview

이 프로젝트는 단순히 SegFormer-B0를 학습한 구현 프로젝트가 아니라,  
**“왜 성능이 좋아졌는가?”**를 실험적으로 해석하기 위해 설계한 semantic segmentation 연구 프로젝트입니다.

핵심 가설은 다음과 같습니다.

> SegFormer는 encoder 중심 구조이지만,  
> MiT-B0 같은 lightweight encoder에서는 feature representation이 제한될 수 있다.  
> 따라서 decoder 단계의 **multi-scale feature fusion**이 small/thin object segmentation과 generalization에 더 큰 영향을 줄 수 있다.

본 프로젝트에서는 이 가설을 검증하기 위해:

- Stage 1: controlled experiment
- Stage 2: combined improvement experiment
- Stage 2+: decoder effect isolation

순서로 실험을 설계했습니다.

---

## 2. Research Motivation

SegFormer는 transformer 기반 semantic segmentation 모델로,  
대부분의 표현 학습을 encoder가 담당하고 decoder는 비교적 단순한 All-MLP 구조를 사용합니다.

하지만 이 프로젝트에서는 다음 문제의식에서 출발했습니다.

### 문제의식

- SegFormer-B0는 lightweight encoder이기 때문에 표현력이 제한될 수 있음
- 기본 MLP decoder는 multi-scale feature를 단순히 projection 후 concat하는 구조
- small object, thin object, boundary 영역에서는 feature fusion 방식이 중요할 수 있음
- 단순히 성능이 오른 것보다, **무엇 때문에 좋아졌는지 해석하는 과정**이 더 중요함

따라서 이 프로젝트의 핵심 질문은 다음과 같습니다.

> Lightweight encoder 기반 segmentation에서  
> decoder의 multi-scale feature fusion은 실제 성능과 generalization에 의미 있는 영향을 주는가?

---

## 3. Experimental Flow

<추후 추가할 Experimental Flow Diagram>

```text
Research Hypothesis
    ↓
Stage 1: Controlled Experiment
- E0 baseline
- decoder / loss 단일 변수 비교
    ↓
FPN decoder improvement 발견
    ↓
Stage 2: E5 Combined Experiment
- FPN + pretrained + augmentation + combined loss
    ↓
큰 성능 향상 확인
    ↓
하지만 원인 분리 한계 발견
"FPN 때문인가? pretrained 때문인가?"
    ↓
Stage 2+: Decoder Effect Isolation
- pretrained, loss, scheduler 고정
- MLP vs FPN decoder만 비교
    ↓
Final Insight
- FPN decoder 효과가 독립적으로도 유지됨
- small/thin object에서 개선 집중
```

이 흐름의 핵심은 단순히 결과를 나열하는 것이 아니라,  
**가설 → 실험 → 한계 인지 → 추가 검증 → 해석**의 연구 과정을 보여주는 것입니다.

---

## 4. Key Results at a Glance

<추후 추가할 핵심 결과 시각화 PNG>

| Exp | Main Change | Test mIoU | Change |
|---|---|---:|---:|
| E0 | Baseline: MLP + CE | 0.5682 | — |
| E1 | FPN Decoder | **0.5829** | **+0.0147** |
| E2 | Focal Loss | 0.5669 | **-0.0013** |
| E3 | CE + Dice | **0.5796** | **+0.0114** |
| E4 | CE + Boundary | 0.5705 | +0.0023 |
| E5 | Full Pipeline | **0.7572** | **+0.1890** |

### 핵심 해석

- FPN decoder는 E0 대비 test mIoU를 **+0.0147** 개선했습니다.
- CE+Dice는 loss 변경 중 가장 안정적인 generalization을 보였습니다.
- Focal loss는 validation에서는 좋아졌지만 test에서는 오히려 감소했습니다.
- E5는 큰 향상을 보였지만, 여러 요소가 동시에 바뀐 복합 실험이므로 원인 분리가 필요했습니다.

---

## 5. Research Questions

| RQ | Question | Related Experiment |
|---|---|---|
| RQ1 | FPN decoder는 lightweight encoder의 한계를 보완할 수 있는가? | E1, Stage 2+ |
| RQ2 | Loss function 조합은 generalization에 어떤 영향을 주는가? | E0–E4 |
| RQ3 | E5의 큰 성능 향상은 어떻게 해석해야 하는가? | E5, Stage 2+ |
| RQ4 | Validation/test discrepancy는 어떤 패턴으로 나타나는가? | E0–E5 |

---

## 6. Experimental Design

## Stage 1 — Controlled Single-Variable Experiment

Stage 1에서는 실험 해석 가능성을 높이기 위해  
**한 번에 하나의 변수만 바꾸는 controlled experiment**를 수행했습니다.

| Exp | Decoder | Loss | Changed Variable | Purpose |
|---|---|---|---|---|
| E0 | MLP | CE | Baseline | 기준 성능 확보 |
| E1 | FPN | CE | Decoder only | FPN decoder 효과 확인 |
| E2 | MLP | Focal | Loss only | class imbalance 영향 분석 |
| E3 | MLP | CE+Dice | Loss only | overlap quality 분석 |
| E4 | MLP | CE+Boundary | Loss only | boundary quality 분석 |

### Stage 1에서 보고자 한 것

- decoder 구조 변경이 실제로 의미 있는가?
- loss 변경은 validation뿐 아니라 test에서도 유지되는가?
- small/thin object에서 FPN의 효과가 나타나는가?
- validation improvement가 generalization improvement로 이어지는가?

---

## Stage 2 — Combined Improvement Experiment

Stage 1 결과를 바탕으로, 성능 향상 가능성이 높은 요소를 조합한 E5 실험을 구성했습니다.

| Exp | Decoder | Loss | Pretrained | Augmentation | Purpose |
|---|---|---|---|---|---|
| E5 | FPN | CE+Dice+Boundary | Yes | Paper-like | pipeline-level improvement |

E5는 test mIoU **0.7572**를 기록하며 E0 대비 **+0.1890** 향상되었습니다.

하지만 이 결과는 신중하게 해석했습니다.

> E5는 FPN decoder, pretrained encoder, augmentation, scheduler, combined loss가 동시에 적용된 복합 실험입니다.  
> 따라서 성능 향상을 특정 요소 하나의 효과로 단정할 수 없습니다.

---

## Stage 2+ — Decoder Effect Isolation

<추후 추가할 E5 해석 한계 & Stage 2+ 필요성 설명 그림>

E5의 가장 큰 한계는  
**“FPN 때문에 좋아진 것인지, pretrained 때문에 좋아진 것인지 분리하기 어렵다”**는 점이었습니다.

이를 보완하기 위해 Stage 2+에서는 다음 조건을 고정했습니다.

- pretrained encoder
- CE loss
- warmup poly scheduler
- augmentation
- train/val/test split

그리고 decoder만 변경했습니다.

| Setting | Decoder | Pretrained | Loss | Test mIoU | Change |
|---|---|---|---|---:|---:|
| Paper-like E0 | MLP | Yes | CE | 0.7314 | — |
| FPN + CE | FPN | Yes | CE | **0.7490** | **+0.0176** |

### Stage 2+ 해석

Stage 2+에서도 FPN decoder가 MLP보다 높은 test mIoU를 보였습니다.

즉, E5의 성능 향상이 전부 pretrained 때문이라고 보기는 어렵고,  
decoder 구조 자체의 효과도 일정 부분 존재한다고 해석할 수 있습니다.

---

## 7. Class-Level Improvement

FPN decoder의 개선은 전체 mIoU뿐 아니라  
특정 small/thin object class에서 더 뚜렷하게 나타났습니다.

<추후 추가할 class-level improvement visualization>

| Class | MLP | FPN | Change |
|---|---:|---:|---:|
| Pole | 0.4076 | **0.4672** | **+0.0596** |
| SignSymbol | 0.4646 | **0.5106** | **+0.0460** |
| Pedestrian | 0.5453 | **0.5791** | **+0.0338** |
| Pavement | 0.8483 | **0.8679** | +0.0196 |
| Car | 0.8531 | **0.8695** | +0.0164 |
| Bicyclist | **0.6425** | 0.6222 | **-0.0203** |

### 가설과의 연결

RQ1에서 세운 가설은 다음이었습니다.

> FPN decoder의 multi-scale feature fusion이 lightweight encoder의 표현 한계를 일부 보완할 수 있다.

Stage 2+ 결과에서 Pole, SignSymbol, Pedestrian처럼  
작거나 얇은 객체에서 improvement가 집중적으로 나타났습니다.

이는 FPN의 top-down pathway가 깊은 semantic feature를 얕은 high-resolution feature로 전달하면서  
small/thin object segmentation에 도움이 되었을 가능성과 연결됩니다.

다만 Bicyclist class에서는 성능이 감소했기 때문에,  
FPN이 모든 class에 동일하게 유리하다고 해석하지 않았습니다.

---

## 8. Core Implementation Highlights

이 프로젝트의 중요한 구현 포인트는  
단순히 모델을 학습한 것이 아니라,  
**실험 가능한 segmentation research pipeline을 직접 구현했다는 점**입니다.

---

### 1. Encoder–Decoder Modularization

SegFormer를 encoder와 decoder가 분리된 구조로 구현했습니다.

```text
Input Image
    ↓
MiT-B0 Encoder
    ↓
Multi-scale Features: c1, c2, c3, c4
    ↓
Decoder: MLP or FPN
    ↓
Segmentation Logits
```

이를 통해 동일한 MiT-B0 encoder 위에서:

- MLP decoder
- FPN decoder

를 교체 실험할 수 있도록 만들었습니다.

핵심은 다음입니다.

> 실험 통제를 위해 decoder를 interchangeable component로 분리했습니다.

---

### 2. Baseline MLP Decoder Reproduction

Baseline MLP decoder는 SegFormer 논문 구조에 맞춰 구현했습니다.

```text
c1, c2, c3, c4
    ↓
1×1 projection
    ↓
upsample to c1 resolution
    ↓
concat
    ↓
1×1 fusion conv
    ↓
segmentation head
```

MLP decoder는 각 stage feature를 독립적으로 projection한 뒤 concat합니다.  
즉, stage 간 top-down interaction은 거의 없고, 마지막 fusion conv에서만 정보가 섞입니다.

---

### 3. Custom FPN Decoder

FPN decoder는 lightweight encoder의 multi-scale feature 표현 한계를 보완하기 위해 직접 구현했습니다.

```text
c4 → c3 → c2 → c1
top-down semantic propagation
```

구현 요소:

- lateral connection
- top-down pathway
- 3×3 output convolution
- multi-scale feature fusion

FPN decoder의 목적은  
깊은 layer의 semantic 정보를 얕은 layer의 high-resolution feature에 전달하는 것입니다.

---

### 4. YAML Config-Based Experiment Pipeline

실험 재현성을 위해 주요 설정을 YAML config로 관리했습니다.

- model type
- loss type
- pretrained 여부
- augmentation
- scheduler
- learning rate
- epoch
- input size

이를 통해 실험마다 코드를 직접 수정하지 않고,  
config 변경만으로 E0–E5 및 Stage 2+ 실험을 수행할 수 있도록 구성했습니다.

---

### 5. Pretrained Weight Remapping

pretrained 실험에서는 HuggingFace `mit-b0` weight를  
직접 구현한 MiTEncoder 구조에 맞게 remapping하는 로직을 구현했습니다.

특히 attention 구조에서:

- HuggingFace: key/value projection 분리
- Custom Encoder: key/value projection 통합

차이가 있었기 때문에,  
state_dict key 변환과 k/v weight concat 처리를 직접 구현했습니다.

---

### Core Implementation Message

> 이 프로젝트는 단순히 SegFormer를 학습한 것이 아닙니다.  
> 실험 통제를 위해 구조를 modular하게 구현하고,  
> baseline decoder와 custom FPN decoder를  
> 동일 조건에서 비교 가능한 형태로 직접 구현한 프로젝트입니다.

---

## 9. Results Linked to Research Questions

## RQ1. FPN decoder는 lightweight encoder의 한계를 보완했는가?

E1과 Stage 2+에서 FPN decoder는 MLP decoder보다 일관되게 높은 성능을 보였습니다.

| Comparison | Result |
|---|---|
| E0 → E1 | +0.0147 test mIoU |
| Stage 2+ MLP → FPN | +0.0176 test mIoU |

특히 small/thin object에서 improvement가 집중되었습니다.

따라서 RQ1에 대해서는 다음과 같이 해석했습니다.

> FPN decoder는 lightweight encoder에서 부족할 수 있는 multi-scale representation을 일부 보완할 가능성이 있다.

---

## RQ2. Loss function은 generalization에 어떤 영향을 주었는가?

| Loss | Test mIoU Change | Interpretation |
|---|---:|---|
| Focal | -0.0013 | validation improvement가 test로 이어지지 않음 |
| CE+Dice | **+0.0114** | 가장 안정적인 generalization |
| CE+Boundary | +0.0023 | test에서 효과 약화 |

Loss 실험에서 중요한 점은  
validation score만으로는 실제 generalization을 판단하기 어렵다는 것이었습니다.

특히 Focal loss는 validation에서는 좋아졌지만 test에서는 감소했습니다.  
따라서 이 프로젝트에서는 단순 val score보다  
**validation/test discrepancy**를 함께 해석했습니다.

---

## RQ3. E5의 큰 성능 향상은 어떻게 해석해야 하는가?

E5는 E0 대비 test mIoU가 **+0.1890** 향상되었습니다.

하지만 이 향상은 다음 요소들이 동시에 작용한 결과입니다.

- FPN decoder
- pretrained encoder
- augmentation
- combined loss
- scheduler change

따라서 E5의 결과는:

> “FPN 하나로 좋아졌다”가 아니라,  
> “pipeline-level improvement는 확인했지만, 개별 요인 분리는 추가 실험이 필요하다”

로 해석했습니다.

이 문제의식이 Stage 2+ 실험으로 이어졌습니다.

---

## RQ4. Validation/Test Discrepancy는 어떻게 나타났는가?

| Exp | Val Best mIoU | Test mIoU | Gap |
|---|---:|---:|---:|
| E0 | 0.6369 | 0.5682 | 0.0687 |
| E1 | 0.6626 | 0.5829 | 0.0797 |
| E2 | 0.6503 | 0.5669 | 0.0834 |
| E3 | 0.6518 | 0.5796 | 0.0722 |
| E4 | 0.6510 | 0.5705 | 0.0805 |
| E5 | 0.8043 | 0.7572 | **0.0471** |

E0–E4는 scratch training 조건에서 비교적 큰 val/test gap을 보였습니다.

반면 E5는 pretrained와 augmentation을 적용하면서 gap이 줄었습니다.  
이는 generalization 측면에서 pretrained + augmentation이 도움이 되었을 가능성을 보여줍니다.

---

## 10. Qualitative Comparison

<추후 추가할 qualitative comparison>

예정 비교 구성:

| Input | Ground Truth | MLP Prediction | FPN Prediction |
|---|---|---|---|
| 추후 추가 | 추후 추가 | 추후 추가 | 추후 추가 |

분석 예정 포인트:

- small object 검출 여부
- thin object boundary 보존
- false positive 감소 여부
- class confusion 발생 여부
- 실패 사례 분석

README에는 성공 사례만 넣지 않고,  
FPN이 실패한 사례도 함께 포함할 예정입니다.

---

## 11. Key Insights

### 1. Decoder 구조는 lightweight segmentation에서 중요한 변수였다

FPN decoder는 단순히 전체 mIoU만 올린 것이 아니라,  
small/thin object에서 improvement가 집중적으로 나타났습니다.

---

### 2. Loss function은 validation보다 test generalization 기준으로 봐야 한다

Focal loss처럼 validation에서는 좋아 보여도  
test에서는 오히려 떨어지는 경우가 있었습니다.

---

### 3. E5의 큰 성능 향상은 신중하게 해석해야 한다

E5는 성능 향상 실험으로는 의미가 있지만,  
여러 요소가 동시에 변경되었기 때문에 원인 분석에는 한계가 있습니다.

---

### 4. 한계를 발견한 뒤 Stage 2+로 추가 검증했다

이 프로젝트에서 중요한 부분은  
E5의 한계를 그대로 인정하고,  
decoder effect를 분리하기 위한 Stage 2+ 실험을 추가 설계했다는 점입니다.

---

## 12. Limitations

- E5는 복합 실험이므로 개별 요소 기여도를 완전히 분리할 수 없습니다.
- Stage 2+에서도 FPN의 구조적 효과와 parameter 증가 효과를 완전히 분리하지는 못했습니다.
- multi-seed 반복 실험을 수행하지 않아 통계적 안정성 검증은 제한적입니다.
- CamVid는 train set이 작기 때문에 validation/test discrepancy가 발생하기 쉽습니다.
- 일부 class, 특히 Bicyclist에서는 FPN이 MLP보다 낮은 결과를 보였습니다.

---

## 13. Team / Role

## 2인 프로젝트

### 전민지 — PM / Research & Implementation

- 전체 실험 일정 관리 및 PM 역할
- 연구 질문 및 실험 구조 설계 참여
- E0, E1, E5 실험 수행 및 결과 분석
- SegFormer encoder-decoder 분리 구조 구현
- FPN decoder 구현
- pretrained weight remapping 구현
- config 기반 실험 pipeline 구현
- metric 및 visualization pipeline 구현
- 실험 결과 해석 및 README 작성

### 팀원

- E2, E4, Stage 2+ 실험 수행
- 실험 결과 정리 및 보조 분석 참여

---

## 14. What I Learned

이 프로젝트를 통해 가장 크게 배운 점은 다음입니다.

> segmentation 성능은 하나의 최종 점수만으로 해석할 수 없다.

성능 향상을 해석하려면:

- 어떤 변수를 바꿨는지
- 다른 조건은 통제되었는지
- validation improvement가 test에서도 유지되는지
- 특정 class에서만 좋아진 것은 아닌지
- 복합 실험의 원인을 분리할 수 있는지

를 함께 봐야 했습니다.

특히 E5 실험을 통해  
성능이 크게 올랐다고 해서 바로 특정 구조의 효과라고 주장하면 안 된다는 점을 배웠습니다.

그래서 Stage 2+를 추가 설계했고,  
이 과정에서 controlled experiment의 중요성을 직접 경험했습니다.

---

## 15. Repository Structure

```bash
├── configs/
├── data/
├── models/
│   ├── encoder/
│   ├── decoder/
│   └── loss/
├── scripts/
├── utils/
├── checkpoints/
├── outputs/
└── README.md
```

---

## 16. Tech Stack

- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Albumentations
- Semantic Segmentation
- Computer Vision

---

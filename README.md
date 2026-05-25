# SegFormer-B0 Semantic Segmentation Research Project

**Decoder 구조 변경은 경량 encoder의 한계를 완화할 수 있는가?**

> 본 프로젝트는 SegFormer-B0에서 **decoder 구조(FPN)의 실제 기여**를  
> **통제 실험**, **decoder 효과 분리 실험** 기반으로 분석한 연구 프로젝트입니다.

<p align="center">
  <sub>가천대학교 의공학과 iNES AI팀 · 전민지, 백찬호 &nbsp;|&nbsp; 프로젝트 기간 : 2026.03 – 2026.04</sub>
</p>

---

# 1. 개요

본 프로젝트는 **SegFormer-B0**를 기반으로,  
**경량 encoder 환경에서 decoder 구조 변경이 segmentation 성능에 미치는 영향**을 분석한 연구입니다.

본 연구는 **SegFormer 논문의 model size ablation 결과**에서 출발했습니다.  
SegFormer는 동일한 lightweight All-MLP decoder를 사용하면서도,  
**MiT-B0부터 MiT-B5까지 encoder 규모에 따라 mIoU 차이가 크게 발생**합니다.

<p align="center">
  <img src="assets/reference.png" width="700"/>
</p>

> **SegFormer 논문 참고 결과**  
> MiT encoder scale에 따라 segmentation 성능 차이가 발생했습니다.

이 결과를 바탕으로 다음과 같은 연구 가설을 설정했습니다.

> **encoder capacity가 제한된 SegFormer-B0에서는  
> decoder 단계의 multi-scale feature fusion이 더 중요한 역할을 할 수 있다.**

참고 논문

> **SegFormer : Simple and Efficient Design for Semantic Segmentation with Transformers**  
> https://arxiv.org/abs/2105.15203

본 프로젝트에서는 단순 성능 향상보다,  
**"왜 성능이 좋아졌는가?"를 검증하기 위한 통제 실험과 decoder 효과 분리 분석**에 집중했습니다.

**validation 지표만으로는 실제 일반화 성능을 충분히 설명하기 어렵다**는 점도 실험 과정에서 확인되었습니다.  
E2(Focal Loss) 실험에서 **validation 성능과 test 성능 간 괴리**가 관찰되었고,  
E5에서는 수치상의 큰 향상에도 불구하고 **기여 원인을 분리하지 않으면 결론이 달라질 수 있었습니다.**  
이에 따라 단순 성능 수치보다 **class-wise 분석과 failure case 검증을 실험의 핵심 과정으로 포함**했습니다.

---

## 핵심 결과 요약

E5 combined experiment에서 **+0.1890 mIoU**가 관찰되었지만, 복수의 변수가 동시에 변경된 실험이었기 때문에 **FPN 단독 기여로 해석할 수 없었습니다.**  
이를 보완하기 위해 decoder만 변경하는 Stage 2+ isolation experiment를 추가 설계했고, scratch/pretrained 두 조건 모두에서 **FPN 향상이 재현(+0.0147 → +0.0176)** 되었습니다.  
단, **Bicyclist class에서는 오히려 성능이 감소(-0.0203)** 하여 FPN 효과가 모든 class에 보편적으로 작용하지 않는다는 점도 함께 검증했습니다.

| 실험 | 목적 | 결과 |
|---|---|---|
| Stage 1 | decoder/loss 단일 변수 비교 | FPN **+0.0147** mIoU |
| Stage 2 | 성능 향상 요소 조합 | E5 **+0.1890** mIoU |
| Stage 2+ | decoder 효과 분리 검증 | FPN **+0.0176** mIoU (재현) |
| 주요 분석 | class-wise / failure case 분석 | Bicyclist **-0.0203** 포함 |

---

## 실험 흐름

<details>
<summary>실험 흐름 보기</summary>

```mermaid
flowchart TB
    A["연구 가설"]
    B["Stage 1<br/>E0–E4"]
    C["FPN +0.0147"]
    D["Stage 2<br/>E5"]
    E["Pipeline Gain<br/>+0.1890"]
    F["Attribution Issue"]
    G["Stage 2+<br/>Decoder Isolation"]
    H["FPN Retained<br/>+0.0176"]

    A --> B --> C --> D --> E --> F --> G --> H
```

</details>

---

# 2. 팀 및 역할

### 전민지 — PM · Research

<details>
<summary>세부 역할</summary>

- Stage 1 controlled experiment 설계 및 구현
- FPN / MLP decoder 구현
- 클래스별(class-wise) 성능 및 failure case 분석
- YAML 기반 실험 설정 관리 구조 구현
- 실험 결과 시각화 및 정리
- GitHub repository 및 Notion 관리

</details>

### 백찬호 — Research

<details>
<summary>세부 역할</summary>

- Stage 2 및 pretrained pipeline 실험 설계 및 구현
- Encoder Stage별 모듈화 구조 구현
- decoder isolation 실험 구현 및 결과 비교
- 학습 pipeline 최적화
- validation/test 성능 차이 분석

</details>

### 공동 기여

<details>
<summary>세부 역할</summary>

- 연구 질문 및 전체 실험 전략 설계
- controlled experiment 환경 구성
- 실험 결과 비교 및 검증

</details>

---

## 연구 진행 일정

<details>
<summary>주차별 진행 과정 보기</summary>

| 기간 | 진행 내용 |
|---|---|
| 1주차 | 연구 주제 선정, 실험 전략 설계, GitHub·Notion 환경 구성, Encoder 모듈화 |
| 2주차 | Stage 1 controlled experiment 진행, 결과 분석 및 시각화 |
| 3주차 | Stage 2 실험 진행, pretrained pipeline 비교 및 성능 검증 |
| 4주차 | Stage 2+ decoder isolation 실험 설계·검증, 결과 정리 및 발표 준비 |

</details>

---

# 3. 연구 동기

## SegFormer-B0 한계 가설

- encoder 규모에 따른 큰 mIoU 차이
- 경량 MLP decoder 구조
- 제한적인 multi-scale interaction 가능성

본 프로젝트에서는 다음 가설을 설정했습니다.

> **경량 encoder 환경에서는 decoder 단계의 feature fusion 구조가 segmentation 품질에 더 큰 영향을 줄 수 있다.**

또한 본 연구는 단순 성능 향상이 아니라,

- **decoder 구조 자체의 효과가 존재하는가**
- **validation 성능 향상이 실제 test generalization으로 이어지는가**
- **combined experiment에서 어떤 요소가 실제 성능 향상에 기여했는가**

를 검증하는 방향에 집중했습니다.

---

# 4. 연구 목표

| 목표 | 질문 | 범위 |
|---|---|---|
| G1 | Decoder 구조 변경(FPN)이 경량 encoder의 한계를 완화할 수 있는가? | 본 repo |
| G2 | Loss function 조합이 segmentation 성능과 일반화 성능에 어떤 영향을 주는가? | 본 repo |
| G3 | 제안한 pipeline이 medical segmentation domain에도 적용 가능한가? | 별도 repo |

현재 repo에서는 **G1, G2**를 중심으로 다룹니다.

G3는 Kvasir-SEG 기반 medical segmentation project에서 확인할 수 있습니다.

- https://github.com/iNES-Segmentation-Project/medical-seg-core

---

# 5. 실험 설계

## 핵심 원칙

### Encoder 고정

<details>
<summary>세부 내용</summary>

- MiT-B0 구조 유지
- encoder 기능 변경 금지
- 실험 재사용성을 위한 encoder 모듈화만 수행
- 추가 연산 또는 구조적 변경 없음

</details>

### 단일 변수 원칙

<details>
<summary>세부 내용</summary>

- E0~E4는 decoder 또는 loss 중 하나만 변경
- 동일 학습 조건 유지
- 변수별 영향 분리

</details>

### 공정 비교

<details>
<summary>세부 내용</summary>

- 동일 dataset
- 동일 epoch
- 동일 augmentation 조건
- 동일 metric 기준

</details>

### E5 조합 실험

E0~E4 결과를 기반으로 **성능 향상 가능성이 높은 요소를 조합**했습니다.

<details>
<summary>적용 요소</summary>

- FPN decoder
- CE + Dice + Boundary loss
- pretrained encoder
- diff-LR
- augmentation
- warmup scheduler

> diff-LR은 pretrained encoder와 새로 학습되는 decoder의 학습 속도를 분리하기 위한 설정입니다.  
> encoder representation을 과도하게 깨뜨리지 않으면서 decoder 학습을 안정화하는 목적입니다.

</details>

---

# 6. Stage 1 — 통제 실험

## 설정

Stage 1에서는 **decoder 또는 loss 중 하나만 변경하는 통제 실험**을 수행했습니다.

| Exp | Decoder | Loss | 변경 변수 |
|---|---|---|---|
| E0 | MLP | CE | Baseline |
| E1 | FPN | CE | Decoder |
| E2 | MLP | Focal | Loss |
| E3 | MLP | CE + Dice | Loss |
| E4 | MLP | CE + Boundary | Loss |

---

## 결과 시각화

![Figure 2](assets/fig2_stage1_per_class_heatmap.png)

> E0–E4 단일 변수 실험의 class별 성능 변화입니다.  
> E1(FPN)은 class 전반에 걸쳐 비교적 일관된 향상을 보인 반면, loss 기반 실험(E2–E4)은 class별 변화가 불안정하게 나타났습니다.

## 결과

| Exp | 주요 변경 사항 | Test mIoU | Δ |
|---|---|---:|---:|
| E0 | MLP + CE | 0.5682 | — |
| E1 | FPN Decoder | **0.5829** | **+0.0147** |
| E2 | Focal Loss | 0.5669 | -0.0013 |
| E3 | CE + Dice | **0.5796** | **+0.0114** |
| E4 | CE + Boundary | 0.5705 | +0.0023 |

## 해석

FPN decoder(E1)는 **Pole, SignSymbol, Pedestrian과 같은 작고 얇은 객체 class에서 비교적 일관된 성능 향상**을 보인 반면,  
loss 기반 실험(E2–E4)은 class별 성능 변화가 불안정하게 나타났습니다.

이는 경량 encoder 환경에서 **decoder 기반 multi-scale feature fusion**이  
loss 변경보다 더 안정적인 성능 향상 신호로 작동할 가능성을 시사합니다.

이 결과는 이후 **E5 combined experiment**와 **Stage 2+ decoder isolation 설계**의 핵심 근거가 되었습니다.

---

# 7. Stage 2 — 조합 실험 (E5)

## 설정

Stage 1 결과를 기반으로 **성능 향상 가능성이 높은 요소를 조합**했습니다.

<details>
<summary>적용 요소</summary>

- FPN decoder
- CE + Dice + Boundary loss
- pretrained encoder
- diff-LR
- augmentation
- warmup scheduler

</details>

## 결과

![Figure 1](assets/fig1_e0_to_e5_per_class_improvement.png)

| Exp | 주요 변경 사항 | Test mIoU | Δ |
|---|---|---:|---:|
| E0 | MLP + CE | 0.5682 | — |
| E5 | Combined Pipeline | **0.7572** | **+0.1890** |

## 정성 분석

![Figure 4](assets/fig4_boundary_activation_mlp_vs_fpn.png)

> Stage 1에서 관찰된 패턴이 activation 수준에서도 확인됩니다.  
> 이는 정량 결과를 보조하는 정성적 근거로 해석할 수 있습니다.

<details>
<summary>정성적 segmentation 비교</summary>

### E0 Baseline

![E0](assets/fig5b_qualitative_e0_mlp_ce.png)

### E5 Combined Pipeline

![E5](assets/fig5a_qualitative_e5_fpn_compound.png)

> Stage 1에서 관찰된 경향이 pipeline 전체에서도 유지되었습니다.  
> 단, combined experiment 결과이므로 FPN 단독 효과로는 해석하지 않았습니다.

</details>

## 성능 향상 원인 분리의 한계

> ⚠️ **E5는 combined experiment입니다.**

**FPN decoder · pretrained encoder · augmentation · scheduler · combined loss**가 동시에 변경되었기 때문에,  
**+0.1890의 성능 향상을 특정 요소 하나의 효과로 해석할 수 없습니다.**  
이를 보완하기 위해 Stage 2+ decoder isolation experiment를 추가 설계했습니다.

---

# 8. Stage 2+ — Decoder 효과 분리 실험

## 효과 분리 설계

E5의 attribution issue를 보완하기 위해 **decoder만 변경**하고,  
나머지 모든 조건(pretrained encoder / CE loss / scheduler / augmentation / train-val-test split)은 고정했습니다.

## 결과

| 설정 | Decoder | Test mIoU | Δ |
|---|---|---:|---:|
| Baseline | MLP | 0.7314 | — |
| FPN | FPN | **0.7490** | **+0.0176** |

scratch 조건(E0→E1)과 pretrained 조건(Stage 2+)의 FPN 향상 비교입니다.

| 조건 | MLP → FPN Δ |
|---|---:|
| Scratch (E0 → E1) | +0.0147 |
| Pretrained (Stage 2+) | +0.0176 |

> **training condition 변화와 무관하게 FPN 성능 향상이 재현되었습니다.**  
> 이는 E5 improvement가 단순 pretrained effect만으로 설명되기 어렵고,  
> decoder 구조 자체의 기여 가능성도 존재한다는 근거로 해석할 수 있습니다.

## Class-wise 분석

<p align="center">
  <img src="assets/class-wise barplot.png" width="850"/>
</p>

<details>
<summary>Class별 원본 점수</summary>

| Class | MLP | FPN | Δ |
|---|---:|---:|---:|
| Pole | 0.4076 | **0.4672** | **+0.0596** |
| SignSymbol | 0.4646 | **0.5106** | **+0.0460** |
| Pedestrian | 0.5453 | **0.5791** | **+0.0338** |
| Pavement | 0.8483 | **0.8679** | +0.0196 |
| Car | 0.8531 | **0.8695** | +0.0164 |
| Bicyclist | **0.6425** | 0.6222 | **-0.0203** |

</details>

### 구조적 해석

Pole · SignSymbol · Pedestrian과 같이 **얇고 작은 구조를 가진 class**에서 향상이 집중된 것은,  
FPN decoder의 **top-down feature propagation**이 shallow feature의 부족한 맥락 정보를 deep feature로 보완하면서  
작은 객체의 연속성 유지와 얇은 경계 정보 보존에 기여했을 가능성과 연결됩니다.

## 실패 사례 분석

앞서 관찰된 경향과 달리, **Bicyclist class에서는 오히려 성능이 감소(-0.0203)** 했습니다.

| Class | MLP | FPN | Δ |
|---|---:|---:|---:|
| Bicyclist | **0.6425** | 0.6222 | **-0.0203** |

Bicyclist는 사람과 자전거가 결합된 복합적인 형태를 가진 class입니다.  
FPN의 top-down pathway가 deep feature의 맥락 정보를 강하게 전파할 때,  
**Pedestrian · Pole 등 인접 class와의 경계가 혼동**되었을 가능성이 있습니다.  
또한 CamVid dataset에서 Bicyclist는 상대적으로 빈도가 낮은 class이기 때문에,  
feature fusion의 이점이 충분히 작동하지 않았을 수 있습니다.

> FPN decoder의 효과는 모든 class에 동일하게 나타나는 보편적 성능 향상이라기보다,  
> **특정 객체 특성과 더 강하게 연결된 현상**으로 해석됩니다.

## Validation / Test 성능 차이

![Figure 3](assets/fig3_val_vs_test_trustworthiness.png)

> E2(Focal Loss)에서 validation 성능과 test 성능 간 괴리가 관찰되었습니다.  
> validation score만으로는 loss 효과를 충분히 판단하기 어렵다는 점을 보여주는 결과입니다.

---

# 9. 구현 핵심

- SegFormer encoder/decoder modularization
- FPN / MLP decoder 구현
- YAML 기반 experiment config 관리
- HuggingFace pretrained weight remapping
- metric / visualization pipeline 구현

<details>
<summary>구현 세부 내용</summary>

### Encoder 모듈화

- MiT-B0 구조 유지
- encoder 기능 변경 없음
- 실험 재사용성을 위한 모듈화
- decoder 교체 가능 구조 구성

### Decoder 구현

MLP decoder

- feature projection
- upsampling
- concatenation
- fusion conv

FPN decoder

- lateral connection
- top-down pathway
- multi-scale feature fusion
- feature propagation

### YAML 설정 파이프라인

- model type / decoder type / loss type
- pretrained 여부 / augmentation / scheduler
- learning rate / epoch

**config 변경만으로 E0~E5 및 Stage 2+ 실험을 재현할 수 있도록 구성했습니다.**

### Pretrained 가중치 변환

HuggingFace `mit-b0` weight를 custom MiTEncoder 구조에 맞게 remapping했습니다.

- state_dict key 변환
- key/value projection mapping
- pretrained encoder loading 지원

</details>

---

# 10. 종합 결과

- **FPN decoder는 경량 encoder 환경에서 일관된 성능 향상 신호**를 보였습니다. (scratch +0.0147 / pretrained +0.0176)
- **E5 combined experiment의 +0.1890은 특정 요소 단독 효과로 단정할 수 없으며**, Stage 2+ isolation을 통해 decoder 기여 가능성을 추가 검증했습니다.
- **Bicyclist failure case**를 통해 FPN 효과가 모든 class에 동일하게 적용되지 않는다는 점이 관찰되었습니다.
- **validation/test 성능 괴리** 분석을 통해, metric 수치만으로 실험 결과를 판단하는 데 한계가 있음을 확인했습니다.

---

# 11. 한계점

- **E5는 combined experiment**이므로, 개별 요소의 기여를 완전히 분리하는 데 한계가 존재합니다.
- **Stage 2+에서도 FPN 구조 효과와 parameter 증가 효과를 완전히 분리하기는 어렵습니다.**
- **multi-seed 기반 통계 검증은 컴퓨팅 자원 및 일정 제약으로 수행하지 않았습니다.**
- **FPN decoder는 GFLOPs 증가에 따른 연산량 trade-off**가 존재합니다.
- **CamVid dataset은 CityScapes 대비 데이터 규모가 작기 때문에**, statistical diversity 측면의 한계가 존재합니다.

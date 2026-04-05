"""
models/loss/combined_loss.py

Loss 조합 관리 모듈 — Experiments E3, E4, E5.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[실험별 loss 구성]

  실험 │ Loss 구성                    │ 비고
  ─────┼──────────────────────────────┼──────────────────
  E0   │ CE                           │ baseline
  E1   │ CE                           │ decoder만 FPN으로 변경
  E2   │ Focal                        │ FocalLoss 단독
  E3   │ CE + Dice                    │ CombinedLoss(ce+dice)
  E4   │ CE + Boundary                │ CombinedLoss(ce+boundary)
  E5   │ FPN + best loss              │ 실험 후 결정

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[가중치 설계 원칙]

  CE 계열 loss를 anchor(weight=1.0)로 두고
  보조 loss(Dice, Boundary)의 weight를 조정한다.
  두 loss의 스케일 차이가 크므로 weight로 보정이 필요하다.

  CE ≈ 0.5~2.0,  Dice ≈ 0.3~0.8,  Boundary ≈ 0.1~0.5
  기본값: ce_weight=1.0, aux_weight=1.0 (스케일 비교 후 조정 권장)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reference:
  각 개별 loss 파일의 Reference 참조.
"""

import torch.nn as nn
from torch import Tensor

from .cross_entropy import CrossEntropyLoss
from .dice_loss     import DiceLoss
from .boundary_loss import BoundaryLoss


class CombinedLoss(nn.Module):
    """
    CE + Dice 또는 CE + Boundary 조합 Loss — E3, E4 실험용.

    forward()는 scalar Tensor를 반환한다.
    train.py의 기존 학습 루프와 완전히 호환.

    Args:
        mode         (str):   "ce+dice" | "ce+boundary". 실험 구성 선택.
        num_classes  (int):   클래스 수. DiceLoss에 필요.
        ignore_index (int):   무시할 class index. Default: 255.
        ce_weight    (float): CE loss 가중치. Default: 1.0.
        aux_weight   (float): 보조 loss(Dice|Boundary) 가중치. Default: 1.0.
    """

    VALID_MODES = ("ce+dice", "ce+boundary")

    def __init__(
        self,
        mode: str,
        num_classes: int,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        aux_weight: float = 1.0,
    ):
        super().__init__()

        assert mode in self.VALID_MODES, (
            f"mode must be one of {self.VALID_MODES}, got '{mode}'"
        )

        self.mode       = mode
        self.ce_weight  = ce_weight
        self.aux_weight = aux_weight

        # ── CE loss (공통) ────────────────────────────────────────────────────
        self.ce = CrossEntropyLoss(ignore_index=ignore_index)

        # ── 보조 loss (mode에 따라 선택) ──────────────────────────────────────
        if mode == "ce+dice":
            self.aux = DiceLoss(
                num_classes=num_classes,
                ignore_index=ignore_index,
            )
        elif mode == "ce+boundary":
            self.aux = BoundaryLoss(ignore_index=ignore_index)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits  : (B, num_classes, H, W)  float32 — raw logits
            targets : (B, H, W)               int64   — class index
        Returns:
            loss : scalar Tensor — backward() 대상
        """
        # ── 개별 loss 계산 ────────────────────────────────────────────────────
        ce_loss  = self.ce(logits, targets)
        aux_loss = self.aux(logits, targets)

        # ── 가중합 후 scalar 반환 ─────────────────────────────────────────────
        return self.ce_weight * ce_loss + self.aux_weight * aux_loss
"""
data/camvid.py

CamVid Dataset loader.

CamVid 디렉토리 구조 (실제 다운로드 기준):
    datasets/CamVid/
    ├── train/          *.png   (이미지)
    ├── train_labels/   *.png   (RGB palette mask, 이미지와 동일 파일명)
    ├── val/            *.png
    ├── val_labels/     *.png
    ├── test/           *.png
    └── test_labels/    *.png

CamVid 11 classes (void 제외):
    0  Sky
    1  Building
    2  Pole
    3  Road
    4  Pavement
    5  Tree
    6  SignSymbol
    7  Fence
    8  Car
    9  Pedestrian
    10 Bicyclist
    255 Void / ignore

RGB palette mask → class index 변환:
    CamVid의 label 이미지는 RGB 컬러로 저장되어 있다.
    각 pixel의 RGB 값을 CLASS_MAP으로 lookup해서 class index로 변환한다.

"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import numpy as np
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset


# ── class_dict.csv 기준 32개 원본 클래스 RGB값 ────────────────────────────────
# (R, G, B) → 32-class index
# 출처: datasets/CamVid/class_dict.csv (정확한 값)
_RAW_COLORMAP = {
    # name              R    G    B     32-class idx
    "Animal":        ( 64, 128,  64),  # 0
    "Archway":       (192,   0, 128),  # 1
    "Bicyclist":     (  0, 128, 192),  # 2
    "Bridge":        (  0, 128,  64),  # 3
    "Building":      (128,   0,   0),  # 4
    "Car":           ( 64,   0, 128),  # 5
    "CartLuggagePram":(64,   0, 192),  # 6
    "Child":         (192, 128,  64),  # 7
    "Column_Pole":   (192, 192, 128),  # 8
    "Fence":         ( 64,  64, 128),  # 9
    "LaneMkgsDriv":  (128,   0, 192),  # 10
    "LaneMkgsNonDriv":(192,  0,  64),  # 11
    "Misc_Text":     (128, 128,  64),  # 12
    "MotorcycleScooter":(192,0, 192),  # 13
    "OtherMoving":   (128,  64,  64),  # 14
    "ParkingBlock":  ( 64, 192, 128),  # 15
    "Pedestrian":    ( 64,  64,   0),  # 16
    "Road":          (128,  64, 128),  # 17
    "RoadShoulder":  (128, 128, 192),  # 18
    "Sidewalk":      (  0,   0, 192),  # 19  ← Pavement (실제 색상)
    "SignSymbol":    (192, 128, 128),  # 20
    "Sky":           (128, 128, 128),  # 21
    "SUVPickupTruck":( 64, 128, 192),  # 22
    "TrafficCone":   (  0,   0,  64),  # 23
    "TrafficLight":  (  0,  64,  64),  # 24
    "Train":         (192,  64, 128),  # 25
    "Tree":          (128, 128,   0),  # 26
    "Truck_Bus":     (192, 128, 192),  # 27
    "Tunnel":        ( 64,   0,  64),  # 28
    "VegetationMisc":(192, 192,   0),  # 29
    "Void":          (  0,   0,   0),  # → IGNORE
    "Wall":          ( 64, 192,   0),  # 30
}

# ── 32 → 11 클래스 매핑 (SegNet 논문 기준) ────────────────────────────────────
#
# 11-class index:
#   0  Sky
#   1  Building
#   2  Pole
#   3  Road
#   4  Pavement (Sidewalk)
#   5  Tree
#   6  SignSymbol
#   7  Fence
#   8  Car
#   9  Pedestrian
#  10  Bicyclist
# 255  Void / ignore
#
_NAME_TO_11CLASS = {
    "Sky":              0,
    "Building":         1,  "Wall": 1, "Archway": 1, "Tunnel": 1, "Bridge": 1,
    "Column_Pole":      2,  "TrafficLight": 2, "TrafficCone": 2,
    "Road":             3,  "LaneMkgsDriv": 3, "LaneMkgsNonDriv": 3, "RoadShoulder": 3,
    "Sidewalk":         4,  "ParkingBlock": 4,
    "Tree":             5,  "VegetationMisc": 5,
    "SignSymbol":       6,  "Misc_Text": 6,
    "Fence":            7,
    "Car":              8,  "SUVPickupTruck": 8, "Truck_Bus": 8, "Train": 8,
                            "OtherMoving": 8, "MotorcycleScooter": 8,
                            "CartLuggagePram": 8,
    "Pedestrian":       9,  "Child": 9, "Animal": 9,
    "Bicyclist":       10,
    "Void":           255,
}

NUM_CLASSES  = 11
IGNORE_INDEX = 255


def _build_color_lookup() -> dict:
    """
    RGB → 11-class index lookup table 생성.
    class_dict.csv의 정확한 RGB값 사용.
    """
    table = {}
    for name, rgb in _RAW_COLORMAP.items():
        cls11 = _NAME_TO_11CLASS.get(name, IGNORE_INDEX)
        table[rgb] = cls11
    return table


# 모듈 로드 시 1회만 생성
_COLOR_TABLE = _build_color_lookup()


def _rgb_mask_to_index(mask_rgb: np.ndarray) -> np.ndarray:
    """
    RGB mask (H, W, 3) → 11-class index mask (H, W), dtype=int64.

    JPEG 압축 등으로 인해 픽셀값에 ±수 정도의 오차가 있을 수 있으므로
    정확히 일치하는 색상이 없으면 L2 거리가 가장 가까운 색상으로 매핑한다.

    알 수 없는 색상(Void)은 IGNORE_INDEX(255)로 처리.
    """
    H, W, _ = mask_rgb.shape
    index_mask = np.full((H, W), IGNORE_INDEX, dtype=np.int64)

    # ── Step 1: 정확히 일치하는 픽셀 먼저 처리 (빠름) ───────────────────────
    for rgb, cls_idx in _COLOR_TABLE.items():
        match = (
            (mask_rgb[:, :, 0] == rgb[0]) &
            (mask_rgb[:, :, 1] == rgb[1]) &
            (mask_rgb[:, :, 2] == rgb[2])
        )
        index_mask[match] = cls_idx

    # ── Step 2: 미매핑 픽셀 → nearest color 매핑 (JPEG 노이즈 처리) ─────────
    unmatched = index_mask == IGNORE_INDEX

    # Void(0,0,0) 픽셀은 실제 void이므로 제외
    void_mask = (
        (mask_rgb[:, :, 0] == 0) &
        (mask_rgb[:, :, 1] == 0) &
        (mask_rgb[:, :, 2] == 0)
    )
    unmatched = unmatched & ~void_mask

    if unmatched.any():
        # 팔레트 색상 배열: (num_colors, 3)
        palette_rgb = np.array(list(_COLOR_TABLE.keys()), dtype=np.int32)   # (C, 3)
        palette_cls = np.array(list(_COLOR_TABLE.values()), dtype=np.int64) # (C,)

        # 미매핑 픽셀 추출: (N, 3)
        unmatched_pixels = mask_rgb[unmatched].astype(np.int32)  # (N, 3)

        # L2 거리 계산: (N, C)
        diff = unmatched_pixels[:, None, :] - palette_rgb[None, :, :]  # (N, C, 3)
        dist = (diff ** 2).sum(axis=2)                                  # (N, C)

        # 가장 가까운 색상 index
        nearest = dist.argmin(axis=1)                                   # (N,)
        index_mask[unmatched] = palette_cls[nearest]

    return index_mask  # (H, W), values ∈ {0..10, 255}


class CamVidDataset(Dataset):
    """
    CamVid Semantic Segmentation Dataset.

    Args:
        image_dir  (str): 이미지 폴더 경로 (e.g., "camvid/images/train")
        label_dir  (str): 라벨 폴더 경로 (e.g., "camvid/labels/train")
        transforms (callable, optional): data/transforms.py의 SegTransform 인스턴스.
                   None이면 PIL Image 그대로 반환하지 않고 기본 ToTensor 적용.
        img_suffix  (str): 이미지 파일 확장자. Default: ".png"
        mask_suffix (str): 마스크 파일 확장자. Default: ".png"

    Returns (per __getitem__):
        image : Tensor (3, H, W)  float32, [0,1] 또는 normalized
        mask  : Tensor (H, W)     int64,   values ∈ {0..10, 255}
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        transforms: Optional[Callable] = None,
        img_suffix: str = ".png",
        mask_suffix: str = ".png",
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transforms = transforms
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix

        # 이미지 파일 목록 수집 및 정렬 (재현성을 위해 정렬)
        self.image_paths = sorted(
            self.image_dir.glob(f"*{img_suffix}")
        )

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {image_dir} with suffix {img_suffix}"
            )

        # 대응하는 mask 경로 생성
        # 실제 CamVid 다운로드 구조:
        #   train/0001TP_009210.png  →  train_labels/0001TP_009210_L.png
        # 라벨이 없는 이미지(공백·괄호 포함 파일 등)는 조용히 스킵한다.
        self.mask_paths = []
        skipped = []
        for img_path in self.image_paths:
            mask_name = img_path.stem + "_L" + mask_suffix
            mask_path = self.label_dir / mask_name
            if mask_path.exists():
                self.mask_paths.append(mask_path)
            else:
                skipped.append(img_path.name)

        # 라벨 없는 이미지는 image_paths에서도 제거 (1:1 대응 유지)
        if skipped:
            print(f"[CamVidDataset] {len(skipped)}개 이미지 스킵 (라벨 없음): {skipped[:3]}{'...' if len(skipped) > 3 else ''}")
            self.image_paths = [
                p for p in self.image_paths
                if (self.label_dir / (p.stem + "_L" + mask_suffix)).exists()
            ]

        assert len(self.image_paths) == len(self.mask_paths), \
            "image / mask 수가 일치하지 않습니다."

        assert len(self.image_paths) == len(self.mask_paths), \
            "image / mask 수가 일치하지 않습니다."

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            image : Tensor (3, H, W)  float32
            mask  : Tensor (H, W)     int64
        """
        # ── Load image ────────────────────────────────────────────────────────
        image = Image.open(self.image_paths[idx]).convert("RGB")

        # ── Load mask ─────────────────────────────────────────────────────────
        mask_pil = Image.open(self.mask_paths[idx]).convert("RGB")
        # RGB PIL Image (H, W, 3) → numpy
        mask_rgb = np.array(mask_pil, dtype=np.uint8)    # (H, W, 3)
        # RGB → class index (H, W), int64
        mask_np  = _rgb_mask_to_index(mask_rgb)          # (H, W)

        # PIL Image로 다시 감싸서 transforms에서 resize 가능하게 함
        # dtype 보존을 위해 int32 PIL mode "I" 사용
        mask_pil_idx = Image.fromarray(mask_np.astype(np.int32), mode="I")

        # ── Apply transforms ──────────────────────────────────────────────────
        if self.transforms is not None:
            image, mask_pil_idx = self.transforms(image, mask_pil_idx)
            # transforms가 Tensor로 변환해서 반환하는 경우
            if isinstance(image, Tensor):
                mask = mask_pil_idx
                if not isinstance(mask, Tensor):
                    mask = torch.from_numpy(
                        np.array(mask_pil_idx, dtype=np.int64)
                    )
                return image, mask  # (3,H,W), (H,W)

        # transforms 없을 때: 기본 변환
        # image: PIL → float32 Tensor (3, H, W), [0, 1]
        image_np = np.array(image, dtype=np.float32) / 255.0  # (H, W, 3)
        image_tensor = torch.from_numpy(
            image_np.transpose(2, 0, 1)                        # (3, H, W)
        )
        mask_tensor = torch.from_numpy(
            np.array(mask_pil_idx, dtype=np.int64)             # (H, W)
        )

        return image_tensor, mask_tensor  # (3,H,W) float32, (H,W) int64

    def get_class_names(self) -> List[str]:
        return [
            "Sky", "Building", "Pole", "Road", "Pavement",
            "Tree", "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist",
        ]
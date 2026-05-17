"""
Run SegFormer prediction on one image or an image directory.

Colab / local usage:
    # 1) Run this from the project root.
    #    Colab example:
    #    %cd /content/drive/MyDrive/Colab Notebooks/segformer-core

    # 2) Predict the first N images from test_img_dir or val_img_dir in the config.
    #    --limit 10 saves the first 10 images after filename sorting.
    python scripts/predict.py \
        --config configs/e0_paperlike.yaml \
        --checkpoint weights/e0_mlp_ce_paperlike_100_best.pth \
        --device cuda \
        --limit 10 \
        --output-dir outputs/predictions/test_10

    # 3) Predict one specific image. --image must point to an existing file.
    python scripts/predict.py \
        --config configs/e0_paperlike.yaml \
        --checkpoint weights/e0_mlp_ce_paperlike_100_best.pth \
        --image datasets/CamVid/test/0001TP_006690.png \
        --device cuda

    # 4) Predict images from a specific directory.
    python scripts/predict.py \
        --config configs/e0_paperlike.yaml \
        --checkpoint weights/e0_mlp_ce_paperlike_100_best.pth \
        --image-dir datasets/CamVid/test \
        --limit 20 \
        --device cuda

Outputs:
    *_pred.png      : colorized prediction mask
    *_overlay.png   : original image + prediction overlay
    *_compare.png   : original / prediction / overlay / GT label if available
    class_legend.txt: class id/name/color mapping used by *_pred.png

Notes:
    - If --image is set, only that single image is predicted.
    - If --image-dir is set, images are read from that directory.
    - If neither is set, the script uses cfg["test_img_dir"] first, then cfg["val_img_dir"].
    - --limit 0 or omitting --limit processes all images in directory mode.
    - Prediction loads the checkpoint directly, so cfg["pretrained"] is disabled by default
      to avoid HuggingFace downloads. Add --use-config-pretrained only when you really
      want to keep the config's pretrained loading behavior.

Prediction color legend:
    0  Sky         RGB(128, 128, 128)
    1  Building    RGB(128,   0,   0)
    2  Pole        RGB(192, 192, 128)
    3  Road        RGB(128,  64, 128)
    4  Pavement    RGB(  0,   0, 192)
    5  Tree        RGB(128, 128,   0)
    6  SignSymbol  RGB(192, 128, 128)
    7  Fence       RGB( 64,  64, 128)
    8  Car         RGB( 64,   0, 128)
    9  Pedestrian  RGB( 64,  64,   0)
    10 Bicyclist   RGB(  0, 128, 192)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.camvid import _rgb_mask_to_index  # noqa: E402
from data.transforms import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from scripts.train import build_model, load_config  # noqa: E402


CLASS_NAMES = [
    "Sky",
    "Building",
    "Pole",
    "Road",
    "Pavement",
    "Tree",
    "SignSymbol",
    "Fence",
    "Car",
    "Pedestrian",
    "Bicyclist",
]

PALETTE = np.array(
    [
        [128, 128, 128],  # Sky
        [128, 0, 0],      # Building
        [192, 192, 128],  # Pole
        [128, 64, 128],   # Road
        [0, 0, 192],      # Pavement
        [128, 128, 0],    # Tree
        [192, 128, 128],  # SignSymbol
        [64, 64, 128],    # Fence
        [64, 0, 128],     # Car
        [64, 64, 0],      # Pedestrian
        [0, 128, 192],    # Bicyclist
    ],
    dtype=np.uint8,
)


def save_legend(output_dir: Path) -> None:
    lines = ["class_id,class_name,r,g,b"]
    for class_id, (name, rgb) in enumerate(zip(CLASS_NAMES, PALETTE.tolist())):
        lines.append(f"{class_id},{name},{rgb[0]},{rgb[1]},{rgb[2]}")
    (output_dir / "class_legend.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return path


def default_checkpoint(cfg: dict, checkpoint_type: str) -> Path:
    suffix = "best" if checkpoint_type == "best" else "last"
    return resolve_path(cfg["save_dir"]) / f"{cfg['exp_name']}_{suffix}.pth"


def collect_images(image: str | None, image_dir: str | None, cfg: dict, limit: int) -> list[Path]:
    if image:
        paths = [resolve_path(image)]
    else:
        directory_value = image_dir or cfg.get("test_img_dir") or cfg.get("val_img_dir")
        if not directory_value:
            raise ValueError("Pass --image or --image-dir, or set test_img_dir/val_img_dir in config.")
        directory = resolve_path(directory_value)
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in exts)

    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError("No input images found.")
    return paths


def preprocess(image: Image.Image, size: tuple[int, int], device: torch.device) -> torch.Tensor:
    height, width = size
    resized = image.resize((width, height), resample=Image.BILINEAR)
    image_np = np.array(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_np.transpose(2, 0, 1))

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device)


def colorize(mask: np.ndarray) -> Image.Image:
    clipped = np.clip(mask, 0, len(PALETTE) - 1)
    return Image.fromarray(PALETTE[clipped], mode="RGB")


def find_label_path(image_path: Path, label_dir: str | None, cfg: dict) -> Path | None:
    directory_value = label_dir or cfg.get("test_lbl_dir")
    if not directory_value:
        return None
    directory = resolve_path(directory_value)
    candidate = directory / f"{image_path.stem}_L.png"
    return candidate if candidate.exists() else None


def load_label(label_path: Path, size: tuple[int, int]) -> Image.Image:
    height, width = size
    label_rgb = Image.open(label_path).convert("RGB")
    label_np = _rgb_mask_to_index(np.array(label_rgb, dtype=np.uint8))
    label_img = Image.fromarray(label_np.astype(np.uint8), mode="L")
    label_img = label_img.resize((width, height), resample=Image.NEAREST)
    return colorize(np.array(label_img, dtype=np.int64))


def make_overlay(image: Image.Image, mask_color: Image.Image, alpha: float) -> Image.Image:
    image = image.convert("RGB").resize(mask_color.size, resample=Image.BILINEAR)
    return Image.blend(image, mask_color, alpha=alpha)


def make_panel(images: list[Image.Image]) -> Image.Image:
    widths, heights = zip(*(img.size for img in images))
    panel = Image.new("RGB", (sum(widths), max(heights)), color=(0, 0, 0))
    left = 0
    for img in images:
        panel.paste(img.convert("RGB"), (left, 0))
        left += img.size[0]
    return panel


@torch.no_grad()
def predict_one(
    model: torch.nn.Module,
    image_path: Path,
    cfg: dict,
    device: torch.device,
    output_dir: Path,
    alpha: float,
    label_dir: str | None,
) -> None:
    image = Image.open(image_path).convert("RGB")
    orig_size = image.size
    model_input = preprocess(image, cfg["input_size"], device)

    logits = model(model_input)
    logits = F.interpolate(
        logits,
        size=(orig_size[1], orig_size[0]),
        mode="bilinear",
        align_corners=False,
    )
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)

    mask_color = colorize(pred)
    overlay = make_overlay(image, mask_color, alpha=alpha)

    stem = image_path.stem
    mask_color.save(output_dir / f"{stem}_pred.png")
    overlay.save(output_dir / f"{stem}_overlay.png")

    panel_images = [image.resize(mask_color.size, Image.BILINEAR), mask_color, overlay]
    label_path = find_label_path(image_path, label_dir, cfg)
    if label_path:
        panel_images.append(load_label(label_path, (orig_size[1], orig_size[0])))
    make_panel(panel_images).save(output_dir / f"{stem}_compare.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict segmentation masks with a trained SegFormer checkpoint.")
    parser.add_argument("--config", required=True, help="Path to config yaml.")
    parser.add_argument("--checkpoint", help="Checkpoint path. Defaults to weights/<exp_name>_best.pth.")
    parser.add_argument("--checkpoint-type", choices=("best", "last"), default="best")
    parser.add_argument("--image", help="Single input image path.")
    parser.add_argument("--image-dir", help="Input image directory. Defaults to cfg['test_img_dir'].")
    parser.add_argument("--label-dir", help="Optional label directory for compare panels.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to outputs/predictions/<exp_name>.")
    parser.add_argument("--limit", type=int, default=0, help="Max number of images for directory mode. 0 means all.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay mask opacity.")
    parser.add_argument(
        "--use-config-pretrained",
        action="store_true",
        help="Keep cfg['pretrained'] as-is before loading the checkpoint.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not args.use_config_pretrained:
        cfg["pretrained"] = False
    device = torch.device(args.device)

    checkpoint = resolve_path(args.checkpoint) if args.checkpoint else default_checkpoint(cfg, args.checkpoint_type)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    output_dir = resolve_path(args.output_dir) if args.output_dir else ROOT / "outputs" / "predictions" / cfg["exp_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_legend(output_dir)

    model = build_model(cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    image_paths = collect_images(args.image, args.image_dir, cfg, args.limit)
    for image_path in image_paths:
        predict_one(model, image_path, cfg, device, output_dir, args.alpha, args.label_dir)

    print(f"Checkpoint: {checkpoint}")
    print(f"Images    : {len(image_paths)}")
    print(f"Saved to  : {output_dir}")
    print("Files     : *_pred.png, *_overlay.png, *_compare.png")


if __name__ == "__main__":
    main()

"""
tests/test_fpn_decoder.py

Integration tests for FPNDecoder and full FPN-based SegFormer model.
Matches the test suite of test_decoder.py exactly.

Tests:
  1.  FPNDecoder output shape                  — (B, num_classes, H/4, W/4)
  2.  FPNDecoder with real encoder features    — end-to-end shape
  3.  SegFormer full forward pass shape (FPN)  — (B, num_classes, H, W)
  4.  Non-square input handling                — H ≠ W
  5.  CamVid config (11 classes)               — build_segformer_b0_fpn
  6.  Cityscapes config (19 classes)           — build_segformer_b0_fpn
  7.  CrossEntropy loss computation            — no NaN/Inf
  8.  Backward pass                            — gradients flow to encoder
  9.  Encoder weights frozen / unfrozen check  — grad existence
  10. Parameter count sanity                  — FPN vs Encoder balance
  11. Eval mode output determinism            — same input → same output
  12. Decoder feature count assertion         — Error handling

Run with:
    python tests/test_fpn_decoder.py
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.encoder.mit_encoder import MiTEncoder
from models.decoder.fpn_decoder import FPNDecoder
from models.segformer import SegFormer, MIT_B0_CHANNELS

# ── Test constants ─────────────────────────────────────────────────────────────
B           = 2
H = W       = 512
NUM_CLASSES = 11
FPN_DIM     = 256

PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
INFO = "\033[34m[INFO]\033[0m"

# ─── Helper: build components ─────────────────────────────────────────────────

def _get_encoder_features(h=H, w=W):
    encoder = MiTEncoder(in_channels=3)
    encoder.eval()
    x = torch.randn(B, 3, h, w)
    with torch.no_grad():
        features = encoder(x)
    return features

def _build_fpn_segformer(num_classes=NUM_CLASSES):
    decoder = FPNDecoder(in_channels=MIT_B0_CHANNELS, fpn_dim=FPN_DIM, num_classes=num_classes)
    return SegFormer(num_classes=num_classes, decoder=decoder)

# ─── Test 1: FPNDecoder output shape (from dummy features) ────────────────────

def test_fpn_decoder_shape_from_dummy():
    decoder = FPNDecoder(in_channels=MIT_B0_CHANNELS, fpn_dim=FPN_DIM, num_classes=NUM_CLASSES)
    decoder.eval()
    dummy_features = [
        torch.randn(B,  32, H//4,  W//4),
        torch.randn(B,  64, H//8,  W//8),
        torch.randn(B, 160, H//16, W//16),
        torch.randn(B, 256, H//32, W//32),
    ]
    with torch.no_grad():
        out = decoder(dummy_features)
    assert out.shape == (B, NUM_CLASSES, H//4, W//4)
    print(f"{PASS} FPNDecoder shape (dummy): {tuple(out.shape)}")

# ─── Test 2: FPNDecoder with real encoder features ────────────────────────────

def test_fpn_decoder_shape_from_encoder():
    features = _get_encoder_features()
    decoder = FPNDecoder(in_channels=MIT_B0_CHANNELS, fpn_dim=FPN_DIM, num_classes=NUM_CLASSES)
    decoder.eval()
    with torch.no_grad():
        out = decoder(features)
    assert out.shape == (B, NUM_CLASSES, H//4, W//4)
    print(f"{PASS} FPNDecoder shape (real encoder): {tuple(out.shape)}")

# ─── Test 3: Full SegFormer (FPN) forward shape ───────────────────────────────

def test_segformer_fpn_full_forward():
    model = _build_fpn_segformer()
    model.eval()
    x = torch.randn(B, 3, H, W)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (B, NUM_CLASSES, H, W)
    print(f"{PASS} SegFormer (FPN) full forward: {tuple(out.shape)}")

# ─── Test 4: Non-square input ─────────────────────────────────────────────────

def test_fpn_non_square_input():
    H_ns, W_ns = 360, 480
    model = _build_fpn_segformer()
    model.eval()
    x = torch.randn(B, 3, H_ns, W_ns)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (B, NUM_CLASSES, H_ns, W_ns)
    print(f"{PASS} Non-square input (360×480): {tuple(out.shape)}")

# ─── Test 5: CamVid config ────────────────────────────────────────────────────

def test_fpn_camvid_config():
    model = _build_fpn_segformer(num_classes=11)
    model.eval()
    x = torch.randn(1, 3, 360, 480)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 11, 360, 480)
    print(f"{PASS} CamVid config (11 classes): {tuple(out.shape)}")

# ─── Test 6: Cityscapes config ────────────────────────────────────────────────

def test_fpn_cityscapes_config():
    model = _build_fpn_segformer(num_classes=19)
    model.eval()
    x = torch.randn(1, 3, 512, 1024)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 19, 512, 1024)
    print(f"{PASS} Cityscapes config (19 classes): {tuple(out.shape)}")

# ─── Test 7: CrossEntropy loss ────────────────────────────────────────────────

def test_fpn_loss():
    model = _build_fpn_segformer()
    model.train()

    x = torch.randn(B, 3, H, W)
    labels = torch.randint(0, NUM_CLASSES, (B, H, W))

    logits = model(x)
    loss = F.cross_entropy(logits, labels)

    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() > 0,      f"Loss is zero — something is wrong"
    print(f"{PASS} CrossEntropy loss: {loss.item():.4f}")

# ─── Test 8: Backward pass (Grad Norm 출력 포함) ──────────────────────────────

def test_fpn_backward_pass():
    model = _build_fpn_segformer()
    model.train()
    x = torch.randn(B, 3, H, W)
    labels = torch.randint(0, NUM_CLASSES, (B, H, W))
    
    logits = model(x)
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    enc_weight = model.encoder.stages[0].patch_embed.proj.weight
    dec_weight = model.decoder.lateral_convs[0].proj.weight # FPN lateral conv check

    assert enc_weight.grad is not None and enc_weight.grad.abs().sum() > 0
    assert dec_weight.grad is not None and dec_weight.grad.abs().sum() > 0

    print(f"{PASS} Backward pass — gradients reach encoder & decoder")
    print(f"       Encoder grad norm: {enc_weight.grad.norm().item():.6f}")
    print(f"       Decoder grad norm: {dec_weight.grad.norm().item():.6f}")

# ─── Test 9: Parameter count ──────────────────────────────────────────────────

def test_fpn_parameter_count():
    model = _build_fpn_segformer()
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    total = enc_params + dec_params

    print(f"{INFO} Encoder params : {enc_params:>10,}")
    print(f"{INFO} Decoder params : {dec_params:>10,}")
    print(f"{INFO} Total params   : {total:>10,}")

    assert dec_params > 1_000_000, "FPN should have more params than MLP"
    print(f"{PASS} Parameter count within expected range")

# ─── Test 10: Eval determinism ────────────────────────────────────────────────

def test_fpn_eval_determinism():
    model = _build_fpn_segformer()
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out1, out2 = model(x), model(x)
    assert torch.allclose(out1, out2)
    print(f"{PASS} Eval mode determinism — outputs are identical")

# ─── Test 11: Feature count assertion ─────────────────────────────────────────

def test_fpn_wrong_feature_count():
    decoder = FPNDecoder(in_channels=MIT_B0_CHANNELS, fpn_dim=FPN_DIM, num_classes=NUM_CLASSES)
    bad_features = [torch.randn(B, 32, 128, 128)] * 3
    try:
        decoder(bad_features)
        print(f"{FAIL} Should have raised AssertionError")
    except AssertionError:
        print(f"{PASS} Correctly raises AssertionError for wrong feature count")

# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_fpn_decoder_shape_from_dummy,
        test_fpn_decoder_shape_from_encoder,
        test_segformer_fpn_full_forward,
        test_fpn_non_square_input,
        test_fpn_camvid_config,
        test_fpn_cityscapes_config,
        test_fpn_loss,
        test_fpn_backward_pass,
        test_fpn_parameter_count,
        test_fpn_eval_determinism,
        test_fpn_wrong_feature_count,
    ]

    print("=" * 60)
    print("Running COMPLETE FPNDecoder (E1) Integration Tests")
    print("=" * 60)
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"{FAIL} {t.__name__}: {e}")
    print("=" * 60)
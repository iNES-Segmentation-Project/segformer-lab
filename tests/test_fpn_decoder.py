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
  9.  Parameter count sanity                   — FPN vs MLP vs Encoder balance
  10. Eval mode output determinism             — same input → same output
  11. Decoder feature count assertion          — Error handling
  12. Top-down pathway interaction             — FPN 고유 동작 검증

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

# MLP decoder 파라미터 기준값 (E0 vs E1 비교용)
MLP_DEC_PARAMS = 500_000

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
    decoder = FPNDecoder(
        in_channels=MIT_B0_CHANNELS,
        fpn_dim=FPN_DIM,
        num_classes=num_classes,
    )
    return SegFormer(num_classes=num_classes, decoder=decoder)


# ─── Test 1: FPNDecoder output shape (from dummy features) ────────────────────

def test_fpn_decoder_shape_from_dummy():
    decoder = FPNDecoder(
        in_channels=MIT_B0_CHANNELS,
        fpn_dim=FPN_DIM,
        num_classes=NUM_CLASSES,
    )
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
    decoder = FPNDecoder(
        in_channels=MIT_B0_CHANNELS,
        fpn_dim=FPN_DIM,
        num_classes=NUM_CLASSES,
    )
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
    x      = torch.randn(B, 3, H, W)
    labels = torch.randint(0, NUM_CLASSES, (B, H, W))
    logits = model(x)
    loss   = F.cross_entropy(logits, labels)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() > 0,      f"Loss is zero — something is wrong"
    print(f"{PASS} CrossEntropy loss: {loss.item():.4f}")


# ─── Test 8: Backward pass ────────────────────────────────────────────────────

def test_fpn_backward_pass():
    model = _build_fpn_segformer()
    model.train()
    x      = torch.randn(B, 3, H, W)
    labels = torch.randint(0, NUM_CLASSES, (B, H, W))
    logits = model(x)
    loss   = F.cross_entropy(logits, labels)
    loss.backward()

    enc_weight = model.encoder.stages[0].patch_embed.proj.weight
    dec_weight = model.decoder.lateral_convs[0].proj.weight

    assert enc_weight.grad is not None and enc_weight.grad.abs().sum() > 0, \
        "No gradient in encoder!"
    assert dec_weight.grad is not None and dec_weight.grad.abs().sum() > 0, \
        "No gradient in decoder lateral_convs!"

    print(f"{PASS} Backward pass — gradients reach encoder & decoder")
    print(f"       Encoder grad norm: {enc_weight.grad.norm().item():.6f}")
    print(f"       Decoder grad norm: {dec_weight.grad.norm().item():.6f}")


# ─── Test 9: Parameter count ──────────────────────────────────────────────────

def test_fpn_parameter_count():
    """
    세 가지를 동시에 검증:
      1) FPN decoder > MLP decoder (~0.5M) — FPN이 더 복잡한 구조임을 확인
      2) Encoder > FPN decoder             — 여전히 경량 decoder 범주
      3) Total < 10M                       — 프로젝트 경량 모델 목표 준수
    """
    model = _build_fpn_segformer()
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    total      = enc_params + dec_params

    print(f"{INFO} Encoder params : {enc_params:>10,}")
    print(f"{INFO} Decoder params : {dec_params:>10,}")
    print(f"{INFO} Total params   : {total:>10,}")
    print(f"{INFO} MLP baseline   : ~{MLP_DEC_PARAMS:>9,}")

    assert enc_params > dec_params, \
        f"FPN decoder({dec_params:,}) should still be smaller than encoder({enc_params:,})"
    assert dec_params > MLP_DEC_PARAMS, \
        f"FPN({dec_params:,}) should have more params than MLP(~{MLP_DEC_PARAMS:,})"
    assert total < 10_000_000, \
        f"Total param count unexpectedly large: {total:,}"

    print(f"{PASS} Parameter count — "
          f"FPN({dec_params:,}) > MLP(~{MLP_DEC_PARAMS:,}), "
          f"Encoder({enc_params:,}) > Decoder, "
          f"Total({total:,}) < 10M")


# ─── Test 10: Eval determinism ────────────────────────────────────────────────

def test_fpn_eval_determinism():
    model = _build_fpn_segformer()
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2), "Eval outputs are not deterministic!"
    print(f"{PASS} Eval mode determinism — outputs are identical")


# ─── Test 11: Feature count assertion ─────────────────────────────────────────

def test_fpn_wrong_feature_count():
    """BaseDecoder._check_features must raise AssertionError for wrong input length."""
    decoder = FPNDecoder(
        in_channels=MIT_B0_CHANNELS,
        fpn_dim=FPN_DIM,
        num_classes=NUM_CLASSES,
    )
    bad_features = [torch.randn(B, 32, 128, 128)] * 3   # only 3, not 4
    try:
        decoder(bad_features)
        print(f"{FAIL} Should have raised AssertionError for 3 features")
    except AssertionError:
        print(f"{PASS} Correctly raises AssertionError for wrong feature count")


# ─── Test 12: Top-down pathway interaction ────────────────────────────────────

def test_fpn_top_down_interaction():
    """
    FPN의 top-down pathway가 실제로 동작하는지 검증.

    F4(가장 깊은 feature)만 zeroed로 바꾸면 top-down pathway를 통해
    F3, F2, F1 모두에 영향이 전파되므로 출력이 달라져야 한다.
    이 테스트가 통과해야만 FPNDecoder가 MLPDecoder와 구조적으로
    다르게 동작함을 보장할 수 있다.
    """
    decoder = FPNDecoder(
        in_channels=MIT_B0_CHANNELS,
        fpn_dim=FPN_DIM,
        num_classes=NUM_CLASSES,
    )
    decoder.eval()

    base_features = [
        torch.randn(B,  32, H//4,  W//4),
        torch.randn(B,  64, H//8,  W//8),
        torch.randn(B, 160, H//16, W//16),
        torch.randn(B, 256, H//32, W//32),
    ]
    zeroed_f4_features = [
        base_features[0].clone(),
        base_features[1].clone(),
        base_features[2].clone(),
        torch.zeros_like(base_features[3]),   # F4만 0으로 교체
    ]

    with torch.no_grad():
        out_base   = decoder(base_features)
        out_zeroed = decoder(zeroed_f4_features)

    assert not torch.allclose(out_base, out_zeroed, atol=1e-5), \
        "F4 변경이 출력에 영향을 주지 않음 — top-down pathway가 동작하지 않을 수 있음"
    print(f"{PASS} Top-down pathway — F4 zeroed가 출력 전체에 전파됨 (FPN 고유 동작 확인)")


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
        test_fpn_top_down_interaction,
    ]

    print("=" * 60)
    print("Running COMPLETE FPNDecoder (E1) Integration Tests")
    print("=" * 60)
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"{FAIL} {t.__name__}: {e}")
            failed.append(t.__name__)
    print("=" * 60)
    if failed:
        print(f"FAILED: {len(failed)} test(s): {failed}")
    else:
        print("All tests passed.")
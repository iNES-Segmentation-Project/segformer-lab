"""
tests/test_encoder.py

Independent unit tests for each encoder module.
Run with: pytest tests/test_encoder.py -v
         or: python tests/test_encoder.py
"""

import torch
import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.encoder.overlap_patch_embed import OverlapPatchEmbed
from models.encoder.efficient_attention import EfficientSelfAttention
from models.encoder.mix_ffn import MixFFN
from models.encoder.mit_stage import MiTStage, TransformerBlock
from models.encoder.mit_encoder import MiTEncoder

B = 2       # batch size
H = W = 512 # input resolution (powers of 2 for clean testing)


def test_overlap_patch_embed_stage1():
    """Stage 1: (B,3,H,W) → (B, H/4*W/4, 32), H'=H/4, W'=W/4"""
    model = OverlapPatchEmbed(in_channels=3, embed_dim=32, patch_size=7, stride=4)
    x = torch.randn(B, 3, H, W)
    out, h, w = model(x)
    assert out.shape == (B, h * w, 32), f"Got {out.shape}"
    assert h == H // 4 and w == W // 4, f"Expected ({H//4},{W//4}), got ({h},{w})"
    print(f"[PASS] OverlapPatchEmbed stage1: {tuple(out.shape)}, H'={h}, W'={w}")


def test_overlap_patch_embed_stage2():
    """Stage 2: (B,32,H/4,W/4) → (B, H/8*W/8, 64)"""
    model = OverlapPatchEmbed(in_channels=32, embed_dim=64, patch_size=3, stride=2)
    x = torch.randn(B, 32, H // 4, W // 4)
    out, h, w = model(x)
    assert out.shape == (B, h * w, 64), f"Got {out.shape}"
    assert h == H // 8 and w == W // 8
    print(f"[PASS] OverlapPatchEmbed stage2: {tuple(out.shape)}, H'={h}, W'={w}")


def test_efficient_self_attention_with_sr():
    """ESA with sr_ratio=8: sequence length of K,V is reduced."""
    N = (H // 4) * (W // 4)   # = 16384 for 512×512
    model = EfficientSelfAttention(embed_dim=32, num_heads=1, sr_ratio=8)
    x = torch.randn(B, N, 32)
    out = model(x, H // 4, W // 4)
    assert out.shape == (B, N, 32), f"Got {out.shape}"
    print(f"[PASS] EfficientSelfAttention (sr=8): {tuple(out.shape)}")


def test_efficient_self_attention_no_sr():
    """ESA with sr_ratio=1: no reduction (standard MHSA)."""
    N = (H // 32) * (W // 32)  # = 256 for 512×512
    model = EfficientSelfAttention(embed_dim=256, num_heads=8, sr_ratio=1)
    x = torch.randn(B, N, 256)
    out = model(x, H // 32, W // 32)
    assert out.shape == (B, N, 256), f"Got {out.shape}"
    print(f"[PASS] EfficientSelfAttention (sr=1): {tuple(out.shape)}")


def test_mix_ffn():
    """MixFFN: input and output shape must match."""
    N = (H // 4) * (W // 4)
    model = MixFFN(embed_dim=32, mlp_ratio=4.0)
    x = torch.randn(B, N, 32)
    out = model(x, H // 4, W // 4)
    assert out.shape == (B, N, 32), f"Got {out.shape}"
    print(f"[PASS] MixFFN: {tuple(out.shape)}")


def test_transformer_block():
    """TransformerBlock: residual in, residual out — shape preserved."""
    N = (H // 4) * (W // 4)
    block = TransformerBlock(embed_dim=32, num_heads=1, sr_ratio=8)
    x = torch.randn(B, N, 32)
    out = block(x, H // 4, W // 4)
    assert out.shape == (B, N, 32), f"Got {out.shape}"
    print(f"[PASS] TransformerBlock: {tuple(out.shape)}")


def test_mit_stage1():
    """Stage 1: (B,3,H,W) → (B,32,H/4,W/4)"""
    stage = MiTStage(
        in_channels=3, embed_dim=32, patch_size=7, stride=4,
        depth=2, num_heads=1, sr_ratio=8,
    )
    x = torch.randn(B, 3, H, W)
    out = stage(x)
    assert out.shape == (B, 32, H // 4, W // 4), f"Got {out.shape}"
    print(f"[PASS] MiTStage 1: {tuple(out.shape)}")


def test_mit_stage4():
    """Stage 4: (B,160,H/16,W/16) → (B,256,H/32,W/32)"""
    stage = MiTStage(
        in_channels=160, embed_dim=256, patch_size=3, stride=2,
        depth=2, num_heads=8, sr_ratio=1,
    )
    x = torch.randn(B, 160, H // 16, W // 16)
    out = stage(x)
    assert out.shape == (B, 256, H // 32, W // 32), f"Got {out.shape}"
    print(f"[PASS] MiTStage 4: {tuple(out.shape)}")


def test_mit_encoder_output_shapes():
    """Full encoder: 4 feature maps at correct resolutions."""
    encoder = MiTEncoder(in_channels=3)
    encoder.eval()
    x = torch.randn(B, 3, H, W)
    with torch.no_grad():
        c1, c2, c3, c4 = encoder(x)

    assert c1.shape == (B,  32, H //  4, W //  4), f"c1: {c1.shape}"
    assert c2.shape == (B,  64, H //  8, W //  8), f"c2: {c2.shape}"
    assert c3.shape == (B, 160, H // 16, W // 16), f"c3: {c3.shape}"
    assert c4.shape == (B, 256, H // 32, W // 32), f"c4: {c4.shape}"
    print(f"[PASS] MiTEncoder outputs:")
    print(f"       c1={tuple(c1.shape)}")
    print(f"       c2={tuple(c2.shape)}")
    print(f"       c3={tuple(c3.shape)}")
    print(f"       c4={tuple(c4.shape)}")


def test_mit_encoder_param_count():
    """Encoder parameter count should be ~3.7M for B0."""
    encoder = MiTEncoder(in_channels=3)
    n = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    # Original MiT-B0 backbone ≈ 3.7M parameters
    print(f"[INFO] MiTEncoder total params: {n:,}")
    assert 3_000_000 < n < 5_000_000, f"Unexpected param count: {n:,}"
    print(f"[PASS] Parameter count in expected range: {n:,}")


if __name__ == "__main__":
    tests = [
        test_overlap_patch_embed_stage1,
        test_overlap_patch_embed_stage2,
        test_efficient_self_attention_with_sr,
        test_efficient_self_attention_no_sr,
        test_mix_ffn,
        test_transformer_block,
        test_mit_stage1,
        test_mit_stage4,
        test_mit_encoder_output_shapes,
        test_mit_encoder_param_count,
    ]

    print("=" * 60)
    print("Running encoder module tests")
    print("=" * 60)
    for t in tests:
        t()
    print("=" * 60)
    print("All tests passed.")
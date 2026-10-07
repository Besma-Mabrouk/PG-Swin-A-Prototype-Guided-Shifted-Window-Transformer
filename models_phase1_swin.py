# -*- coding: utf-8 -*-
"""
models_phase1_swin.py — Phase 1 backbone with an optional Swin Transformer
at the bottleneck.

This file does NOT modify models_phase1.py. It imports HybridResUNet3D and
subclasses it, adding a Swin-bottleneck gate as an opt-in module.

Design:
    - The bottleneck features have 8*init_features = 128 channels at 1/8
      resolution of the input.
    - The Swin block operates on 3D windows of size (2, 4, 4) with 4 heads.
    - The output is gated: y = x + alpha * gate * swin(x), where alpha
      starts at 0 (so the module is an identity at initialization) and gate
      is a learned sigmoid. This prevents the Swin block from destroying
      features while it warms up.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_phase1 import HybridResUNet3D, _effective_groups


# ============================================================================
# 3D window attention with relative position bias
# ============================================================================
class WindowAttention3D(nn.Module):
    """Multi-head self-attention within non-overlapping 3D windows."""

    def __init__(self, dim, window_size, num_heads=4, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size       # (wd, wh, ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1)
                * (2 * window_size[1] - 1)
                * (2 * window_size[2] - 1),
                num_heads,
            )
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Precompute relative position index
        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(
            torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij")
        )
        coords_flatten = torch.flatten(coords, 1)                    # (3, N)
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 3)
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)             # (N, N)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (B_, N, C)
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B_, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)                                    # (3, B_, h, N, d)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)                              # (B_, h, N, N)

        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ]
        bias = bias.view(N, N, -1).permute(2, 0, 1).contiguous()    # (h, N, N)
        attn = attn + bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition_3d(x, window_size):
    """(B, C, D, H, W) -> (B*nW, C, wd, wh, ww)."""
    B, C, D, H, W = x.shape
    wd, wh, ww = window_size
    x = x.view(B, C, D // wd, wd, H // wh, wh, W // ww, ww)
    windows = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return windows.view(-1, C, wd, wh, ww)


def window_reverse_3d(windows, window_size, B, C, D, H, W):
    """(B*nW, C, wd, wh, ww) -> (B, C, D, H, W)."""
    wd, wh, ww = window_size
    x = windows.view(B, D // wd, H // wh, W // ww, C, wd, wh, ww)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    return x.view(B, C, D, H, W)


class SwinTransformerBlock3D(nn.Module):
    """One Swin block: LayerNorm -> W-MSA -> residual -> LayerNorm -> MLP -> residual."""

    def __init__(self, dim, num_heads, window_size,
                 mlp_ratio=2.0, qkv_bias=True, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        wd, wh, ww = self.window_size

        # Capture the residual BEFORE any padding so shapes match at the end
        shortcut = x

        # Pad if not divisible by window size
        pad_d = (wd - D % wd) % wd
        pad_h = (wh - H % wh) % wh
        pad_w = (ww - W % ww) % ww
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        Dp, Hp, Wp = D + pad_d, H + pad_h, W + pad_w

        # (B, C, Dp, Hp, Wp) -> (B, Dp, Hp, Wp, C)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm1(x)

        # Partition into windows
        x_windows = window_partition_3d(
            x.permute(0, 4, 1, 2, 3).contiguous(), self.window_size
        )
        x_windows = x_windows.permute(0, 2, 3, 4, 1).reshape(
            -1, wd * wh * ww, C
        )
        attn_windows = self.attn(x_windows)

        # Reverse
        attn_windows = attn_windows.view(
            -1, wd, wh, ww, C
        ).permute(0, 4, 1, 2, 3)
        x = window_reverse_3d(attn_windows, self.window_size,
                              B, C, Dp, Hp, Wp)
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        # Second residual: LayerNorm -> MLP
        x = x + self.mlp(self.norm2(x))
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        # Remove padding
        if pad_d or pad_h or pad_w:
            x = x[:, :, :D, :H, :W]

        # At this point x and shortcut are both (B, C, D, H, W)
        return x + shortcut


# ============================================================================
# Gated Swin bottleneck
# ============================================================================
class SwinBottleneckGate3D(nn.Module):
    """Wraps the Swin block with a lightweight gate.

    y = x + alpha * gate * swin(x)

    alpha starts at 0, so the module is an identity at initialization and
    cannot destroy the network's early training. Over the course of training,
    alpha grows and the Swin branch contributes.
    """

    def __init__(self, channels, window_size=(2, 4, 4),
                 num_heads=4, depth=1, mlp_ratio=2.0, drop=0.0,
                 norm_groups=8):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.alpha = nn.Parameter(torch.zeros(1))

        eg = _effective_groups(norm_groups, channels)
        self.gate_conv = nn.Sequential(
            nn.Conv3d(channels, channels, 1, bias=False),
            nn.GroupNorm(eg, channels),
            nn.Sigmoid(),
        )
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=channels, num_heads=num_heads,
                window_size=window_size, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=drop, attn_drop=drop,
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        identity = x
        y = x
        for blk in self.blocks:
            y = blk(y)
        gate = self.gate_conv(y)
        return identity + self.alpha * gate * y


# ============================================================================
# Subclass of HybridResUNet3D with an opt-in Swin bottleneck
# ============================================================================
class HybridResUNet3D_Swin(HybridResUNet3D):
    """Phase 1 backbone with an optional Swin Transformer at the bottleneck."""

    def __init__(self, cfg, use_swin_bottleneck=False, swin_window=(2, 4, 4),
                 swin_heads=4, swin_depth=1, **kwargs):
        super().__init__(cfg, **kwargs)
        self.use_swin_bottleneck = use_swin_bottleneck

        if use_swin_bottleneck:
            f = cfg["init_features"]
            self.swin_bottleneck = SwinBottleneckGate3D(
                channels=f * 8,
                window_size=swin_window,
                num_heads=swin_heads,
                depth=swin_depth,
                mlp_ratio=2.0,
                drop=cfg.get("dropout_rate", 0.2),
                norm_groups=cfg["norm_groups"],
            )
            self._init_weights()

    def _forward_core(self, x):
        s1, s2, s3, b = self.encode(x)
        if self.use_ms:
            b = self.fusion(b, self.ms(b))
        if self.use_attention:
            b = self.att(b)
        if self.use_swin_bottleneck:
            b = self.swin_bottleneck(b)
        d1 = self.decode(b, s1, s2, s3)

        if self.use_refine:
            refined = self.boundary_refinement(d1)
        else:
            refined = d1

        if self.use_gabor_refine:
            refined = self.gabor_gate(refined, d1)

        if self.use_refine:
            logits = self.final_convs(torch.cat([d1, refined], dim=1))
        else:
            logits = self.final_conv(refined)

        return logits, refined


def build_model_swin(cfg, variant="+swin_bottleneck",
                      swin_window=(2, 4, 4), swin_heads=4, swin_depth=1):
    """Build the Phase 1 backbone with a Swin bottleneck."""
    if variant != "+swin_bottleneck":
        raise ValueError(f"Unknown Swin variant: {variant}")

    return HybridResUNet3D_Swin(
        cfg,
        use_swin_bottleneck=True,
        swin_window=swin_window,
        swin_heads=swin_heads,
        swin_depth=swin_depth,
        # All other flags off, matching the Phase 1 `baseline`
        use_ms=False,
        use_attention=False,
        use_refine=False,
        use_gabor_refine=False,
        use_boundary_supervision=False,
    )
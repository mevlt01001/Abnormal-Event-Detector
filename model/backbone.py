"""Pretrained 3D Video Backbones for feature extraction and end-to-end inference."""

from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn
from torchvision.models.video import (
    MC3_18_Weights,
    MViT_V1_B_Weights,
    MViT_V2_S_Weights,
    R2Plus1D_18_Weights,
    R3D_18_Weights,
    S3D_Weights,
    Swin3D_B_Weights,
    Swin3D_S_Weights,
    Swin3D_T_Weights,
    mc3_18,
    mvit_v1_b,
    mvit_v2_s,
    r2plus1d_18,
    r3d_18,
    s3d,
    swin3d_b,
    swin3d_s,
    swin3d_t,
)

BACKBONE_CREATORS = {
    "swin3d_t": swin3d_t,
    "swin3d_s": swin3d_s,
    "swin3d_b": swin3d_b,
    "mvit_v1_b": mvit_v1_b,
    "mvit_v2_s": mvit_v2_s,
    "r3d_18": r3d_18,
    "mc3_18": mc3_18,
    "r2plus1d_18": r2plus1d_18,
    "s3d": s3d,
}

BACKBONE_WEIGHTS = {
    "swin3d_t": Swin3D_T_Weights.KINETICS400_V1,
    "swin3d_s": Swin3D_S_Weights.KINETICS400_V1,
    "swin3d_b": Swin3D_B_Weights.KINETICS400_V1,
    "mvit_v1_b": MViT_V1_B_Weights.KINETICS400_V1,
    "mvit_v2_s": MViT_V2_S_Weights.KINETICS400_V1,
    "r3d_18": R3D_18_Weights.KINETICS400_V1,
    "mc3_18": MC3_18_Weights.KINETICS400_V1,
    "r2plus1d_18": R2Plus1D_18_Weights.KINETICS400_V1,
    "s3d": S3D_Weights.KINETICS400_V1,
}

BACKBONE_FEATURE_DIMS: Dict[str, int] = {
    "swin3d_t": 768,
    "swin3d_s": 768,
    "swin3d_b": 1024,
    "mvit_v1_b": 768,
    "mvit_v2_s": 768,
    "r3d_18": 512,
    "mc3_18": 512,
    "r2plus1d_18": 512,
    "s3d": 1024,
}


def create_backbone(backbone_name: str = "swin3d_t", pretrained: bool = True) -> nn.Module:
    """Creates a pretrained 3D video backbone with classification head stripped.

    Args:
        backbone_name: Name of video backbone (default: 'swin3d_t').
        pretrained: Whether to load Kinetics-400 pretrained weights.

    Returns:
        backbone: nn.Module taking [B, C, T, H, W] and outputting [B, feature_dim].
    """
    backbone_name = backbone_name.lower()
    if backbone_name not in BACKBONE_CREATORS:
        raise ValueError(f"Unknown backbone: '{backbone_name}'. Supported: {list(BACKBONE_CREATORS.keys())}")

    weights = BACKBONE_WEIGHTS[backbone_name] if pretrained else None
    model = BACKBONE_CREATORS[backbone_name](weights=weights)

    # Strip classification head so model outputs raw feature representations
    if hasattr(model, "head"):
        model.head = nn.Identity()
    elif hasattr(model, "fc"):
        model.fc = nn.Identity()
    elif hasattr(model, "classifier"):
        model.classifier = nn.Identity()

    return model


def get_backbone_dim(backbone_name: str = "swin3d_t") -> int:
    """Returns feature dimension for backbone (e.g., 768 for Swin3D-T)."""
    return BACKBONE_FEATURE_DIMS.get(backbone_name.lower(), 768)

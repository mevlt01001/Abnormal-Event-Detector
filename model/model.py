from __future__ import annotations

import os
from typing import Optional, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models.video import (
    MViT,
    VideoResNet,
    S3D,
    SwinTransformer3d,
    mvit_v1_b,
    mvit_v2_s,
    r3d_18,
    mc3_18,
    r2plus1d_18,
    s3d,
    swin3d_t,
    swin3d_s,
    swin3d_b,
    MViT_V1_B_Weights,
    MViT_V2_S_Weights,
    R3D_18_Weights,
    MC3_18_Weights,
    R2Plus1D_18_Weights,
    S3D_Weights,
    Swin3D_T_Weights,
    Swin3D_S_Weights,
    Swin3D_B_Weights,
)

VideoModel = Union[MViT, VideoResNet, S3D, SwinTransformer3d]

model_creator = {
    "mvit_v1_b": mvit_v1_b,
    "mvit_v2_s": mvit_v2_s,
    "r3d_18": r3d_18,
    "mc3_18": mc3_18,
    "r2plus1d_18": r2plus1d_18,
    "s3d": s3d,
    "swin3d_t": swin3d_t,
    "swin3d_s": swin3d_s,
    "swin3d_b": swin3d_b,
}

model_weights = {
    "mvit_v1_b": MViT_V1_B_Weights,
    "mvit_v2_s": MViT_V2_S_Weights,
    "r3d_18": R3D_18_Weights,
    "mc3_18": MC3_18_Weights,
    "r2plus1d_18": R2Plus1D_18_Weights,
    "s3d": S3D_Weights,
    "swin3d_t": Swin3D_T_Weights,
    "swin3d_s": Swin3D_S_Weights,
    "swin3d_b": Swin3D_B_Weights,
}

model_feature_dims = {
    "mvit_v1_b": 768,
    "mvit_v2_s": 768,
    "r3d_18": 512,
    "mc3_18": 512,
    "r2plus1d_18": 512,
    "s3d": 1024,
    "swin3d_t": 768,
    "swin3d_s": 768,
    "swin3d_b": 1024,
}


@torch.no_grad()
def get_feature_dim(model: VideoModel) -> int:
    """Infers output feature dimension of a video backbone dynamically."""
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, 16, 224, 224, device=device)  # [B, C, T, H, W]
    output = model(dummy_input)
    f_dim = output.shape[-1]
    return f_dim


def create_base_model(model_name: str, pretrained: bool = True) -> VideoModel:
    """Creates a pretrained video backbone and replaces classifier head with Identity."""
    model_name = model_name.lower()
    if model_name not in model_creator:
        raise ValueError(f"There is no {model_name} in {list(model_creator.keys())}")

    base_model_creator = model_creator[model_name]
    weights = model_weights[model_name].DEFAULT if pretrained else None
    base_model = base_model_creator(weights=weights)
    base_model_class = base_model.__class__.__name__

    if base_model_class == "MViT":
        base_model.head = nn.Identity()
    elif base_model_class == "VideoResNet":
        base_model.fc = nn.Identity()
    elif base_model_class == "S3D":
        base_model.classifier = nn.Identity()
    elif base_model_class == "SwinTransformer3d":
        base_model.head = nn.Identity()
    else:
        raise ValueError(f"There is no such model class {base_model_class}")

    return base_model


class FC_head(nn.Module):
    """Fully connected classification head with L2 normalization."""

    def __init__(
        self,
        in_features: int,
        num_classes: int = 1,
        num_hidden_layers: int = 3,
        dropout: float = 0.5,
        use_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        self.use_sigmoid = use_sigmoid
        layers = []
        layers.append(
            nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        )

        for _ in range(num_hidden_layers):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_features // 2, in_features // 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
            )

        final_layer = [nn.Linear(in_features // 2, num_classes)]
        if use_sigmoid:
            final_layer.append(nn.Sigmoid())
        layers.append(nn.Sequential(*final_layer))

        self.MLP = nn.Sequential(*layers)


    def forward(self, x: Tensor) -> Tensor:
        x = F.normalize(x, p=2, dim=-1)
        return self.MLP(x)


class Model(nn.Module):
    """End-to-end video anomaly detection model wrapping backbone and MLP head."""

    def __init__(self, base_model_name: str, pretrained: bool = True, **kwargs):
        super().__init__()
        base_model_name = base_model_name.lower()
        self.base_model_name = base_model_name
        self.video_model = create_base_model(base_model_name, pretrained=pretrained)
        self.feature_dim = model_feature_dims.get(
            base_model_name, get_feature_dim(self.video_model)
        )
        self.classifier = FC_head(self.feature_dim, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = self.video_model(x)
        x = self.classifier(x)
        return x

    @torch.no_grad()
    def extract_features(
        self,
        video_path: str,
        num_segments: int = 32,
        clip_size: int = 16,
        fps: int = 30,
        width: int = 224,
        height: int = 224,
        batch_size: int = 8,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """Extracts (num_segments, feature_dim) representation from video file.

        Divides video into `num_segments` temporal segments using Decord. For each segment,
        extracts 16-frame (`clip_size`) clips, processes them through the backbone, and
        averages clip features across the segment following Sultani et al. CVPR 2018.
        """
        from utils.video_utils import fetch_video_segments
        from tqdm import tqdm

        was_training = self.training
        self.eval()

        device = next(self.parameters()).device

        segment_gen = fetch_video_segments(
            video_path=video_path,
            num_segments=num_segments,
            fps=fps,
            clip_size=clip_size,
            resize_dim=(height, width),
        )

        segment_features = []
        video_name = os.path.basename(video_path)

        pbar = tqdm(
            segment_gen,
            total=num_segments,
            desc=f"  ↳ [{video_name[:24]}]",
            leave=False,
            disable=not show_progress,
        )

        for seg_tensor in pbar:  # [K, 3, clip_size, H, W]
            if seg_tensor.dtype == torch.uint8 or seg_tensor.max() > 1.0:
                seg_tensor = seg_tensor.float() / 255.0
            seg_tensor = seg_tensor.to(device)

            # Process clips in this segment through backbone
            if seg_tensor.shape[0] <= batch_size:
                feats = self.video_model(seg_tensor)  # [K, feature_dim]
            else:
                chunks = torch.split(seg_tensor, batch_size, dim=0)
                feats = torch.cat([self.video_model(c) for c in chunks], dim=0)

            # Sultani paper: average all clip features within that segment
            seg_feature = feats.mean(dim=0, keepdim=True)  # [1, feature_dim]
            segment_features.append(seg_feature.detach().cpu())

        if was_training:
            self.train()

        if not segment_features:
            return torch.empty((0, self.feature_dim))

        return torch.cat(segment_features, dim=0)

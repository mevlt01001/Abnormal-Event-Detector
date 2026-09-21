"""Unified Anomaly Detector module combining 3D backbone and AnomalyHead."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from model.backbone import create_backbone, get_backbone_dim
from model.head import AnomalyHead


class AnomalyDetector(nn.Module):
    """End-to-End Class-Aware Video Anomaly Detection PyTorch Module.

    Integrates a 3D spatio-temporal video backbone (e.g. Swin3D-T) with a pyramidal
    AnomalyHead for segment-level multi-class ranking.

    Args:
        backbone_name: Name of video backbone (default: 'swin3d_t').
            Set to None to initialize a lightweight head-only model for precomputed features.
        num_classes: Number of macro anomaly classes (default: 6).
        in_features: Input feature dimension. If None, derived automatically from backbone.
        pretrained_backbone: Whether to load Kinetics-400 pretrained backbone weights.
    """

    def __init__(
        self,
        backbone_name: Optional[str] = "swin3d_t",
        num_classes: int = 6,
        in_features: Optional[int] = None,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.backbone_name = backbone_name.lower() if backbone_name else None

        if self.backbone_name:
            self.backbone = create_backbone(self.backbone_name, pretrained=pretrained_backbone)
            feat_dim = get_backbone_dim(self.backbone_name)
        else:
            self.backbone = None
            feat_dim = in_features if in_features else 768

        self.feature_dim = feat_dim
        self.head = AnomalyHead(in_features=feat_dim, num_classes=num_classes)

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass for pre-extracted segment feature tensors.

        Args:
            features: Tensor of shape [B, T, feature_dim] or [B, feature_dim].

        Returns:
            logits: Class logits of shape [B, T, num_classes] or [B, num_classes].
        """
        # [B, T, feature_dim] -> [B, T, num_classes]
        logits = self.head(features)
        return logits

    def forward_video(self, video_clips: torch.Tensor) -> torch.Tensor:
        """Forward pass for spatio-temporal video clips.

        Args:
            video_clips: Tensor of shape [B, C, clip_size, H, W] in [0, 1].

        Returns:
            logits: Class logits of shape [B, num_classes].
        """
        if self.backbone is None:
            raise RuntimeError("Backbone is not initialized. Initialize model with backbone_name.")

        # [B, C, clip_size, H, W] -> [B, feature_dim]
        feats = self.backbone(video_clips)

        # [B, feature_dim] -> [B, num_classes]
        logits = self.head(feats)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Universal forward pass dispatching based on input dimensionality.

        Args:
            x: Either:
                - 5D Video Tensor: [B, C, T, H, W]
                - 3D Feature Tensor: [B, T, feature_dim]
                - 2D Feature Tensor: [B, feature_dim]

        Returns:
            Class logits corresponding to input sequence length.
        """
        if x.ndim == 5:
            # [B, C, T, H, W] -> [B, num_classes]
            return self.forward_video(x)
        elif x.ndim in (2, 3):
            # [B, T, feature_dim] -> [B, T, num_classes] or [B, feature_dim] -> [B, num_classes]
            return self.forward_features(x)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}. Expected 2D, 3D, or 5D tensor.")

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        backbone_name: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[AnomalyDetector, Dict[str, Any]]:
        """Instantiates and loads AnomalyDetector from a trained checkpoint.

        Args:
            checkpoint_path: Path to checkpoint .pt file.
            backbone_name: Optional backbone name if end-to-end video inference is desired.
            device: Target torch device.

        Returns:
            Tuple of (detector_instance, checkpoint_dictionary).
        """
        dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        head, ckpt = AnomalyHead.load_from_checkpoint(checkpoint_path, device=dev)
        detector = cls(
            backbone_name=backbone_name,
            num_classes=head.num_classes,
            in_features=head.in_features,
            pretrained_backbone=False,
        ).to(dev)
        detector.head = head
        return detector, ckpt

"""Unified video feature extraction engine using 3D backbones."""

from __future__ import annotations

import os
from typing import Optional, Union
import torch
import torch.nn as nn
from tqdm import tqdm

from core.video_processor import VideoProcessor


class FeatureExtractor:
    """Extracts segment-level representations from video using a 3D backbone.

    Following Sultani et al. CVPR 2018:
    1. Video is partitioned into `num_segments` (default: 32).
    2. Spatio-temporal clips (16 frames) are extracted within each segment.
    3. The 3D backbone projects each clip to a D-dimensional feature vector.
    4. Clip features within each segment are averaged to produce a [num_segments, D] tensor.

    Args:
        backbone: Pretrained PyTorch video backbone module (e.g., Swin3D-T).
        device: Torch device (cuda or cpu).
        target_fps: Target frame rate for video sampling (default: 20.0).
        clip_size: Number of frames per clip (default: 16).
        batch_size: Sub-batch size of clips passed to GPU simultaneously (default: 8).
    """

    def __init__(
        self,
        backbone: nn.Module,
        device: Optional[Union[str, torch.device]] = None,
        target_fps: float = 20.0,
        clip_size: int = 16,
        overlap: Optional[Union[int, float]] = None,
        overlap_ratio: float = 0.0,
        stride: Optional[int] = None,
        max_clips_per_segment: Optional[int] = 16,
        batch_size: int = 8,
    ) -> None:
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = backbone.to(self.device).eval()
        self.target_fps = target_fps
        self.clip_size = clip_size
        self.batch_size = batch_size
        self.processor = VideoProcessor(
            target_fps=target_fps,
            clip_size=clip_size,
            stride=stride,
            overlap_ratio=overlap_ratio,
            overlap=overlap,
            max_clips_per_segment=max_clips_per_segment,
        )

    @torch.no_grad()
    def extract_video(
        self,
        video_path: str,
        num_segments: int = 32,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """Extracts [num_segments, feature_dim] tensor for a single video file.

        Args:
            video_path: Path to video file (.mp4).
            num_segments: Number of temporal segments to divide the video into (default: 32).
            show_progress: If True, displays a tqdm progress bar.

        Returns:
            features: Tensor of shape [num_segments, feature_dim] on CPU.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        segment_gen = self.processor.extract_uniform_segments(video_path, num_segments=num_segments)
        segment_features = []

        iterator = tqdm(segment_gen, total=num_segments, desc="Extracting Features", disable=not show_progress)
        for seg_tensor in iterator:
            # seg_tensor shape: [K, C, clip_size, H, W]
            K, C, T, H, W = seg_tensor.shape
            seg_tensor = seg_tensor.to(self.device)

            # Sub-batch clips if segment contains multiple 16-frame clips
            if K <= self.batch_size:
                # [K, C, T, H, W] -> [K, feature_dim]
                feats = self.backbone(seg_tensor)
            else:
                chunks = torch.split(seg_tensor, self.batch_size, dim=0)
                # [K, feature_dim]
                feats = torch.cat([self.backbone(c) for c in chunks], dim=0)

            # [K, feature_dim] -> [1, feature_dim] (Averaged across clips in segment)
            seg_feature = feats.mean(dim=0, keepdim=True).detach().cpu()
            segment_features.append(seg_feature)

        if not segment_features:
            raise RuntimeError(f"Could not extract any features from {video_path}")

        # [num_segments, feature_dim] (e.g., [32, 768])
        video_features = torch.cat(segment_features, dim=0)
        return video_features

    def extract_and_save(
        self,
        video_path: str,
        output_path: str,
        num_segments: int = 32,
    ) -> str:
        """Extracts features and saves them directly to a .pt file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        feats = self.extract_video(video_path, num_segments=num_segments)
        torch.save(feats, output_path)
        return output_path

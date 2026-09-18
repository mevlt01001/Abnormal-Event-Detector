from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.visualization import plot_anomaly_timeline


class SegmentRankingHead(nn.Module):
    """Segment Ranking Multi-Layer Perceptron (MLP) for video anomaly detection.

    Takes segment feature vectors, applies noise augmentation during training,
    normalizes features with L2 norm, and outputs an anomaly probability per segment.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: Sequence[int] = (512, 256),
        dropout_rates: Sequence[float] = (0.7, 0.6),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.dropout_rates = list(dropout_rates)

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for idx, (h_dim, drop_rate) in enumerate(zip(self.hidden_dims, self.dropout_rates)):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            if drop_rate > 0.0:
                layers.append(nn.Dropout(p=drop_rate))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Feature tensor of shape (..., input_dim).

        Returns:
            Anomaly scores of shape (..., 1) in range [0, 1].
        """
        if self.training:
            noise = torch.rand_like(x) * 0.05 - torch.rand_like(x) * 0.05
            x = x + noise

        x = F.normalize(x, p=2, dim=-1)
        return self.mlp(x)

    @torch.no_grad()
    def score_to_segments(
        self,
        patch_feats: torch.Tensor,
        video_seconds: float,
        threshold: float = 0.3,
        tolerance_sec: float = 3.0,
        padding_sec: float = 3.0,
        plot_graph: bool = False,
        save_file_name: str = "anomaly_segmentation_plot.png",
    ) -> List[Dict[str, Any]]:
        """Scores video feature representations and computes temporal anomaly segment timestamps.

        Args:
            patch_feats: Features tensor for segments [N, input_dim].
            video_seconds: Video duration in seconds.
            threshold: Anomaly score threshold.
            tolerance_sec: Maximum time gap (sec) between abnormal events to merge them.
            padding_sec: Seconds of context added before and after detected anomaly.
            plot_graph: If True, renders and saves a timeline plot.
            save_file_name: Destination filename for the plot.

        Returns:
            List of dicts:
                [{"start_time": float, "end_time": float, "duration": float, "score": float}, ...]
        """
        was_training = self.training
        self.eval()

        scores = self.forward(patch_feats).squeeze(-1)  # [N]
        scores_org = scores.unsqueeze(0).unsqueeze(0)  # [1, 1, N]

        kernel_size = 21
        interpolate_size = max(len(scores), int(video_seconds * 10), 100)

        # 1D linear and nearest interpolation
        scores_linear = F.interpolate(
            scores_org, size=interpolate_size, mode="linear", align_corners=True
        )
        scores_nearest = F.interpolate(
            scores_org, size=interpolate_size, mode="nearest"
        )

        # Smooth curve with average pooling
        pad_size = kernel_size // 2
        scores_linear = F.pad(scores_linear, (pad_size, pad_size), mode="reflect")
        scores_linear = F.avg_pool1d(scores_linear, kernel_size=kernel_size, stride=1)

        scores_linear = scores_linear.squeeze(0).squeeze(0)
        scores_nearest = scores_nearest.squeeze(0).squeeze(0)

        dt = video_seconds / interpolate_size if interpolate_size > 0 else 0.0
        anomaly_indices = torch.where(scores_linear >= threshold)[0].tolist()

        final_segments: List[Dict[str, Any]] = []

        if anomaly_indices:
            raw_segments = []
            curr_start = anomaly_indices[0]
            curr_end = anomaly_indices[0]

            for idx in anomaly_indices[1:]:
                time_gap = (idx - curr_end) * dt
                if time_gap <= tolerance_sec:
                    curr_end = idx
                else:
                    raw_segments.append((curr_start, curr_end))
                    curr_start = idx
                    curr_end = idx
            raw_segments.append((curr_start, curr_end))

            # Apply temporal padding
            padded_segments = []
            for start_idx, end_idx in raw_segments:
                start_time = start_idx * dt
                end_time = end_idx * dt
                p_start = max(0.0, start_time - padding_sec)
                p_end = min(video_seconds, end_time + padding_sec)
                padded_segments.append([p_start, p_end, start_idx, end_idx])

            # Merge overlapping padded segments
            merged_segments = []
            curr_p_start, curr_p_end, curr_s_idx, curr_e_idx = padded_segments[0]

            for p_start, p_end, s_idx, e_idx in padded_segments[1:]:
                if p_start <= curr_p_end:
                    curr_p_end = max(curr_p_end, p_end)
                    curr_e_idx = max(curr_e_idx, e_idx)
                else:
                    merged_segments.append((curr_p_start, curr_p_end, curr_s_idx, curr_e_idx))
                    curr_p_start, curr_p_end, curr_s_idx, curr_e_idx = p_start, p_end, s_idx, e_idx
            merged_segments.append((curr_p_start, curr_p_end, curr_s_idx, curr_e_idx))

            for p_start, p_end, s_idx, e_idx in merged_segments:
                seg_scores = scores_linear[s_idx : e_idx + 1]
                max_score = seg_scores.max().item() if len(seg_scores) > 0 else 0.0
                final_segments.append({
                    "start_time": round(float(p_start), 2),
                    "end_time": round(float(p_end), 2),
                    "duration": round(float(p_end - p_start), 2),
                    "score": round(float(max_score), 4),
                })

        if plot_graph:
            plot_anomaly_timeline(
                scores=scores_linear.cpu(),
                scores_raw=scores_nearest.cpu(),
                segments=final_segments,
                video_seconds=video_seconds,
                threshold=threshold,
                save_path=save_file_name,
            )

        if was_training:
            self.train()

        return final_segments

from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoAnomalyLoss(nn.Module):
    """Sultani et al. Multiple Instance Learning (MIL) Ranking Loss for Video Anomaly Detection.

    Loss components:
    1. Hinge Loss (Ranking): max(0, 1 - max(S_anomaly) + max(S_normal))
       Encourages the maximum anomaly score in an anomalous video to be higher
       than the maximum anomaly score in a normal video by at least a margin of 1.0.
    2. Temporal Smoothness Loss: sum_{t=1}^{T-1} (s_t - s_{t+1})^2
       Encourages temporally adjacent segments in anomalous videos to have continuous scores.
    3. Sparsity Loss: sum_{t=1}^{T} s_t
       Reflects the prior that anomalies occur only in brief periods of an anomalous video.
    """

    def __init__(
        self,
        smoothness_weight: float = 0.0001,
        sparsity_weight: float = 0.0001,
    ) -> None:
        super().__init__()
        self.smoothness_weight = float(smoothness_weight)
        self.sparsity_weight = float(sparsity_weight)

    def _normalize_shape(self, scores: torch.Tensor) -> torch.Tensor:
        """Converts inputs of shapes (N,), (N, 1), (B, N, 1), or (B, N) to (B, N)."""
        if scores.ndim == 1:
            return scores.unsqueeze(0)  # (1, N)
        if scores.ndim == 2:
            if scores.shape[1] == 1:
                return scores.squeeze(1).unsqueeze(0)  # (1, N)
            return scores  # (B, N)
        if scores.ndim == 3 and scores.shape[2] == 1:
            return scores.squeeze(2)  # (B, N)
        return scores.view(scores.shape[0], -1)

    def forward(
        self,
        anomaly_scores: torch.Tensor,
        normal_scores: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Calculates ranking, smoothness, and sparsity losses.

        Args:
            anomaly_scores: Predicted anomaly probabilities for anomalous video segments.
            normal_scores: Predicted anomaly probabilities for normal video segments.

        Returns:
            Dict containing:
                "total": Total weighted scalar loss tensor.
                "hinge": Hinge ranking loss scalar tensor.
                "smoothness": Temporal smoothness loss scalar tensor.
                "sparsity": Sparsity loss scalar tensor.
        """
        y_anom = self._normalize_shape(anomaly_scores)  # (B, N_anom)
        y_norm = self._normalize_shape(normal_scores)  # (B, N_norm)

        # 1. Hinge Ranking Loss
        max_anomaly = torch.max(y_anom, dim=1)[0]  # (B,)
        max_normal = torch.max(y_norm, dim=1)[0]  # (B,)
        hinge_per_sample = F.relu(1.0 - max_anomaly + max_normal)
        hinge_loss = torch.mean(hinge_per_sample)

        # 2. Smoothness Loss (on anomaly video)
        if y_anom.shape[1] > 1:
            diff = y_anom[:, :-1] - y_anom[:, 1:]
            smoothness_loss = torch.mean(torch.sum(diff ** 2, dim=1))
        else:
            smoothness_loss = torch.tensor(0.0, device=anomaly_scores.device)

        # 3. Sparsity Loss (on anomaly video)
        sparsity_loss = torch.mean(torch.sum(y_anom, dim=1))

        # Total Loss
        total_loss = (
            hinge_loss
            + self.smoothness_weight * smoothness_loss
            + self.sparsity_weight * sparsity_loss
        )

        return {
            "total": total_loss,
            "hinge": hinge_loss,
            "smoothness": smoothness_loss,
            "sparsity": sparsity_loss,
        }

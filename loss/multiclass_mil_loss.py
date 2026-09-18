from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassMILLoss(nn.Module):
    """Top-K Multi-Label Multiple Instance Learning (MIL) Loss for Video Anomaly Detection.

    Formulation:
    1. Top-K Bag Pooling:
       For each video (bag) with T segments and C classes:
       Each class prediction is aggregated as the mean of the top-k highest scoring segments.
    2. Multi-Label Classification:
       Computed via Binary Cross Entropy between aggregated bag scores and multi-hot ground truth.
       Normal videos have target = [0, 0, ..., 0].
       Anomalous videos have target multi-hot encoded (e.g. [0, 1, 0, 0, 0, 1]).
    3. Regularization:
       - Temporal Smoothness: penalizes abrupt score fluctuations between consecutive segments.
       - Sparsity: penalizes high anomaly predictions everywhere across the video.
    """

    def __init__(
        self,
        k: int = 1,
        smoothness_weight: float = 0.0001,
        sparsity_weight: float = 0.0001,
        from_logits: bool = True,
    ) -> None:
        super().__init__()
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        self.k = int(k)
        self.smoothness_weight = float(smoothness_weight)
        self.sparsity_weight = float(sparsity_weight)
        self.from_logits = bool(from_logits)

    def _normalize_shapes(
        self, segment_scores: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalizes segment scores to (B, T, C) and target to (B, C)."""
        # segment_scores: (T, C) or (B, T, C)
        if segment_scores.ndim == 2:
            segment_scores = segment_scores.unsqueeze(0)  # (1, T, C)
        elif segment_scores.ndim != 3:
            raise ValueError(
                f"Expected segment_scores to have 2 or 3 dimensions, got {segment_scores.ndim} with shape {segment_scores.shape}"
            )

        # target: (C,) or (B, C)
        if target.ndim == 1:
            target = target.unsqueeze(0)  # (1, C)
        elif target.ndim != 2:
            raise ValueError(
                f"Expected target to have 1 or 2 dimensions, got {target.ndim} with shape {target.shape}"
            )

        if segment_scores.shape[0] != target.shape[0]:
            raise ValueError(
                f"Batch size mismatch: segment_scores batch {segment_scores.shape[0]} != target batch {target.shape[0]}"
            )
        if segment_scores.shape[2] != target.shape[1]:
            raise ValueError(
                f"Class count mismatch: segment_scores classes {segment_scores.shape[2]} != target classes {target.shape[1]}"
            )

        return segment_scores, target.float()

    def forward(
        self,
        segment_scores: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Computes multi-class Top-K MIL loss.

        Args:
            segment_scores: (B, T, C) or (T, C) tensor of segment-level class predictions.
            target: (B, C) or (C,) tensor of multi-hot ground truth labels.

        Returns:
            Dict containing:
                "total": Total weighted scalar loss tensor.
                "bce": Binary Cross Entropy scalar loss tensor.
                "smoothness": Temporal smoothness scalar loss tensor.
                "sparsity": Sparsity scalar loss tensor.
        """
        scores, target = self._normalize_shapes(segment_scores, target)
        b, t, c = scores.shape

        effective_k = min(self.k, t)
        # Top-K pooling along temporal dimension (dim=1)
        topk_scores = torch.topk(scores, k=effective_k, dim=1).values.mean(dim=1)  # (B, C)

        # BCE Loss
        if self.from_logits:
            bce_loss = F.binary_cross_entropy_with_logits(topk_scores, target)
            probs = torch.sigmoid(scores)
        else:
            clamped_scores = torch.clamp(topk_scores, 1e-7, 1.0 - 1e-7)
            bce_loss = F.binary_cross_entropy(clamped_scores, target)
            probs = scores

        # Temporal Smoothness: (s_t - s_{t+1})^2
        if t > 1:
            diff = probs[:, 1:, :] - probs[:, :-1, :]
            smoothness_loss = torch.mean(diff**2)
        else:
            smoothness_loss = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

        # Sparsity: L1-like penalty on anomaly activations
        sparsity_loss = torch.mean(probs)

        total_loss = (
            bce_loss
            + self.smoothness_weight * smoothness_loss
            + self.sparsity_weight * sparsity_loss
        )

        return {
            "total": total_loss,
            "bce": bce_loss,
            "smoothness": smoothness_loss,
            "sparsity": sparsity_loss,
        }

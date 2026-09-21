"""Class-Aware Multiple Instance Learning (MIL) Loss with Top-K Pooling."""

from __future__ import annotations

from typing import Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassAwareMILLoss(nn.Module):
    """Segment-level Class-Aware Multiple Instance Learning (MIL) Loss.

    For each video (bag) consisting of T temporal segments:
    1. Top-K segment probabilities are pooled per class to represent the video-level event score.
    2. Multi-label Binary Cross-Entropy is computed against ground truth macro-class indicators.
    3. Temporal smoothness and feature sparsity regularizations are applied to anomaly bags.

    Args:
        k: Number of highest-scoring segments to average for bag-level pooling (default: 1).
        smoothness_weight: Weight for temporal smoothness regularization (default: 0.0001).
        sparsity_weight: Weight for feature sparsity regularization (default: 0.0001).
        pos_weight: Optional positive class weighting tensor of shape [num_classes] to combat imbalance.
        eps: Epsilon floor for numerical stability (default: 1e-7).
    """

    def __init__(
        self,
        k: int = 1,
        smoothness_weight: float = 0.0001,
        sparsity_weight: float = 0.0001,
        pos_weight: Optional[Union[float, torch.Tensor]] = 3.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.k = max(1, int(k))
        self.smoothness_weight = float(smoothness_weight)
        self.sparsity_weight = float(sparsity_weight)
        self.eps = float(eps)

        if isinstance(pos_weight, torch.Tensor):
            self.register_buffer("pos_weight", pos_weight.float())
        elif isinstance(pos_weight, (int, float)):
            self.pos_weight = float(pos_weight)
        else:
            self.pos_weight = 3.0

    def forward(
        self,
        logits_or_probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Calculates class-aware Top-K BCE, temporal smoothness, and sparsity losses.

        Args:
            logits_or_probs: Tensor of shape [B, T, num_classes] representing segment logits or probabilities.
            targets: Binary ground truth tensor of shape [B, num_classes] in {0.0, 1.0}.

        Returns:
            Dictionary containing:
                "total": Total weighted scalar loss.
                "bce": Multi-label Top-K BCE loss.
                "smoothness": Temporal smoothness penalty.
                "sparsity": Temporal sparsity penalty.
        """
        # Ensure 3D shape: [B, T, num_classes]
        if logits_or_probs.ndim == 2:
            # [B, num_classes] -> [B, 1, num_classes]
            p = torch.sigmoid(logits_or_probs).unsqueeze(1)
        else:
            # [B, T, num_classes] -> [B, T, num_classes]
            p = torch.sigmoid(logits_or_probs) if logits_or_probs.max() > 1.0 or logits_or_probs.min() < 0.0 else logits_or_probs

        B, T, num_classes = p.shape
        device = p.device
        # [B, num_classes]
        y = targets.float().to(device)

        # 1. Top-K pooling per class across temporal segments
        k_val = min(self.k, T)
        # [B, T, num_classes] -> [B, num_classes] (Averaged top-k segment probabilities per class)
        topk_vals = torch.topk(p, k=k_val, dim=1).values.mean(dim=1)
        # [B, T, num_classes] -> [B, num_classes] (Max segment probability per class)
        max_vals = torch.max(p, dim=1).values

        # Clamp probabilities to avoid log(0) loss divergence
        topk_vals = torch.clamp(topk_vals, min=self.eps, max=1.0 - self.eps)
        max_vals = torch.clamp(max_vals, min=self.eps, max=1.0 - self.eps)

        # 2. Multi-label BCE Loss: Anomaly classes use top-k, Normal classes use max
        # [B, num_classes]
        loss_pos = -torch.log(topk_vals)
        # [B, num_classes]
        loss_neg = -torch.log(1.0 - max_vals)

        # Positive class weighting to balance 1 positive vs 5 negative classes per bag
        if isinstance(self.pos_weight, torch.Tensor):
            pos_factor = self.pos_weight.to(device)
        else:
            pos_factor = self.pos_weight

        # [B, num_classes]
        bce_per_sample = pos_factor * (y * loss_pos) + (1.0 - y) * loss_neg

        # Scalar BCE Loss
        bce_loss = torch.mean(bce_per_sample)

        # 3. Temporal Smoothness & Sparsity on anomaly bags only
        # [B] boolean mask
        is_anomaly = (torch.sum(y, dim=1) > 0.5)
        num_anom = torch.sum(is_anomaly)

        if num_anom > 0 and T > 1:
            # [N_anom, T, num_classes]
            p_anom = p[is_anomaly]
            # [N_anom, T-1, num_classes] (Adjacent frame score differences)
            diff = p_anom[:, 1:, :] - p_anom[:, :-1, :]
            # Scalar smoothness penalty
            smoothness_loss = torch.mean(torch.sum(diff ** 2, dim=[1, 2]))
            # Scalar sparsity penalty
            sparsity_loss = torch.mean(torch.sum(p_anom, dim=[1, 2]))
        else:
            smoothness_loss = torch.tensor(0.0, device=device)
            sparsity_loss = torch.tensor(0.0, device=device)

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

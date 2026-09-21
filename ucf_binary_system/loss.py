"""Multiple Instance Learning (MIL) Loss for Binary Video Anomaly Detection."""

from __future__ import annotations

from typing import Dict, Optional, Union
import torch
import torch.nn as nn


class BinaryMILLoss(nn.Module):
    """Segment-level Binary Multiple Instance Learning (MIL) Loss.

    For each video (bag) consisting of T temporal segments:
    - Anomalous bags (y = 1): Loss penalizes if the Top-K segment probabilities are low.
    - Normal bags (y = 0): Loss penalizes if the maximum segment probability is high.
    - Temporal smoothness & sparsity penalties enforce localized, coherent anomaly curves.

    Args:
        k: Number of top-scoring segments pooled per anomaly bag (default: 3).
        smoothness_weight: Weight for temporal smoothness regularization (default: 0.0001).
        sparsity_weight: Weight for anomaly segment sparsity (default: 0.0001).
        eps: Epsilon clamp floor for numerical stability (default: 1e-5).
    """

    def __init__(
        self,
        k: int = 3,
        smoothness_weight: float = 0.0001,
        sparsity_weight: float = 0.0001,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.k = max(1, int(k))
        self.smoothness_weight = float(smoothness_weight)
        self.sparsity_weight = float(sparsity_weight)
        self.eps = float(eps)

    def forward(
        self,
        logits_or_probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Calculates Top-K Binary MIL loss with temporal regularization.

        Args:
            logits_or_probs: Tensor of shape [B, T, 1] or [B, T].
            targets: Binary ground truth tensor of shape [B, 1] or [B] in {0.0, 1.0}.

        Returns:
            Dictionary containing 'total', 'bce', 'smoothness', and 'sparsity' scalar tensors.
        """
        # Squeeze trailing singleton dimension if present: [B, T, 1] -> [B, T]
        if logits_or_probs.ndim == 3 and logits_or_probs.shape[-1] == 1:
            logits_or_probs = logits_or_probs.squeeze(-1)

        # Apply sigmoid to convert to probabilities in [0, 1]
        p = torch.sigmoid(logits_or_probs) if logits_or_probs.max() > 1.0 or logits_or_probs.min() < 0.0 else logits_or_probs

        B, T = p.shape
        device = p.device
        # [B]
        y = targets.squeeze().float().to(device)

        # 1. Top-K segment pooling
        k_val = min(self.k, T)
        # [B] (Mean of top-k segment probabilities for anomaly bags)
        topk_vals = torch.topk(p, k=k_val, dim=1).values.mean(dim=1)
        # [B] (Max segment probability for normal bags)
        max_vals = torch.max(p, dim=1).values

        # Numerical stability clamp
        topk_vals = torch.clamp(topk_vals, min=self.eps, max=1.0 - self.eps)
        max_vals = torch.clamp(max_vals, min=self.eps, max=1.0 - self.eps)

        # 2. Binary Cross-Entropy
        # [B]
        loss_pos = -torch.log(topk_vals)
        loss_neg = -torch.log(1.0 - max_vals)

        # [B]
        bce_per_sample = y * loss_pos + (1.0 - y) * loss_neg
        bce_loss = bce_per_sample.mean()

        # 3. Temporal Regularizations on Anomalous Bags
        anomaly_mask = (y > 0.5)
        if anomaly_mask.any():
            # [N_anom, T]
            p_anom = p[anomaly_mask]
            # [N_anom, T - 1] -> scalar
            diffs = p_anom[:, 1:] - p_anom[:, :-1]
            smoothness_loss = torch.mean(diffs ** 2)
            # Sparsity: penalize overly broad anomaly activations
            sparsity_loss = torch.mean(p_anom)
        else:
            smoothness_loss = torch.tensor(0.0, device=device)
            sparsity_loss = torch.tensor(0.0, device=device)

        # Total weighted objective
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

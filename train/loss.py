"""Multiple Instance Learning (MIL) Loss functions with zero code duplication."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn


class BaseMILLoss(nn.Module):
    """Core Multiple Instance Learning (MIL) loss with temporal regularization.

    Eliminates code duplication by centralizing:
    - Numerical stability clamping and sigmoid activation
    - Top-K segment-level pooling across time
    - Temporal smoothness regularization: sum((p_{t+1} - p_t)^2)
    - Temporal sparsity regularization: sum(p_t)
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

    def _ensure_probabilities(self, logits_or_probs: torch.Tensor) -> torch.Tensor:
        """Applies sigmoid if inputs are outside [0, 1] range."""
        if logits_or_probs.max() > 1.0 or logits_or_probs.min() < 0.0:
            return torch.sigmoid(logits_or_probs)
        return logits_or_probs

    def _compute_regularizations(self, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates temporal smoothness and sparsity penalties on segment probabilities.

        Args:
            p: Probabilities tensor of shape [B, T] or [B, T, C].

        Returns:
            Tuple of (smoothness_penalty, sparsity_penalty) scalars.
        """
        # Temporal smoothness: adjacent segment difference squared
        diff = p[:, 1:] - p[:, :-1]
        smoothness = torch.mean(torch.sum(diff ** 2, dim=1))

        # Sparsity: sum of anomaly segment probabilities
        sparsity = torch.mean(torch.sum(p, dim=1))

        return smoothness, sparsity

    def _pool_topk(self, p: torch.Tensor, dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs Top-K mean pooling for anomaly and Max pooling for normal bags."""
        T = p.shape[dim]
        # Sultani 2018 MIL formulation: dynamic k based on temporal length
        k_val = max(1, T // 16)

        # Anomaly score: mean of top-k segments
        topk_scores = torch.topk(p, k=k_val, dim=dim).values.mean(dim=dim)
        # Normal score: maximum segment
        max_scores = torch.max(p, dim=dim).values

        # Numerical stability clamp
        topk_scores = torch.clamp(topk_scores, min=self.eps, max=1.0 - self.eps)
        max_scores = torch.clamp(max_scores, min=self.eps, max=1.0 - self.eps)

        return topk_scores, max_scores


class BinaryLoss(BaseMILLoss):
    """Segment-level Binary Multiple Instance Learning (MIL) Ranking Loss.

    For anomaly videos (y=1), penalizes low Top-K segment probabilities.
    For normal videos (y=0), penalizes high maximum segment probability.
    """

    def forward(
        self,
        logits_or_probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Calculates Binary Top-K BCE loss with temporal smoothness and sparsity.

        Args:
            logits_or_probs: Tensor of shape [B, T, 1] or [B, T].
            targets: Tensor of shape [B, 1] or [B] in {0.0, 1.0}.

        Returns:
            Dict containing 'loss', 'bce', 'smoothness', and 'sparsity' scalar tensors.
        """
        if logits_or_probs.ndim == 3 and logits_or_probs.shape[-1] == 1:
            logits_or_probs = logits_or_probs.squeeze(-1)

        p = self._ensure_probabilities(logits_or_probs)  # [B, T]
        y = targets.view(-1).float().to(p.device)        # [B]

        # 1. Top-K pooling
        topk_scores, max_scores = self._pool_topk(p, dim=1)  # [B], [B]

        # 2. Binary Cross-Entropy
        loss_pos = -torch.log(topk_scores)
        loss_neg = -torch.log(1.0 - max_scores)
        bce = torch.mean(y * loss_pos + (1.0 - y) * loss_neg)

        # 3. Regularizations (applied to anomalous bags only)
        anom_mask = (y > 0)
        if anom_mask.any():
            smoothness, sparsity = self._compute_regularizations(p[anom_mask])
        else:
            smoothness = torch.tensor(0.0, device=p.device)
            sparsity = torch.tensor(0.0, device=p.device)
            
        total_loss = bce + (self.smoothness_weight * smoothness) + (self.sparsity_weight * sparsity)

        return {
            "loss": total_loss,
            "bce": bce,
            "smoothness": smoothness,
            "sparsity": sparsity,
        }


class MultiClassLoss(BaseMILLoss):
    """Segment-level Multi-Class Multiple Instance Learning (MIL) Loss.

    For each class in {1..C}, computes Top-K BCE against one-hot / multi-hot bag targets.
    """

    def __init__(
        self,
        k: int = 1,
        smoothness_weight: float = 0.0001,
        sparsity_weight: float = 0.0001,
        pos_weight: Optional[Union[float, torch.Tensor]] = 3.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__(
            k=k,
            smoothness_weight=smoothness_weight,
            sparsity_weight=sparsity_weight,
            eps=eps,
        )
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
        """Calculates Multi-Class Top-K BCE loss with temporal smoothness and sparsity.

        Args:
            logits_or_probs: Tensor of shape [B, T, C] (or [B, C]).
            targets: Tensor of shape [B, C] in {0.0, 1.0}.

        Returns:
            Dict containing 'loss', 'bce', 'smoothness', and 'sparsity' scalar tensors.
        """
        if logits_or_probs.ndim == 2:
            logits_or_probs = logits_or_probs.unsqueeze(1)

        p = self._ensure_probabilities(logits_or_probs)  # [B, T, C]
        y = targets.float().to(p.device)                 # [B, C]

        # 1. Top-K pooling per class across temporal segments (dim=1)
        topk_scores, max_scores = self._pool_topk(p, dim=1)  # [B, C], [B, C]

        # 2. Multi-label BCE
        loss_pos = -torch.log(topk_scores)
        loss_neg = -torch.log(1.0 - max_scores)

        pos_factor = self.pos_weight.to(p.device) if isinstance(self.pos_weight, torch.Tensor) else self.pos_weight
        bce = torch.mean((pos_factor * y * loss_pos) + ((1.0 - y) * loss_neg))

        # 3. Regularizations (applied to anomalous bags only)
        anom_mask = (y.sum(dim=-1) > 0)
        if anom_mask.any():
            smoothness, sparsity = self._compute_regularizations(p[anom_mask])
        else:
            smoothness = torch.tensor(0.0, device=p.device)
            sparsity = torch.tensor(0.0, device=p.device)

        total_loss = bce + (self.smoothness_weight * smoothness) + (self.sparsity_weight * sparsity)

        return {
            "loss": total_loss,
            "bce": bce,
            "smoothness": smoothness,
            "sparsity": sparsity,
        }

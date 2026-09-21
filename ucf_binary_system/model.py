"""Binary Anomaly Ranking Head for UCF-Crime Video Anomaly Detection.

Produces a single scalar anomaly probability per temporal segment via Sigmoid activation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryAnomalyHead(nn.Module):
    """Pyramidal Multiple Instance Learning (MIL) binary ranking head.

    Projects segment features [in_features -> 512 -> 256 -> 1] to produce
    a single segment-level anomaly score with Sigmoid activation.

    Args:
        in_features: Input feature dimension (default: 768 for Swin3D-T).
        hidden_dims: Intermediate layer dimensions (default: (512, 256)).
        dropout_rates: Dropout probabilities for hidden layers (default: (0.5, 0.3)).
        enable_noise: If True, adds training-time noise regularization.
    """

    def __init__(
        self,
        in_features: int = 768,
        hidden_dims: Sequence[int] = (512, 256),
        dropout_rates: Sequence[float] = (0.65, 0.55),
        enable_noise: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.num_classes = 1  # Strictly binary anomaly score
        self.hidden_dims = tuple(hidden_dims)
        self.dropout_rates = tuple(dropout_rates)
        self.enable_noise = bool(enable_noise)

        layers = []
        prev_dim = in_features
        for h_dim, drop in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            if drop > 0.0:
                layers.append(nn.Dropout(p=drop))
            prev_dim = h_dim

        # Final single-neuron projection
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits.

        Args:
            x: Feature tensor of shape [B, T, in_features] or [B, in_features].

        Returns:
            logits: Scalar logit predictions of shape [B, T, 1] or [B, 1].
        """
        # [B, T, in_features] -> [B, T, in_features] (Noise augmentation during training)
        if self.training and self.enable_noise:
            noise = (torch.rand_like(x) * 0.075) - (torch.rand_like(x) * 0.075)
            x = x + noise

        # [B, T, in_features] -> [B, T, in_features] (L2 feature normalization)
        x = F.normalize(x, p=2, dim=-1)

        # [B, T, in_features] -> [B, T, 1]
        logits = self.mlp(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning sigmoid-activated anomaly probabilities in [0, 1].

        Args:
            x: Feature tensor of shape [B, T, in_features] or [B, in_features].

        Returns:
            probs: Anomaly probabilities of shape [B, T, 1] or [B, 1].
        """
        # [B, T, 1] logits -> [B, T, 1] probabilities
        logits = self.forward(x)
        return torch.sigmoid(logits)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[BinaryAnomalyHead, Dict[str, Any]]:
        """Instantiates and loads a BinaryAnomalyHead from a saved checkpoint."""
        dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)

        # Clean prefix keys if necessary
        cleaned_dict = {}
        for k, v in state_dict.items():
            clean_k = k.replace("head.", "").replace("ranking_head.", "")
            cleaned_dict[clean_k] = v

        config = ckpt.get("config", {})
        in_features = config.get("in_features", cleaned_dict.get("mlp.0.weight", torch.empty(512, 768)).shape[1])
        hidden_dims = config.get("hidden_dims", (512, 256))

        model = cls(in_features=in_features, hidden_dims=hidden_dims).to(dev)
        model.load_state_dict(cleaned_dict, strict=True)
        model.eval()
        return model, ckpt

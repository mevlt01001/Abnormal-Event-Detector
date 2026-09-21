"""Deep Binary Anomaly Ranking Head for Combined UCF-Crime and XD-Violence Detection.

Features:
- Deeper 4-layer pyramidal MLP (in_features -> 512 -> 256 -> 128 -> 1).
- High dropout regularization (0.70, 0.60, 0.50) to prevent overfitting across 5,400+ videos.
- Elevated training-time noise injection (noise_std = 0.10).
- L2 feature normalization before projection.
- Sigmoid activation on logits for segment-level anomaly score in [0.0, 1.0].
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepBinaryAnomalyHead(nn.Module):
    """Deep Pyramidal Multiple Instance Learning (MIL) binary anomaly ranking head.

    Projects segment features [in_features -> 512 -> 256 -> 128 -> 1] to produce
    a single segment-level anomaly score with Sigmoid activation.

    Args:
        in_features: Input feature dimension (default: 768 for Swin3D-T).
        hidden_dims: Intermediate layer dimensions (default: (512, 256, 128)).
        dropout_rates: Dropout probabilities for hidden layers (default: (0.70, 0.60, 0.50)).
        noise_std: Scale of random uniform noise during training (default: 0.10).
        enable_noise: If True, adds training-time noise regularization.
    """

    def __init__(
        self,
        in_features: int = 768,
        hidden_dims: Sequence[int] = (512, 256, 128),
        dropout_rates: Sequence[float] = (0.70, 0.60, 0.50),
        noise_std: float = 0.10,
        enable_noise: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.num_classes = 1  # Strictly binary anomaly score
        self.hidden_dims = tuple(hidden_dims)
        self.dropout_rates = tuple(dropout_rates)
        self.noise_std = float(noise_std)
        self.enable_noise = bool(enable_noise)

        if len(self.hidden_dims) != len(self.dropout_rates):
            raise ValueError(
                f"Length mismatch: hidden_dims ({len(self.hidden_dims)}) vs dropout_rates ({len(self.dropout_rates)})"
            )

        layers: list[nn.Module] = []
        prev_dim = self.in_features
        for h_dim, drop in zip(self.hidden_dims, self.dropout_rates):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            if drop > 0.0:
                layers.append(nn.Dropout(p=drop))
            prev_dim = h_dim

        # Final single-neuron projection: 128 -> 1
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits.

        Args:
            x: Feature tensor of shape [B, T, in_features] or [B, in_features].

        Returns:
            logits: Scalar logit predictions of shape [B, T, 1] or [B, 1].
        """
        # [B, T, in_features] -> [B, T, in_features] (High-noise augmentation during training)
        if self.training and self.enable_noise and self.noise_std > 0.0:
            noise = (torch.rand_like(x) * self.noise_std) - (torch.rand_like(x) * self.noise_std)
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
        # [B, T, 1] logits -> [B, T, 1] probabilities in [0.0, 1.0]
        logits = self.forward(x)
        return torch.sigmoid(logits)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[DeepBinaryAnomalyHead, Dict[str, Any]]:
        """Instantiates and loads a DeepBinaryAnomalyHead from a saved checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
            device: Torch device for model placement.

        Returns:
            Tuple of (model, checkpoint_dict).
        """
        dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)

        # Clean prefix keys if saved from a wrapper or DataParallel
        cleaned_dict: Dict[str, Any] = {}
        for k, v in state_dict.items():
            clean_k = k
            for prefix in ("model.", "head.", "module."):
                if clean_k.startswith(prefix):
                    clean_k = clean_k[len(prefix):]
            cleaned_dict[clean_k] = v

        config = ckpt.get("config", {})
        in_features = config.get("in_features", 768)
        hidden_dims = config.get("hidden_dims", (512, 256, 128))
        dropout_rates = config.get("dropout_rates", (0.70, 0.60, 0.50))
        noise_std = config.get("noise_std", 0.10)

        model = cls(
            in_features=in_features,
            hidden_dims=hidden_dims,
            dropout_rates=dropout_rates,
            noise_std=noise_std,
            enable_noise=False,
        )
        model.load_state_dict(cleaned_dict)
        model.to(dev)
        model.eval()
        return model, ckpt

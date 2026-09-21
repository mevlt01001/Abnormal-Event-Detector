"""Pyramidal Anomaly Ranking Head for video segment anomaly classification."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyHead(nn.Module):
    """Pyramidal Multiple Instance Learning (MIL) classification head.

    Replaces legacy FC_head and SegmentRankingHead with a single, highly performant
    pyramidal MLP architecture (in_features -> 512 -> 256 -> num_classes) equipped
    with L2 feature normalization and training-time noise regularization.

    Args:
        in_features: Dimensionality of input feature vectors (default: 768 for Swin3D-T).
        num_classes: Number of target anomaly classes (default: 6 for macro-classes).
        hidden_dims: Tuple of intermediate layer dimensions (default: (512, 256)).
        dropout_rates: Tuple of dropout probabilities for each hidden layer (default: (0.5, 0.3)).
        enable_noise: If True, injects subtle noise during training to prevent overfitting.
    """

    def __init__(
        self,
        in_features: int = 768,
        num_classes: int = 6,
        hidden_dims: Sequence[int] = (512, 256),
        dropout_rates: Sequence[float] = (0.65, 0.55),
        enable_noise: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.enable_noise = bool(enable_noise)

        layers = []
        prev_dim = in_features

        for h_dim, drop in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            if drop > 0.0:
                layers.append(nn.Dropout(p=drop))
            prev_dim = h_dim

        # Final projection to class logits
        layers.append(nn.Linear(prev_dim, num_classes))

        self.hidden_dims = tuple(hidden_dims)
        self.dropout_rates = tuple(dropout_rates)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input feature tensor of shape [B, T, in_features] or [B, in_features].

        Returns:
            logits: Class logit predictions of shape [B, T, num_classes] or [B, num_classes].
        """
        # [B, T, in_features] -> [B, T, in_features] (Noise augmentation during training)
        if self.training and self.enable_noise:
            noise = (torch.rand_like(x) * 0.05) - (torch.rand_like(x) * 0.05)
            x = x + noise

        # [B, T, in_features] -> [B, T, in_features] (L2 normalization along feature dimension)
        x = F.normalize(x, p=2, dim=-1)

        # [B, T, in_features] -> [B, T, num_classes]
        logits = self.mlp(x)
        return logits

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[AnomalyHead, Dict[str, Any]]:
        """Dynamically instantiates and loads an AnomalyHead from a checkpoint.

        Reads model architecture (in_features, num_classes, hidden_dims) from checkpoint config
        or automatically infers them from state_dict weight shapes, eliminating all hardcoding.

        Args:
            checkpoint_path: Path to checkpoint .pt file.
            device: Target torch device.

        Returns:
            Tuple of (loaded_head_module, full_checkpoint_dict).
        """
        dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)

        # Clean prefix keys if saved from a wrapper (e.g. AnomalyDetector)
        cleaned_dict = {}
        for k, v in state_dict.items():
            clean_k = k.replace("head.", "").replace("ranking_head.", "").replace("classifier.", "")
            cleaned_dict[clean_k] = v

        # Read config if saved, else infer dynamically from weight tensor shapes
        config = ckpt.get("config", {})
        if "in_features" in config:
            in_features = int(config["in_features"])
        elif "mlp.0.weight" in cleaned_dict:
            in_features = cleaned_dict["mlp.0.weight"].shape[1]
        else:
            in_features = 768

        if "num_classes" in config:
            num_classes = int(config["num_classes"])
        else:
            # Find output layer weight shape
            out_keys = [k for k in cleaned_dict if k.endswith(".weight")]
            if out_keys:
                last_weight_key = sorted(out_keys, key=lambda x: [int(s) for s in x.split('.') if s.isdigit() or 0])[-1]
                num_classes = cleaned_dict[last_weight_key].shape[0]
            else:
                num_classes = 6

        hidden_dims = config.get("hidden_dims", (512, 256))

        head = cls(
            in_features=in_features,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
        ).to(dev)

        head.load_state_dict(cleaned_dict, strict=True)
        head.eval()
        return head, ckpt

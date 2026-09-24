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
        num_classes: Number of target anomaly classes (default: 4 for macro wrapper classes).
        hidden_dims: Tuple of intermediate layer dimensions (default: (512, 256)).
        dropout_rates: Tuple of dropout probabilities for each hidden layer (default: (0.6, 0.6)).
        enable_noise: If True, injects subtle noise during training to prevent overfitting.
    """

    def __init__(
        self,
        in_features: int = 768,
        num_classes: int = 4,
        hidden_dims: Sequence[int] = (512, 256),
        dropout_rates: Sequence[float] = (0.6, 0.6),
        enable_noise: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.enable_noise = bool(enable_noise)

        layers = [nn.LayerNorm(in_features)]
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
        # LayerNorm and MLP pass: [B, T, in_features] -> [B, T, num_classes]
        logits = self.mlp(x)
        return logits

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[Union[str, torch.device]] = None,
        num_classes: Optional[int] = None,
        dropout_rates: Optional[Sequence[float]] = None,
        hidden_dims: Optional[Sequence[int]] = None,
        in_features: Optional[int] = None,
    ) -> Tuple[AnomalyHead, Dict[str, Any]]:
        """Instantiates and loads an AnomalyHead directly from a checkpoint file.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
            device: Target torch device.
            num_classes: Optional user override for number of classes.
            dropout_rates: Optional user override for dropout probabilities.
            hidden_dims: Optional user override for hidden layer dimensions.
            in_features: Optional user override for input feature dimension.

        Returns:
            Tuple of (loaded_head_module, full_checkpoint_dict).
        """
        dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)

        if not isinstance(ckpt, dict) or "head_state_dict" not in ckpt:
            raise KeyError(
                f"Checkpoint '{checkpoint_path}' does not contain 'head_state_dict'. "
                "Ensure the checkpoint was saved using the canonical format."
            )
        state_dict = ckpt["head_state_dict"]

        # Read config if saved, else infer dynamically from weight tensor shapes
        config = ckpt.get("config", {})
        if "in_features" in config:
            cfg_in_features = int(config["in_features"])
        elif "mlp.0.weight" in state_dict:
            cfg_in_features = state_dict["mlp.0.weight"].shape[1]
        else:
            cfg_in_features = 768

        if "num_classes" in config:
            cfg_num_classes = int(config["num_classes"])
        else:
            # Find output layer weight shape
            out_keys = [k for k in state_dict if k.endswith(".weight")]
            if out_keys:
                last_weight_key = sorted(out_keys, key=lambda x: [int(s) for s in x.split('.') if s.isdigit() or 0])[-1]
                cfg_num_classes = state_dict[last_weight_key].shape[0]
            else:
                cfg_num_classes = 4

        cfg_hidden_dims = config.get("hidden_dims", (512, 256))
        cfg_dropout_rates = config.get("dropout_rates", ckpt.get("dropout_rates", (0.6, 0.6)))

        # User-provided overrides take absolute precedence over checkpoint
        final_in_features = int(in_features) if in_features is not None else cfg_in_features
        final_num_classes = int(num_classes) if num_classes is not None else cfg_num_classes
        final_hidden_dims = tuple(hidden_dims) if hidden_dims is not None else cfg_hidden_dims
        final_dropout_rates = tuple(dropout_rates) if dropout_rates is not None else cfg_dropout_rates

        head = cls(
            in_features=final_in_features,
            num_classes=final_num_classes,
            hidden_dims=final_hidden_dims,
            dropout_rates=final_dropout_rates,
        ).to(dev)

        # Load weights: if architecture was overridden, use non-strict matching to allow transfer
        strict_load = (final_num_classes == cfg_num_classes and final_in_features == cfg_in_features)
        head.load_state_dict(state_dict, strict=strict_load)
        head.eval()
        return head, ckpt

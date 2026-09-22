"""Unit tests for combined_binary_system module."""

from __future__ import annotations

import tempfile
import torch
import pytest

from combined_binary_system.model import DeepBinaryAnomalyHead
from combined_binary_system.dataset import (
    CombinedBinaryDataset,
    scan_combined_features,
    create_stratified_train_val_split,
)
from combined_binary_system.loss import BinaryMILLoss


def test_deep_binary_anomaly_head_forward():
    """Tests DeepBinaryAnomalyHead architecture, tensor shapes, and activations."""
    model = DeepBinaryAnomalyHead(
        in_features=768,
        hidden_dims=(512, 256, 128),
        dropout_rates=(0.70, 0.60, 0.50),
        noise_std=0.10,
        enable_noise=True,
    )

    # 1. Check layer counts
    # 4 Linear layers: 768->512, 512->256, 256->128, 128->1
    linear_layers = [m for m in model.mlp if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) == 4
    assert linear_layers[0].in_features == 768 and linear_layers[0].out_features == 512
    assert linear_layers[1].in_features == 512 and linear_layers[1].out_features == 256
    assert linear_layers[2].in_features == 256 and linear_layers[2].out_features == 128
    assert linear_layers[3].in_features == 128 and linear_layers[3].out_features == 1

    # 2. Check 3D batch shape: [B, T, D]
    x_3d = torch.randn(4, 32, 768)
    model.train()
    logits_train = model(x_3d)
    assert logits_train.shape == (4, 32, 1)

    model.eval()
    logits_eval = model(x_3d)
    assert logits_eval.shape == (4, 32, 1)

    probs = model.predict_proba(x_3d)
    assert probs.shape == (4, 32, 1)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # 3. Check 2D batch shape: [B, D]
    x_2d = torch.randn(4, 768)
    logits_2d = model(x_2d)
    assert logits_2d.shape == (4, 1)


def test_checkpoint_save_and_load():
    """Tests model checkpoint serialization and deserialization."""
    model = DeepBinaryAnomalyHead(
        in_features=768,
        hidden_dims=(512, 256, 128),
        dropout_rates=(0.70, 0.60, 0.50),
    )
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(
            {
                "epoch": 5,
                "model_state_dict": model.state_dict(),
                "val_roc_auc": 0.92,
                "config": {
                    "in_features": 768,
                    "hidden_dims": (512, 256, 128),
                    "dropout_rates": (0.70, 0.60, 0.50),
                    "noise_std": 0.10,
                },
            },
            f.name,
        )
        loaded_model, ckpt = DeepBinaryAnomalyHead.load_from_checkpoint(f.name, device="cpu")
        assert ckpt["epoch"] == 5
        assert ckpt["val_roc_auc"] == 0.92
        assert loaded_model.in_features == 768


def test_binary_mil_loss():
    """Tests BinaryMILLoss computation and dictionary keys."""
    loss_fn = BinaryMILLoss(k=3, smoothness_weight=0.0001, sparsity_weight=0.0001)

    logits = torch.randn(4, 32, 1, requires_grad=True)
    targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

    loss_dict = loss_fn(logits, targets)
    assert "total" in loss_dict
    assert "bce" in loss_dict
    assert "smoothness" in loss_dict
    assert "sparsity" in loss_dict
    assert loss_dict["total"].ndim == 0
    assert loss_dict["total"].item() > 0.0

    # Test backward pass
    loss_dict["total"].backward()
    assert logits.grad is not None


def test_dataset_scanning_and_split():
    """Tests feature scanning on swin3d_t_extracted_features and stratified splitting."""
    samples = scan_combined_features("swin3d_t_extracted_features")
    assert len(samples) == 5459

    n_norm = sum(1 for _, lbl, _ in samples if lbl == 0.0)
    n_anom = sum(1 for _, lbl, _ in samples if lbl == 1.0)
    assert n_norm == 2847
    assert n_anom == 2612

    train_s, val_s = create_stratified_train_val_split(samples, val_ratio=0.20, seed=42)
    assert len(train_s) + len(val_s) == 5459
    assert len(set(s[0] for s in train_s).intersection(set(s[0] for s in val_s))) == 0


def test_feature_extractor_and_analyzer_parameters():
    """Tests clip_size, overlap, and stride parameter propagation."""
    from combined_binary_system.infer import CombinedBinaryAnalyzer
    from core.feature_extractor import FeatureExtractor
    import torch.nn as nn

    dummy_model = nn.Identity()
    analyzer = CombinedBinaryAnalyzer(
        model=dummy_model,
        target_fps=20.0,
        clip_size=16,
        overlap=8.0,
    )
    assert analyzer.processor.target_fps == 20.0
    assert analyzer.processor.clip_size == 16
    assert analyzer.processor.stride == 8
    assert abs(analyzer.processor.overlap_ratio - 0.5) < 1e-4

    dummy_backbone = nn.Identity()
    extractor = FeatureExtractor(
        backbone=dummy_backbone,
        target_fps=15.0,
        clip_size=32,
        overlap=16.0,
    )
    assert extractor.processor.target_fps == 15.0
    assert extractor.processor.clip_size == 32
    assert extractor.processor.stride == 16
    assert abs(extractor.processor.overlap_ratio - 0.5) < 1e-4


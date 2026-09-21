"""Unit tests for UCF-Crime isolated binary anomaly detection system."""

import os
import shutil
import tempfile
import pytest
import torch

from ucf_binary_system.dataset import (
    UCFBinaryDataset,
    create_stratified_train_val_split,
    load_feature_tensor,
    scan_ucf_crime_features,
)
from ucf_binary_system.infer import UCFBinaryAnalyzer
from ucf_binary_system.loss import BinaryMILLoss
from ucf_binary_system.model import BinaryAnomalyHead


def test_ucf_binary_model():
    head = BinaryAnomalyHead(in_features=768)
    feats = torch.randn(4, 32, 768)

    # Forward logits
    logits = head(feats)
    assert logits.shape == (4, 32, 1)

    # Probabilities in [0, 1]
    probs = head.predict_proba(feats)
    assert probs.shape == (4, 32, 1)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_ucf_binary_loss():
    loss_fn = BinaryMILLoss(k=3)
    # [B=4, T=32, 1]
    logits = torch.randn(4, 32, 1, requires_grad=True)
    # 2 positive (anomaly), 2 negative (normal)
    targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

    losses = loss_fn(logits, targets)
    assert "total" in losses
    assert "bce" in losses
    assert "smoothness" in losses
    assert "sparsity" in losses
    assert losses["total"].item() > 0.0

    losses["total"].backward()
    assert logits.grad is not None


def test_ucf_binary_checkpoint_factory(tmp_path):
    head = BinaryAnomalyHead(in_features=512, hidden_dims=(256, 128))
    ckpt_file = tmp_path / "test_binary_model.pt"
    torch.save(
        {
            "model_state_dict": head.state_dict(),
            "config": {"in_features": 512, "hidden_dims": (256, 128)},
        },
        ckpt_file,
    )

    loaded_head, _ = BinaryAnomalyHead.load_from_checkpoint(str(ckpt_file), device="cpu")
    assert loaded_head.in_features == 512
    assert loaded_head.hidden_dims == (256, 128)
    out = loaded_head.predict_proba(torch.randn(2, 16, 512))
    assert out.shape == (2, 16, 1)


def test_ucf_binary_analyzer_plot(tmp_path):
    head = BinaryAnomalyHead(in_features=768)
    analyzer = UCFBinaryAnalyzer(model=head, device="cpu", target_fps=20.0)

    # Mock features: [32, 768]
    feats = torch.randn(32, 768)
    analysis = analyzer.analyze_features(
        features=feats,
        video_duration_sec=30.0,
        threshold=0.35,
        video_name="Test_Video",
    )

    assert "time_axis" in analysis
    assert "anomaly_scores" in analysis
    assert "is_anomaly" in analysis
    assert len(analysis["anomaly_scores"]) > 0

    plot_path = str(tmp_path / "test_timeline.png")
    saved = analyzer.render_timeline_plot(analysis, save_path=plot_path)
    assert os.path.isfile(saved)

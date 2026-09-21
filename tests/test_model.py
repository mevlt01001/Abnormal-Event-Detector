"""Unit tests for AnomalyHead, AnomalyDetector, and video backbones."""

import pytest
import torch
import torch.nn as nn

from model.backbone import get_backbone_dim
from model.head import AnomalyHead
from model.model import AnomalyDetector


def test_anomaly_head_forward_and_backward():
    in_features = 768
    num_classes = 6
    head = AnomalyHead(in_features=in_features, num_classes=num_classes)

    # 3D Sequence Input: [batch_size=4, num_segments=32, in_features=768]
    feats = torch.randn(4, 32, in_features, requires_grad=True)

    head.train()
    logits_train = head(feats)
    # [B, T, num_classes]
    assert logits_train.shape == (4, 32, num_classes)

    # Test backward pass
    loss = logits_train.sum()
    loss.backward()
    assert feats.grad is not None
    assert not torch.isnan(feats.grad).any()

    # Eval mode
    head.eval()
    with torch.no_grad():
        logits_eval = head(feats)
        assert logits_eval.shape == (4, 32, num_classes)


def test_anomaly_head_2d_input():
    head = AnomalyHead(in_features=768, num_classes=6)
    feats = torch.randn(8, 768)
    logits = head(feats)
    assert logits.shape == (8, 6)


def test_anomaly_detector_forward_features():
    detector = AnomalyDetector(backbone_name=None, in_features=768, num_classes=6)
    feats = torch.randn(2, 32, 768)
    out = detector(feats)
    assert out.shape == (2, 32, 6)


def test_anomaly_head_checkpoint_factory(tmp_path):
    # Test saving with arbitrary custom dimensions (e.g. 512 in_features, 3 classes)
    ckpt_file = tmp_path / "custom_head.pt"
    orig_head = AnomalyHead(in_features=512, num_classes=3, hidden_dims=(256, 128))
    torch.save(
        {
            "model_state_dict": orig_head.state_dict(),
            "config": {
                "in_features": 512,
                "num_classes": 3,
                "class_list": ["CatA", "CatB", "CatC"],
                "hidden_dims": (256, 128),
            },
        },
        ckpt_file,
    )

    loaded_head, ckpt = AnomalyHead.load_from_checkpoint(str(ckpt_file), device="cpu")
    assert loaded_head.in_features == 512
    assert loaded_head.num_classes == 3
    assert loaded_head.hidden_dims == (256, 128)
    assert ckpt["config"]["class_list"] == ["CatA", "CatB", "CatC"]

    # Forward test with custom shape
    custom_in = torch.randn(2, 16, 512)
    out = loaded_head(custom_in)
    assert out.shape == (2, 16, 3)


def test_get_backbone_dim():
    assert get_backbone_dim("swin3d_t") == 768
    assert get_backbone_dim("r3d_18") == 512


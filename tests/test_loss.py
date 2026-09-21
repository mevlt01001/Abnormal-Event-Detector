"""Unit tests for ClassAwareMILLoss."""

import pytest
import torch
from loss.mil_loss import ClassAwareMILLoss


def test_class_aware_mil_loss():
    criterion = ClassAwareMILLoss(k=1, smoothness_weight=0.001, sparsity_weight=0.001)

    # [B=4, T=32, num_classes=6]
    logits = torch.randn(4, 32, 6, requires_grad=True)

    # 4 bags: 2 normal (all zeros), 2 anomaly (one-hot)
    targets = torch.zeros(4, 6)
    targets[1, 0] = 1.0  # Bag 1: Class 0
    targets[3, 4] = 1.0  # Bag 3: Class 4

    loss_dict = criterion(logits, targets)

    assert "total" in loss_dict
    assert "bce" in loss_dict
    assert "smoothness" in loss_dict
    assert "sparsity" in loss_dict

    total = loss_dict["total"]
    assert total.item() > 0.0
    assert not torch.isnan(total)

    total.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()

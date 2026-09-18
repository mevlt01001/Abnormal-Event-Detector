import pytest
import torch

from loss.mil_loss import VideoAnomalyLoss


def test_loss_components_and_keys():
    criterion = VideoAnomalyLoss(smoothness_weight=0.01, sparsity_weight=0.001)

    y_anom = torch.rand(32, 1, requires_grad=True)
    y_norm = torch.rand(32, 1, requires_grad=True)

    loss_dict = criterion(y_anom, y_norm)

    assert "total" in loss_dict
    assert "hinge" in loss_dict
    assert "smoothness" in loss_dict
    assert "sparsity" in loss_dict

    total = loss_dict["total"]
    total.backward()

    assert y_anom.grad is not None
    assert y_norm.grad is not None


def test_hinge_loss_behavior():
    criterion = VideoAnomalyLoss(smoothness_weight=0.0, sparsity_weight=0.0)

    # Perfect ranking: anomaly max is 1.0, normal max is 0.0 -> Hinge = relu(1 - 1 + 0) = 0
    anom_perfect = torch.ones(32, 1)
    norm_perfect = torch.zeros(32, 1)
    loss_perfect = criterion(anom_perfect, norm_perfect)
    assert torch.isclose(loss_perfect["hinge"], torch.tensor(0.0), atol=1e-6)

    # Inverted ranking: anomaly max is 0.0, normal max is 1.0 -> Hinge = relu(1 - 0 + 1) = 2.0
    anom_inverted = torch.zeros(32, 1)
    norm_inverted = torch.ones(32, 1)
    loss_inverted = criterion(anom_inverted, norm_inverted)
    assert torch.isclose(loss_inverted["hinge"], torch.tensor(2.0), atol=1e-6)


def test_smoothness_and_sparsity_behavior():
    criterion = VideoAnomalyLoss(smoothness_weight=1.0, sparsity_weight=1.0)

    # Smooth constant signal
    anom_constant = torch.full((16, 1), 0.5)
    norm = torch.zeros(16, 1)
    loss_const = criterion(anom_constant, norm)
    assert torch.isclose(loss_const["smoothness"], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(loss_const["sparsity"], torch.tensor(8.0), atol=1e-6)

    # Oscillating signal (high roughness)
    anom_osc = torch.tensor([0.0, 1.0] * 8).unsqueeze(1)
    loss_osc = criterion(anom_osc, norm)
    assert loss_osc["smoothness"].item() > 5.0


def test_batched_inputs():
    criterion = VideoAnomalyLoss()

    # Batch of 4 videos, each with 32 segments
    y_anom = torch.rand(4, 32, 1)
    y_norm = torch.rand(4, 32, 1)

    loss_dict = criterion(y_anom, y_norm)
    assert loss_dict["total"].ndim == 0  # Scalar loss
    assert loss_dict["total"].item() > 0.0

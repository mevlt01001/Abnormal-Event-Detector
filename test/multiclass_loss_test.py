import pytest
import torch
from loss.multiclass_mil_loss import MultiClassMILLoss


def test_multiclass_loss_keys():
    loss_fn = MultiClassMILLoss(k=1)
    scores = torch.randn(2, 32, 6, requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    losses = loss_fn(scores, target)

    assert set(losses.keys()) == {"total", "bce", "smoothness", "sparsity"}
    assert losses["total"].dim() == 0
    assert losses["bce"].dim() == 0
    assert losses["smoothness"].dim() == 0
    assert losses["sparsity"].dim() == 0


def test_multiclass_loss_backprop():
    loss_fn = MultiClassMILLoss(k=2)
    scores = torch.randn(4, 32, 6, requires_grad=True)
    target = torch.randint(0, 2, (4, 6)).float()
    losses = loss_fn(scores, target)

    losses["total"].backward()
    assert scores.grad is not None
    assert not torch.isnan(scores.grad).any()
    assert not torch.isinf(scores.grad).any()


def test_multiclass_loss_zero_target_normal_video():
    loss_fn = MultiClassMILLoss(k=1)
    target = torch.zeros(1, 6)

    # When scores are strongly negative (predicted prob ~ 0), BCE should be very low
    good_scores = torch.full((1, 32, 6), -10.0)
    low_loss = loss_fn(good_scores, target)["bce"]

    # When scores are strongly positive (predicted prob ~ 1), BCE should be high
    bad_scores = torch.full((1, 32, 6), 10.0)
    high_loss = loss_fn(bad_scores, target)["bce"]

    assert low_loss < high_loss
    assert low_loss < 0.01
    assert high_loss > 5.0


def test_multiclass_loss_k_parameter():
    loss_fn_k1 = MultiClassMILLoss(k=1)
    loss_fn_k3 = MultiClassMILLoss(k=3)

    scores = torch.zeros(1, 32, 1)
    # Set top 3 values to 10.0, 8.0, 6.0
    scores[0, 0, 0] = 10.0
    scores[0, 1, 0] = 8.0
    scores[0, 2, 0] = 6.0
    target = torch.tensor([[1.0]])

    out_k1 = loss_fn_k1(scores, target)
    out_k3 = loss_fn_k3(scores, target)

    # In k1, aggregated score is 10.0
    # In k3, aggregated score is (10 + 8 + 6) / 3 = 8.0
    # Higher logit for target=1 yields lower BCE
    assert out_k1["bce"] < out_k3["bce"]


def test_multiclass_loss_2d_and_3d_shapes():
    loss_fn = MultiClassMILLoss(k=1)
    scores_2d = torch.randn(32, 6)
    target_1d = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    loss_2d = loss_fn(scores_2d, target_1d)
    loss_3d = loss_fn(scores_2d.unsqueeze(0), target_1d.unsqueeze(0))

    assert torch.isclose(loss_2d["total"], loss_3d["total"])

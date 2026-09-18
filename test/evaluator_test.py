import os
import tempfile
import numpy as np
import pytest
import torch

from model.model import FC_head
from scripts.evaluator import (
    compute_auc_roc,
    compute_average_precision,
    compute_temporal_iou,
    extract_intervals_from_scores,
    evaluate_binary_model,
    evaluate_multiclass_model,
)


def test_compute_auc_roc():
    # Single class edge case
    assert compute_auc_roc(np.zeros(5), np.random.rand(5)) == 0.5

    # Perfect prediction
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert np.isclose(compute_auc_roc(y_true, y_score), 1.0)

    # Inverted prediction
    y_score_inv = np.array([0.9, 0.8, 0.2, 0.1])
    assert np.isclose(compute_auc_roc(y_true, y_score_inv), 0.0)


def test_compute_temporal_iou():
    # Identical
    assert np.isclose(compute_temporal_iou((10, 20), (10, 20)), 1.0)
    # Non-overlapping
    assert compute_temporal_iou((10, 20), (30, 40)) == 0.0
    # Half-overlap: [10, 30] and [20, 40] -> intersection 10, union 30 -> 1/3
    assert np.isclose(compute_temporal_iou((10, 30), (20, 40)), 10.0 / 30.0)


def test_extract_intervals():
    scores = np.array([0.1, 0.8, 0.9, 0.2, 0.7, 0.1])
    intervals = extract_intervals_from_scores(scores, threshold=0.5)
    assert intervals == [(1, 3), (4, 5)]


def test_evaluate_binary_and_multiclass_mock():
    with tempfile.TemporaryDirectory() as tmpdir:
        norm_dir = os.path.join(tmpdir, "normal")
        anom_dir = os.path.join(tmpdir, "anomal")
        os.makedirs(norm_dir)
        os.makedirs(anom_dir)

        feat_dim = 32
        torch.save(torch.randn(32, feat_dim), os.path.join(norm_dir, "norm1__label_A.pt"))
        torch.save(torch.randn(32, feat_dim), os.path.join(anom_dir, "anom1__label_B1-0-0.pt"))

        # Create dummy annotation file
        ann_file = os.path.join(tmpdir, "annotations.txt")
        with open(ann_file, "w") as f:
            f.write("anom1__label_B1-0-0 100 200\n")

        # Binary model evaluation
        binary_head = FC_head(in_features=feat_dim, num_classes=1, use_sigmoid=True)
        bin_results = evaluate_binary_model(
            model=binary_head,
            features_dir=tmpdir,
            ann_file=ann_file,
            device="cpu",
        )
        assert "frame_auc" in bin_results
        assert "video_auc" in bin_results
        assert "tiou_recalls" in bin_results

        # Multiclass model evaluation
        multi_head = FC_head(in_features=feat_dim, num_classes=6, use_sigmoid=False)
        multi_results = evaluate_multiclass_model(
            model=multi_head,
            features_dir=tmpdir,
            device="cpu",
        )
        assert "mAP" in multi_results
        assert "per_class_ap" in multi_results
        assert len(multi_results["per_class_ap"]) == 6

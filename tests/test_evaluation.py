"""Unit tests for Evaluator, academic metrics, and threshold sweep."""

import os
import tempfile
import numpy as np
import torch
import pytest

from evaluation.metrics import (
    compute_roc_auc,
    compute_pr_auc,
    compute_multiclass_map,
    compute_threshold_sweep,
)
from evaluation.evaluator import Evaluator
from model import AnomalyHead


def test_academic_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    auc = compute_roc_auc(y_true, y_score)
    assert auc == 1.0

    pr = compute_pr_auc(y_true, y_score)
    assert pr == 1.0


def test_multiclass_map():
    y_true = np.array([[1, 0], [0, 1], [0, 0], [1, 0]])
    y_score = np.array([[0.9, 0.1], [0.2, 0.8], [0.1, 0.1], [0.8, 0.2]])

    res = compute_multiclass_map(y_true, y_score, class_names=["ClassA", "ClassB"])
    assert "mAP" in res
    assert res["mAP"] > 0.9


def test_threshold_sweep_calculation():
    video_scores = {
        "video_1": np.array([0.1, 0.2, 0.85, 0.9, 0.88, 0.2, 0.1]),
        "video_2": np.array([0.05, 0.05, 0.08, 0.02]),
    }
    gt_intervals = {
        "video_1": [(2.0, 5.0)],
        "video_2": [],
    }
    video_durations = {
        "video_1": 7.0,
        "video_2": 4.0,
    }

    res = compute_threshold_sweep(
        video_scores_dict=video_scores,
        gt_intervals_dict=gt_intervals,
        video_durations_dict=video_durations,
        thresholds=[0.3, 0.5, 0.7],
        iou_threshold=0.3,
    )
    assert "best_threshold" in res
    assert "best_f1" in res
    assert "markdown_report" in res
    assert "multi_tiou_summary" in res
    assert len(res["multi_tiou_summary"]) == 4
    assert res["best_f1"] > 0.0

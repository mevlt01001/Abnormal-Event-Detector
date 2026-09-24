"""Unit tests for datasets, taxonomy, and ground truth annotations."""

import os
import torch
from torch.utils.data import Subset
import pytest

from data.dataset import (
    BaseVideoAnomalyDataset,
    UCF_Dataset,
    XDV_Dataset,
    XDV_UCF_Combined_Dataset,
    get_dataset,
)
from data.taxonomy import WRAPPER_CLASSES, UCF_CLASSES, XDV_CLASSES


def test_base_dataset_mode_switching():
    paths = ["dummy_1.pt", "dummy_2.pt"]
    labels = ["Normal", "Fighting"]
    classes = ["Fighting", "Shooting"]

    ds = BaseVideoAnomalyDataset(
        file_paths=paths,
        labels=labels,
        classes=classes,
        mode="multiclass",
        fps=25.0,
        clip_size=16,
    )
    assert ds.fps == 25.0
    assert ds.clip_size == 16
    assert ds.mode == "multiclass"

    # Multiclass target
    t_normal = ds._get_target("Normal")
    assert torch.equal(t_normal, torch.zeros(2))
    t_ano = ds._get_target("Fighting")
    assert torch.equal(t_ano, torch.tensor([1.0, 0.0]))

    # Switch to binary
    ds.set_mode("binary")
    assert ds.mode == "binary"
    assert ds._get_target("Normal").item() == 0.0
    assert ds._get_target("Fighting").item() == 1.0


def test_real_dataset_instantiation():
    # Test split loading
    test_ds = get_dataset("combined", split="test")
    assert len(test_ds) > 0
    assert test_ds.num_classes == 4
    assert test_ds.fps == 20.0
    assert test_ds.clip_size == 16
    assert test_ds.num_segments == 32

    # Check ground truth intervals loaded from annotation files
    assert isinstance(test_ds.ground_truth_intervals, dict)
    assert len(test_ds.ground_truth_intervals) > 0

    # User configured overrides
    custom_ds = get_dataset("combined", split="test", fps=30.0, clip_size=32)
    assert custom_ds.fps == 30.0
    assert custom_ds.clip_size == 32


def test_subset_compatibility():
    test_ds = get_dataset("combined", split="test")
    subset = Subset(test_ds, indices=[0, 1, 2])
    assert len(subset) == 3
    # Verify underlying dataset attributes are accessible
    assert subset.dataset.classes == WRAPPER_CLASSES


def test_combined_dataset_exact_match_mode():
    from data.taxonomy import EXACT_MATCH_CLASSES

    # 1. Instantiation in exact_match mode
    ds = XDV_UCF_Combined_Dataset(split="test", mode="exact_match")
    assert ds.mode == "exact_match"
    assert ds.num_classes == 5
    assert ds.classes == EXACT_MATCH_CLASSES
    assert ds.classes == ["Abuse", "Explosion", "Fighting", "RoadAccidents", "Shooting"]

    # 2. Check that all samples belong strictly to exact match classes + Normal
    valid_set = set(EXACT_MATCH_CLASSES) | {"Normal"}
    for lbl in ds.labels:
        assert lbl in valid_set

    # Filtered sample count must be smaller than full combined
    full_ds = XDV_UCF_Combined_Dataset(split="test", mode="multiclass")
    assert len(ds) < len(full_ds)
    assert len(ds) == 878

    # 3. Target vector verification
    t_normal = ds._get_target("Normal")
    assert torch.equal(t_normal, torch.zeros(5))

    t_fighting = ds._get_target("Fighting")
    expected_fighting = torch.zeros(5)
    expected_fighting[ds.class_to_idx["Fighting"]] = 1.0
    assert torch.equal(t_fighting, expected_fighting)

    # 4. Dynamic mode switching
    ds.set_mode("multiclass")
    assert ds.mode == "multiclass"
    assert ds.num_classes == 4
    assert len(ds) == len(full_ds)

    ds.set_mode("exact_match")
    assert ds.mode == "exact_match"
    assert ds.num_classes == 5
    assert len(ds) == 878

    ds.set_mode("binary")
    assert ds.mode == "binary"
    assert ds.num_classes == 1
    assert ds.classes == ["Anomaly"]
    assert len(ds) == len(full_ds)


def test_datasets_binary_mode_single_class():
    """Verifies UCF, XDV, and Combined datasets all output num_classes=1 and classes=['Anomaly'] in binary mode."""
    for cls in [UCF_Dataset, XDV_Dataset, XDV_UCF_Combined_Dataset]:
        ds = cls(split="test", mode="binary")
        assert ds.num_classes == 1
        assert ds.classes == ["Anomaly"]
        assert ds.class_to_idx == {"Anomaly": 0}

        # Check target vector is a scalar/single float
        t_norm = ds._get_target("Normal")
        assert float(t_norm) == 0.0
        t_ano = ds._get_target("AnyAnomaly")
        assert float(t_ano) == 1.0

        # Switch to multiclass and back
        ds.set_mode("multiclass")
        assert ds.num_classes > 1
        ds.set_mode("binary")
        assert ds.num_classes == 1
        assert ds.classes == ["Anomaly"]



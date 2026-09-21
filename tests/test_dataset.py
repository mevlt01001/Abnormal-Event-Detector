"""Unit tests for ClassAwareVideoDataset and Stratified K-Fold splitter."""

import os
import shutil
import tempfile
import pytest
import torch

from data.dataset import (
    DEFAULT_MACRO_CLASSES,
    ClassAwareVideoDataset,
    create_stratified_kfold_splits,
    load_feature_tensor,
)


def test_dataset_loading_and_stratification():
    temp_dir = tempfile.mkdtemp(prefix="dataset_test_")
    try:
        # Create mock folders for 6 macro classes + Normal
        all_files = []
        for c in DEFAULT_MACRO_CLASSES + ["Normal"]:
            c_dir = os.path.join(temp_dir, c)
            os.makedirs(c_dir, exist_ok=True)
            for i in range(8):
                fpath = os.path.join(c_dir, f"video_{i:03d}.pt")
                torch.save(torch.randn(32, 768), fpath)
                all_files.append(fpath)

        dataset = ClassAwareVideoDataset(all_files, class_list=DEFAULT_MACRO_CLASSES)
        assert len(dataset) == len(all_files)

        feats, target, name = dataset[0]
        assert feats.shape == (32, 768)
        assert target.shape == (len(DEFAULT_MACRO_CLASSES),)

        # Stratified 4-Fold split
        splits = create_stratified_kfold_splits(all_files, num_folds=4, seed=42, class_list=DEFAULT_MACRO_CLASSES)
        assert len(splits) == 4
        for train_files, val_files in splits:
            assert len(train_files) + len(val_files) == len(all_files)
            assert len(set(train_files).intersection(set(val_files))) == 0

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

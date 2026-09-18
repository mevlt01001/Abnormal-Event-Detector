import os
import tempfile
import pytest
import torch
from torch.utils.data import DataLoader

from data.multiclass_dataset import MultiClassFeatureDataset, create_multiclass_kfold_splits
from utils.annotation_parser import XD_CLASSES


def test_multiclass_dataset_loading():
    with tempfile.TemporaryDirectory() as tmpdir:
        norm_dir = os.path.join(tmpdir, "normal")
        anom_dir = os.path.join(tmpdir, "anomal")
        os.makedirs(norm_dir)
        os.makedirs(anom_dir)

        # Create dummy normal feature
        norm_file = os.path.join(norm_dir, "video1__label_A.pt")
        torch.save(torch.randn(32, 512), norm_file)

        # Create dummy anomalous feature with B1
        anom_file1 = os.path.join(anom_dir, "video2__label_B1-0-0.pt")
        torch.save(torch.randn(32, 512), anom_file1)

        # Create dummy anomalous feature with B2 and G
        anom_file2 = os.path.join(anom_dir, "video3__label_B2-G-0.pt")
        torch.save(torch.randn(32, 512), anom_file2)

        # Test from normal_dir and anomal_dir
        ds = MultiClassFeatureDataset(normal_dir=norm_dir, anomal_dir=anom_dir)
        assert len(ds) == 3

        # Test DataLoader
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        for feats, targets, names in loader:
            assert feats.shape[1:] == (32, 512)
            assert targets.shape[1] == len(XD_CLASSES)
            assert len(names) == feats.shape[0]
            break

        # Test from features_dir
        ds_parent = MultiClassFeatureDataset(features_dir=tmpdir)
        assert len(ds_parent) == 3


def test_create_multiclass_kfold_splits():
    normal_files = [f"norm_{i}__label_A.pt" for i in range(30)]
    anomal_files = [f"anom_{i}__label_B1-0-0.pt" for i in range(50)]
    all_files = normal_files + anomal_files

    num_folds = 5
    folds = create_multiclass_kfold_splits(all_files, num_folds=num_folds, seed=123)

    assert len(folds) == num_folds
    for train_files, val_files in folds:
        assert len(train_files) + len(val_files) == 80
        assert len(val_files) == 16  # 6 normal + 10 anomalous
        assert len(train_files) == 64
        # Disjoint check
        assert set(train_files).isdisjoint(set(val_files))

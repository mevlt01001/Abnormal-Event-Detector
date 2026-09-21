"""Unified Dataset and Stratified K-Fold Splitter for Class-Aware Video Anomaly Detection."""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from .taxonomy import DEFAULT_MACRO_CLASSES, load_dataset_taxonomy


def load_feature_tensor(file_path: str) -> torch.Tensor:
    """Loads a feature tensor from a .pt file safely.

    Handles both raw tensor dumps and dictionary formats (e.g. {'feats': tensor}).
    """
    try:
        data = torch.load(file_path, weights_only=True, map_location="cpu")
    except Exception:
        data = torch.load(file_path, weights_only=False, map_location="cpu")

    if isinstance(data, dict):
        if "feats" in data:
            feats = data["feats"]
        elif "features" in data:
            feats = data["features"]
        else:
            first = next((v for v in data.values() if isinstance(v, torch.Tensor)), None)
            if first is not None:
                feats = first
            else:
                raise ValueError(f"No tensor found in dictionary loaded from {file_path}")
    elif isinstance(data, torch.Tensor):
        feats = data
    else:
        raise TypeError(f"Unexpected data type loaded from {file_path}: {type(data)}")

    # [32, D] float32 tensor
    return feats.float()


def scan_feature_files(root_dir: str = "data/unified_features") -> List[str]:
    """Recursively scans a directory for valid .pt feature files."""
    collected = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".pt") and not f.startswith("."):
                collected.append(os.path.join(dirpath, f))
    return sorted(collected)


class ClassAwareVideoDataset(Dataset):
    """Class-Aware Video Feature Dataset.

    Returns segment representations and one-hot multi-class target vectors:
    - Normal videos have zero vector: [0, 0, ..., 0]
    - Anomaly videos have one-hot target: [0, 1, ..., 0] corresponding to their macro class.

    Args:
        file_paths: List of absolute or relative paths to .pt feature files.
        class_list: Ordered list of anomaly class names (default: DEFAULT_MACRO_CLASSES).
    """

    def __init__(
        self,
        file_paths: Sequence[str],
        class_list: Optional[Sequence[str]] = None,
    ) -> None:
        self.file_paths = list(sorted(file_paths))
        if class_list is not None:
            self.class_list = list(class_list)
        elif self.file_paths:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(self.file_paths[0])))
            tax = load_dataset_taxonomy(base_dir)
            self.class_list = tax["anomaly_classes"]
        else:
            self.class_list = list(DEFAULT_MACRO_CLASSES)

        self.class_to_idx = {c: idx for idx, c in enumerate(self.class_list)}
        self.num_classes = len(self.class_list)

    def __len__(self) -> int:
        return len(self.file_paths)

    def _determine_label(self, path: str) -> torch.Tensor:
        """Determines one-hot target vector from folder structure or filename."""
        parent_dir = os.path.basename(os.path.dirname(path))
        target = torch.zeros(self.num_classes, dtype=torch.float32)

        if parent_dir == "Normal" or "label_A" in os.path.basename(path) or "Normal_Video" in path:
            # [num_classes] all zeros for normal video
            return target

        if parent_dir in self.class_to_idx:
            # [num_classes] one-hot vector
            target[self.class_to_idx[parent_dir]] = 1.0
            return target

        # Check filename matching for macro classes
        fname = os.path.basename(path)
        for c_name, c_idx in self.class_to_idx.items():
            if c_name.lower() in fname.lower():
                target[c_idx] = 1.0
                return target

        # Fallback: treat as normal if unspecified
        return target

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Loads and returns a sample.

        Returns:
            features: Tensor of shape [32, feature_dim]
            target: One-hot target tensor of shape [num_classes]
            video_name: Basename string of the video
        """
        path = self.file_paths[idx]
        # [32, feature_dim]
        features = load_feature_tensor(path)
        # [num_classes]
        target = self._determine_label(path)
        video_name = os.path.splitext(os.path.basename(path))[0]
        return features, target, video_name


def create_stratified_kfold_splits(
    file_paths: Sequence[str],
    num_folds: int = 8,
    seed: int = 42,
    class_list: Optional[Sequence[str]] = None,
) -> List[Tuple[List[str], List[str]]]:
    """Creates balanced, stratified K-Fold cross-validation splits.

    Ensures each fold contains a proportional representation of Normal videos
    and all 6 anomaly macro-classes.

    Args:
        file_paths: List of all feature file paths.
        num_folds: Number of folds (default: 8).
        seed: Random seed for reproducibility.
        class_list: Macro class names.

    Returns:
        List of (train_file_paths, val_file_paths) tuples for each fold.
    """
    if class_list is not None:
        classes = list(class_list)
    elif file_paths:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(file_paths[0])))
        tax = load_dataset_taxonomy(base_dir)
        classes = tax["anomaly_classes"]
    else:
        classes = list(DEFAULT_MACRO_CLASSES)

    rng = random.Random(seed)

    # Bucket files by category
    buckets: Dict[str, List[str]] = {c: [] for c in classes}
    buckets["Normal"] = []

    for path in file_paths:
        parent = os.path.basename(os.path.dirname(path))
        if parent in buckets:
            buckets[parent].append(path)
        elif "Normal" in parent or "label_A" in path:
            buckets["Normal"].append(path)
        else:
            # Match by prefix/name
            matched = False
            for c in classes:
                if c.lower() in path.lower():
                    buckets[c].append(path)
                    matched = True
                    break
            if not matched:
                buckets["Normal"].append(path)

    # Shuffle within each bucket
    for k in buckets:
        rng.shuffle(buckets[k])

    fold_buckets: List[List[str]] = [[] for _ in range(num_folds)]
    for cat, items in buckets.items():
        for i, item in enumerate(items):
            fold_buckets[i % num_folds].append(item)

    splits: List[Tuple[List[str], List[str]]] = []
    for f in range(num_folds):
        val_files = sorted(fold_buckets[f])
        train_files = []
        for other_f in range(num_folds):
            if other_f != f:
                train_files.extend(fold_buckets[other_f])
        splits.append((sorted(train_files), val_files))

    return splits

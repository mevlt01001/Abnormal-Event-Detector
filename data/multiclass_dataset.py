from __future__ import annotations

import os
import random
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataset import _load_features
from utils.annotation_parser import (
    XD_CLASSES,
    get_multi_hot_label,
    is_normal_video,
    strip_video_ext,
)


class MultiClassFeatureDataset(Dataset):
    """Multi-Class Feature Dataset for XD-Violence Video Anomaly Detection.

    Supports both normal videos (target is all zeros) and anomalous videos
    (target is multi-hot encoded across the classes, e.g. B1, B2, B4, B5, B6, G).

    Can be initialized with:
    - A direct list of file paths (`files`)
    - Separate `normal_dir` and `anomal_dir`
    - A single `features_dir` containing 'normal' and 'anomal' subdirectories
    """

    def __init__(
        self,
        files: Optional[List[str]] = None,
        normal_dir: Optional[str] = None,
        anomal_dir: Optional[str] = None,
        features_dir: Optional[str] = None,
        class_list: Optional[List[str]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.class_list = class_list if class_list is not None else XD_CLASSES
        self.transform = transform
        self.file_paths: List[str] = []

        if files is not None:
            self.file_paths = sorted(files)
        elif features_dir is not None:
            if not os.path.isdir(features_dir):
                raise NotADirectoryError(f"Features directory does not exist: {features_dir}")
            norm_sub = os.path.join(features_dir, "normal")
            anom_sub = os.path.join(features_dir, "anomal")
            collected = []
            if os.path.isdir(norm_sub):
                collected.extend([os.path.join(norm_sub, f) for f in os.listdir(norm_sub) if f.endswith(".pt")])
            if os.path.isdir(anom_sub):
                collected.extend([os.path.join(anom_sub, f) for f in os.listdir(anom_sub) if f.endswith(".pt")])
            if not collected:
                # Flat directory fallback
                collected = [os.path.join(features_dir, f) for f in os.listdir(features_dir) if f.endswith(".pt")]
            self.file_paths = sorted(collected)
        elif normal_dir is not None and anomal_dir is not None:
            if not os.path.isdir(normal_dir):
                raise NotADirectoryError(f"Normal directory does not exist: {normal_dir}")
            if not os.path.isdir(anomal_dir):
                raise NotADirectoryError(f"Anomalous directory does not exist: {anomal_dir}")
            norm_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith(".pt")]
            anom_files = [os.path.join(anomal_dir, f) for f in os.listdir(anomal_dir) if f.endswith(".pt")]
            self.file_paths = sorted(norm_files + anom_files)
        else:
            raise ValueError("Either 'files', 'features_dir', or both 'normal_dir' and 'anomal_dir' must be provided.")

        if not self.file_paths:
            raise ValueError("No feature files (.pt) found.")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        file_path = self.file_paths[idx]
        features = _load_features(file_path)
        if self.transform is not None:
            features = self.transform(features)

        basename = os.path.basename(file_path)
        video_name = strip_video_ext(basename)
        target = get_multi_hot_label(video_name, valid_classes=self.class_list)

        return features, target, video_name


def create_multiclass_kfold_splits(
    file_list: List[str],
    num_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[List[str], List[str]]]:
    """Generates stratified K-Fold splits (train_files, val_files) preserving normal vs anomalous ratio.

    Args:
        file_list: All feature .pt file paths.
        num_folds: Number of cross-validation folds.
        seed: Random seed for reproducibility.

    Returns:
        List of (train_files, val_files) tuples for each fold.
    """
    normal_files = [f for f in file_list if is_normal_video(f)]
    anomal_files = [f for f in file_list if not is_normal_video(f)]

    rng = random.Random(seed)
    rng.shuffle(normal_files)
    rng.shuffle(anomal_files)

    def split_into_k_chunks(lst: List[str], k: int) -> List[List[str]]:
        chunks: List[List[str]] = [[] for _ in range(k)]
        for i, item in enumerate(lst):
            chunks[i % k].append(item)
        return chunks

    normal_chunks = split_into_k_chunks(normal_files, num_folds)
    anomal_chunks = split_into_k_chunks(anomal_files, num_folds)

    folds: List[Tuple[List[str], List[str]]] = []
    for fold_idx in range(num_folds):
        val_files = normal_chunks[fold_idx] + anomal_chunks[fold_idx]
        train_files: List[str] = []
        for j in range(num_folds):
            if j != fold_idx:
                train_files.extend(normal_chunks[j] + anomal_chunks[j])

        if not train_files:
            train_files = list(val_files)
        if not val_files:
            val_files = list(train_files)

        rng.shuffle(train_files)
        rng.shuffle(val_files)
        folds.append((train_files, val_files))

    return folds


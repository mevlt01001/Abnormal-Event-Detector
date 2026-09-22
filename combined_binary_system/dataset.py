"""Dataset Loader for Combined UCF-Crime & XD-Violence Binary Video Anomaly Detection.

Loads all extracted features (5,459 videos):
- Normal (2,847 videos): Label = 0.0 (UCF normal 1 & 2, XD-Violence normal)
- Anomaly (2,612 videos): Label = 1.0 across 19 anomaly categories (UCF + XD-Violence)
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Sequence, Tuple
import torch
from torch.utils.data import Dataset


def load_feature_tensor(file_path: str) -> torch.Tensor:
    """Loads a feature tensor from a .pt file safely.

    Handles both raw tensor dumps and dictionary formats (e.g. {'feats': tensor}).

    Args:
        file_path: Path to the .pt feature file.

    Returns:
        feats: Tensor of shape [32, feature_dim] with dtype float32.
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

    # Ensure shape is [32, D] float32 tensor
    return feats.float()


def scan_combined_features(
    features_root: str = "swin3d_t_extracted_features",
) -> List[Tuple[str, float, str]]:
    """Scans all UCF-Crime and XD-Violence feature directories and assigns binary labels.

    Args:
        features_root: Directory containing extracted feature folders.

    Returns:
        List of (file_path, label, category_name) tuples:
            label is 0.0 for Normal videos, 1.0 for Anomaly videos.
    """
    if not os.path.isdir(features_root):
        # Check potential fallback paths
        fallbacks = ["extracted_features", "data/features/swin3d_t", "data/unified_features"]
        found = None
        for fb in fallbacks:
            if os.path.isdir(fb):
                found = fb
                break
        if found is not None:
            features_root = found
        else:
            raise FileNotFoundError(f"Features root directory not found: {features_root}")

    samples: List[Tuple[str, float, str]] = []

    for entry in sorted(os.listdir(features_root)):
        sub_path = os.path.join(features_root, entry)
        if not os.path.isdir(sub_path) or entry.startswith("."):
            continue

        entry_lower = entry.lower()
        is_normal = "normal" in entry_lower
        label = 0.0 if is_normal else 1.0
        category = entry

        for fname in sorted(os.listdir(sub_path)):
            if fname.endswith(".pt") and not fname.startswith("."):
                full_path = os.path.join(sub_path, fname)
                samples.append((full_path, label, category))

    return samples


def create_stratified_train_val_split(
    samples: Sequence[Tuple[str, float, str]],
    val_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[List[Tuple[str, float, str]], List[Tuple[str, float, str]]]:
    """Splits samples into train and validation sets while strictly preserving category proportions.

    Args:
        samples: List of (file_path, label, category) tuples.
        val_ratio: Fraction of each category to allocate to validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_samples, val_samples).
    """
    rng = random.Random(seed)

    category_buckets: Dict[str, List[Tuple[str, float, str]]] = {}
    for sample in samples:
        cat = sample[2]
        if cat not in category_buckets:
            category_buckets[cat] = []
        category_buckets[cat].append(sample)

    train_samples: List[Tuple[str, float, str]] = []
    val_samples: List[Tuple[str, float, str]] = []

    for cat, cat_items in sorted(category_buckets.items()):
        shuffled = list(cat_items)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
        val_samples.extend(shuffled[:n_val])
        train_samples.extend(shuffled[n_val:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def create_stratified_kfold_splits(
    samples: Sequence[Tuple[str, float, str]],
    num_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[List[Tuple[str, float, str]], List[Tuple[str, float, str]]]]:
    """Creates balanced, category-stratified K-Fold splits for binary anomaly detection.

    Ensures each fold contains a proportional representation of Normal videos
    and all anomaly sub-categories.

    Args:
        samples: List of (file_path, label, category) tuples.
        num_folds: Number of folds (default: 5).
        seed: Random seed for reproducibility.

    Returns:
        List of (train_samples, val_samples) tuples for each fold.
    """
    rng = random.Random(seed)
    category_buckets: Dict[str, List[Tuple[str, float, str]]] = {}
    for sample in samples:
        cat = sample[2]
        if cat not in category_buckets:
            category_buckets[cat] = []
        category_buckets[cat].append(sample)

    fold_buckets: List[List[Tuple[str, float, str]]] = [[] for _ in range(num_folds)]
    for cat, cat_items in sorted(category_buckets.items()):
        shuffled = list(cat_items)
        rng.shuffle(shuffled)
        for i, item in enumerate(shuffled):
            fold_buckets[i % num_folds].append(item)

    folds: List[Tuple[List[Tuple[str, float, str]], List[Tuple[str, float, str]]]] = []
    for f_idx in range(num_folds):
        val_set = list(fold_buckets[f_idx])
        train_set: List[Tuple[str, float, str]] = []
        for j in range(num_folds):
            if j != f_idx:
                train_set.extend(fold_buckets[j])
        rng.shuffle(train_set)
        rng.shuffle(val_set)
        folds.append((train_set, val_set))

    return folds


class CombinedBinaryDataset(Dataset):
    """PyTorch Dataset for combined UCF-Crime and XD-Violence binary anomaly classification.

    Args:
        samples: List of (file_path, label, category) tuples.
    """

    def __init__(self, samples: Sequence[Tuple[str, float, str]]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Loads and returns a sample.

        Returns:
            features: Float tensor of shape [32, feature_dim].
            label: Binary float tensor of shape [1] in {0.0, 1.0}.
            video_name: Basename string of the video feature file.
        """
        fpath, lbl, _ = self.samples[idx]
        features = load_feature_tensor(fpath)  # [32, D]
        label = torch.tensor([lbl], dtype=torch.float32)  # [1]
        video_name = os.path.splitext(os.path.basename(fpath))[0]
        return features, label, video_name

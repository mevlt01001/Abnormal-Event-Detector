"""Isolated Dataset Loader for UCF-Crime Binary Video Anomaly Detection.

Loads exclusively UCF-Crime features (1,750 videos):
- Normal (800 videos): Label = 0.0
- Anomaly (950 videos across 13 crime categories): Label = 1.0
"""

from __future__ import annotations

import os
import random
from typing import List, Sequence, Tuple
import torch
from torch.utils.data import Dataset


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


def scan_ucf_crime_features(
    features_root: str = "extracted_features",
) -> List[Tuple[str, float, str]]:
    """Scans all UCF-Crime extracted feature folders and labels them binary.

    Args:
        features_root: Path to extracted_features directory.

    Returns:
        List of (file_path, label, category_name) tuples:
            label is 0.0 for Normal videos, 1.0 for Anomaly videos.
    """
    samples: List[Tuple[str, float, str]] = []

    if not os.path.isdir(features_root):
        raise FileNotFoundError(f"Features root directory not found: {features_root}")

    for entry in sorted(os.listdir(features_root)):
        if entry.startswith("ucf_") and os.path.isdir(os.path.join(features_root, entry)):
            sub_path = os.path.join(features_root, entry)
            is_normal = "normal" in entry.lower()
            label = 0.0 if is_normal else 1.0
            category = "Normal" if is_normal else entry.replace("ucf_", "").replace("_videos_root", "")

            for f in sorted(os.listdir(sub_path)):
                if f.endswith(".pt") and not f.startswith("."):
                    full_f = os.path.join(sub_path, f)
                    samples.append((full_f, label, category))

    return samples


class UCFBinaryDataset(Dataset):
    """PyTorch Dataset for UCF-Crime binary anomaly classification.

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
            features: Tensor of shape [32, feature_dim]
            label: Binary float tensor of shape [1] in {0.0, 1.0}
            video_name: Basename string of the video file
        """
        fpath, lbl, cat = self.samples[idx]
        # [32, feature_dim]
        features = load_feature_tensor(fpath)
        # [1]
        label = torch.tensor([lbl], dtype=torch.float32)
        video_name = os.path.splitext(os.path.basename(fpath))[0]
        return features, label, video_name


def create_stratified_train_val_split(
    samples: Sequence[Tuple[str, float, str]],
    val_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[List[Tuple[str, float, str]], List[Tuple[str, float, str]]]:
    """Splits UCF-Crime samples into balanced train and validation partitions.

    Preserves exact category and binary anomaly/normal ratios in both splits.

    Args:
        samples: List of (file_path, label, category) tuples.
        val_ratio: Fraction of samples reserved for validation (default: 0.20).
        seed: Random seed for reproducible stratification.

    Returns:
        (train_samples, val_samples)
    """
    rng = random.Random(seed)
    # Group samples by category
    buckets: dict[str, List[Tuple[str, float, str]]] = {}
    for item in samples:
        cat = item[2]
        buckets.setdefault(cat, []).append(item)

    train_samples = []
    val_samples = []

    for cat, items in sorted(buckets.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val_samples.extend(shuffled[:n_val])
        train_samples.extend(shuffled[n_val:])

    # Shuffle overall sets
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    return train_samples, val_samples

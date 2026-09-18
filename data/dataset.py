from __future__ import annotations

import os
import random
from typing import Callable, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset


def _load_features(file_path: str) -> torch.Tensor:
    """Loads feature tensor from a .pt file safely.

    Handles both raw tensor dumps and dictionary formats like {'feats': tensor}.
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
            first_tensor = next((v for v in data.values() if isinstance(v, torch.Tensor)), None)
            if first_tensor is not None:
                feats = first_tensor
            else:
                raise ValueError(f"No tensor found in dictionary loaded from {file_path}")
    elif isinstance(data, torch.Tensor):
        feats = data
    else:
        raise TypeError(f"Unexpected data type loaded from {file_path}: {type(data)}")

    return feats.float()


class PairwiseFeatureDataset(Dataset):
    """Pairwise Video Feature Dataset for Sultani MIL Anomaly Detection.

    Each item returns a pair: (anomalous_video_features, normal_video_features).
    The dataset length matches the number of anomalous video feature files.
    For each anomalous video, a normal video is randomly selected to form the pair,
    enabling diverse pairwise comparisons across epochs.
    """

    def __init__(
        self,
        anomal_dir: str,
        normal_dir: str,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        if not os.path.isdir(anomal_dir):
            raise NotADirectoryError(f"Anomalous directory does not exist: {anomal_dir}")
        if not os.path.isdir(normal_dir):
            raise NotADirectoryError(f"Normal directory does not exist: {normal_dir}")

        self.anomal_dir = anomal_dir
        self.normal_dir = normal_dir
        self.transform = transform

        self.anomal_files: List[str] = sorted([
            os.path.join(anomal_dir, f)
            for f in os.listdir(anomal_dir)
            if f.endswith(".pt")
        ])
        self.normal_files: List[str] = sorted([
            os.path.join(normal_dir, f)
            for f in os.listdir(normal_dir)
            if f.endswith(".pt")
        ])

        if not self.anomal_files:
            raise ValueError(f"No .pt files found in anomalous directory: {anomal_dir}")
        if not self.normal_files:
            raise ValueError(f"No .pt files found in normal directory: {normal_dir}")

    def __len__(self) -> int:
        return len(self.anomal_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        anomal_path = self.anomal_files[idx]
        normal_path = random.choice(self.normal_files)

        anomal_feats = _load_features(anomal_path)
        normal_feats = _load_features(normal_path)

        if self.transform is not None:
            anomal_feats = self.transform(anomal_feats)
            normal_feats = self.transform(normal_feats)

        return anomal_feats, normal_feats

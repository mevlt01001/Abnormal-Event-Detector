"""Unified and Modular Video Anomaly Datasets (UCF, XDV, and Combined 4-Wrapper).

Designed with strict DRY (Don't Repeat Yourself) principles:
- BaseVideoAnomalyDataset: Core abstraction handling tensor loading, dynamic mode switching
  (.set_mode("binary") vs .set_mode("multiclass")), labeling, and batch indexing.
- UCF_Dataset: Pure UCF-Crime split (14 classes: Normal + 13 anomaly).
- XDV_Dataset: Pure XD-Violence split (7 classes: Normal + 6 anomaly).
- XDV_UCF_Combined_Dataset: Cross-dataset fusion using the 4 Macro Wrapper classes (+ Normal).
- get_dataset: Factory function for one-line instantiation.
"""

from __future__ import annotations

import glob
import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch.utils.data import Dataset

from .taxonomy import (
    DEFAULT_MACRO_CLASSES,
    EXACT_MATCH_CLASSES,
    INTERSECTION_CLASSES,
    RAW_TO_WRAPPER_MAP,
    UCF_CLASSES,
    WRAPPER_CLASSES,
    XDV_CLASSES,
    load_dataset_taxonomy,
    map_to_wrapper,
)


def load_feature_tensor(file_path: str) -> torch.Tensor:
    """Safely loads a segment feature tensor from a .pt file into float32 [num_segments, feature_dim]."""
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

    return feats.float()


def _normalize_xdv_name(ann_key: str, pt_name_set: set) -> Optional[str]:
    """Maps an XDV annotation key to the corresponding .pt filename (without extension).

    Annotation keys:  'Bad.Boys.1995__#01-11-55_01-12-40_label_G-B2-B6'
    PT filenames:      'Shooting_Bad.Boys.1995___01-11-55_01-12-40_label_G-B2-B6'

    The PT file has '{ClassName}_{RestOfName}' where '___' replaces '__#'.
    """
    # Normalize annotation key: __# -> ___
    core = ann_key.replace("__#", "___")
    # Try direct match first (rare but possible for v= prefixed names)
    if core in pt_name_set:
        return core
    # Try matching with each class prefix
    for pt_name in pt_name_set:
        # Strip class prefix: e.g. 'Shooting_Bad.Boys...' -> 'Bad.Boys...'
        underscore_idx = pt_name.find("_")
        if underscore_idx > 0:
            suffix = pt_name[underscore_idx + 1:]
            if suffix == core:
                return pt_name
    return None


class BaseVideoAnomalyDataset(Dataset):
    """Core dataset base class implementing DRY logic for binary and multiclass modes."""

    def __init__(
        self,
        file_paths: Sequence[str],
        labels: Sequence[str],
        classes: Sequence[str],
        mode: str = "multiclass",
        fps: Optional[float] = None,
        clip_size: Optional[int] = None,
        stride: Optional[int] = None,
        overlap: Optional[float] = None,
        num_segments: Optional[int] = None,
        ground_truth_intervals: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    ) -> None:
        if len(file_paths) != len(labels):
            raise ValueError(f"file_paths length ({len(file_paths)}) != labels length ({len(labels)})")

        self.file_paths = list(file_paths)
        self.labels = list(labels)
        self._raw_classes = list(classes)
        self.classes = list(classes)
        self.class_to_idx = {c: idx for idx, c in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        # Spatio-temporal invariant properties (user configurable, default or manifest-driven)
        self.fps = float(fps) if fps is not None else 20.0
        self.clip_size = int(clip_size) if clip_size is not None else 16
        self.stride = int(stride) if stride is not None else 16
        self.overlap = float(overlap) if overlap is not None else 0.5
        self.num_segments = int(num_segments) if num_segments is not None else 32
        self.ground_truth_intervals = dict(ground_truth_intervals) if ground_truth_intervals is not None else {}

        self.mode = "multiclass"
        self.set_mode(mode)

    def set_mode(self, mode: str) -> BaseVideoAnomalyDataset:
        """Dynamically switches dataset mode between 'binary', 'multiclass', and 'exact_match'."""
        mode_clean = mode.lower().strip()
        if mode_clean in ("exact_match", "exact_matching", "exact", "intersection"):
            self.mode = "exact_match"
            self.classes = list(getattr(self, "_raw_classes", self.classes))
            self.class_to_idx = {c: idx for idx, c in enumerate(self.classes)}
            self.num_classes = len(self.classes)
        elif mode_clean == "multiclass":
            self.mode = "multiclass"
            self.classes = list(getattr(self, "_raw_classes", self.classes))
            self.class_to_idx = {c: idx for idx, c in enumerate(self.classes)}
            self.num_classes = len(self.classes)
        elif mode_clean == "binary":
            self.mode = "binary"
            self.classes = ["Anomaly"]
            self.class_to_idx = {"Anomaly": 0}
            self.num_classes = 1
        else:
            raise ValueError(f"Invalid mode '{mode}'. Choose 'binary', 'multiclass', or 'exact_match'.")
        return self

    def _get_target(self, label: str) -> torch.Tensor:
        """Generates target tensor based on current mode."""
        if self.mode == "binary":
            # 0.0 for Normal, 1.0 for Anomaly
            is_anomaly = 0.0 if label == "Normal" else 1.0
            return torch.tensor(is_anomaly, dtype=torch.float32)

        # Multiclass mode: one-hot tensor of shape [num_classes] (all zeros for Normal)
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        if label != "Normal" and label in self.class_to_idx:
            target[self.class_to_idx[label]] = 1.0
        return target

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Loads and returns (features, target, video_name)."""
        path = self.file_paths[idx]
        features = load_feature_tensor(path)
        target = self._get_target(self.labels[idx])
        video_name = os.path.splitext(os.path.basename(path))[0]
        return features, target, video_name

    @staticmethod
    def scan_directory(split_dir: str, folders: Sequence[str]) -> Tuple[List[str], List[str]]:
        """Utility method scanning class subfolders inside split_dir."""
        file_paths = []
        labels = []
        for folder_name in folders:
            folder_path = os.path.join(split_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            pts = sorted(glob.glob(os.path.join(folder_path, "*.pt")))
            for pt in pts:
                file_paths.append(pt)
                labels.append(folder_name)
        return file_paths, labels


DEFAULT_DATA_DIR = "data/s3d_features" if os.path.isdir("data/s3d_features") else ("data/mvit_v2_s_features" if os.path.isdir("data/mvit_v2_s_features") else "data/unified_features")


def _resolve_data_dir(data_dir: str) -> str:
    """Resolves data directory, falling back to s3d_features or mvit_v2_s_features if specified path does not exist."""
    if os.path.isdir(data_dir):
        return data_dir
    for candidate in ["data/s3d_features", "data/mvit_v2_s_features", "data/unified_features"]:
        if os.path.isdir(candidate):
            return candidate
    return data_dir


class UCF_Dataset(BaseVideoAnomalyDataset):
    """Dataset dedicated to UCF-Crime features."""

    def __init__(
        self,
        split: str = "train",
        data_dir: str = DEFAULT_DATA_DIR,
        mode: str = "multiclass",
        classes: Optional[Sequence[str]] = None,
        fps: Optional[float] = None,
        clip_size: Optional[int] = None,
        stride: Optional[int] = None,
        overlap: Optional[float] = None,
        num_segments: Optional[int] = None,
        annotation_file: Optional[str] = None,
    ) -> None:
        data_dir = _resolve_data_dir(data_dir)
        split_dir = os.path.join(data_dir, split, "ucf")
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"UCF split directory not found: {split_dir}")

        # Anomaly classes (excluding Normal)
        anomaly_classes = (
            list(classes) if classes is not None
            else [c for c in UCF_CLASSES if c != "Normal"]
        )
        file_paths, labels = self.scan_directory(split_dir, anomaly_classes + ["Normal"])

        # Spatio-temporal metadata resolution (user arguments override manifest)
        meta = load_dataset_taxonomy(data_dir)
        final_fps = float(fps) if fps is not None else float(meta.get("fps", 20.0))
        final_clip_size = int(clip_size) if clip_size is not None else 16
        final_stride = int(stride) if stride is not None else 16
        final_overlap = float(overlap) if overlap is not None else float(meta.get("overlap_ratio", 0.5))
        final_num_segments = int(num_segments) if num_segments is not None else int(meta.get("num_segments", 32))

        # Ground truth intervals for test split
        gt_intervals: Dict[str, List[Tuple[float, float]]] = {}
        if split == "test":
            ann_candidates = [
                annotation_file,
                os.path.join(data_dir, "test", "ucf-test-annotatisons.txt"),
                os.path.join(data_dir, "test", "ucf-test-annotations.txt"),
            ]
            valid_ann = next((f for f in ann_candidates if f and os.path.isfile(f)), None)
            if valid_ann:
                from utils.annotation_parser import parse_ucf_annotations
                raw_gt = parse_ucf_annotations(valid_ann)
                for v_name, intervals in raw_gt.items():
                    gt_intervals[v_name] = [
                        (round(s / final_fps, 2), round(e / final_fps, 2))
                        for s, e in intervals
                    ]

        super().__init__(
            file_paths=file_paths,
            labels=labels,
            classes=anomaly_classes,
            mode=mode,
            fps=final_fps,
            clip_size=final_clip_size,
            stride=final_stride,
            overlap=final_overlap,
            num_segments=final_num_segments,
            ground_truth_intervals=gt_intervals,
        )
        self.split = split
        self.dataset_name = "UCF-Crime"


class XDV_Dataset(BaseVideoAnomalyDataset):
    """Dataset dedicated to XD-Violence features."""

    def __init__(
        self,
        split: str = "train",
        data_dir: str = DEFAULT_DATA_DIR,
        mode: str = "multiclass",
        classes: Optional[Sequence[str]] = None,
        fps: Optional[float] = None,
        clip_size: Optional[int] = None,
        stride: Optional[int] = None,
        overlap: Optional[float] = None,
        num_segments: Optional[int] = None,
        annotation_file: Optional[str] = None,
    ) -> None:
        data_dir = _resolve_data_dir(data_dir)
        split_dir = os.path.join(data_dir, split, "xdv")
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"XDV split directory not found: {split_dir}")

        # Anomaly classes (excluding Normal)
        anomaly_classes = (
            list(classes) if classes is not None
            else [c for c in XDV_CLASSES if c != "Normal"]
        )
        file_paths, labels = self.scan_directory(split_dir, anomaly_classes + ["Normal"])

        # Spatio-temporal metadata resolution (user arguments override manifest)
        meta = load_dataset_taxonomy(data_dir)
        final_fps = float(fps) if fps is not None else float(meta.get("fps", 20.0))
        final_clip_size = int(clip_size) if clip_size is not None else 16
        final_stride = int(stride) if stride is not None else 16
        final_overlap = float(overlap) if overlap is not None else float(meta.get("overlap_ratio", 0.5))
        final_num_segments = int(num_segments) if num_segments is not None else int(meta.get("num_segments", 32))

        # Ground truth intervals for test split
        gt_intervals: Dict[str, List[Tuple[float, float]]] = {}
        if split == "test":
            ann_candidates = [
                annotation_file,
                os.path.join(data_dir, "test", "xdv-test-annotations.txt"),
            ]
            valid_ann = next((f for f in ann_candidates if f and os.path.isfile(f)), None)
            if valid_ann:
                from utils.annotation_parser import parse_annotations
                raw_gt = parse_annotations(valid_ann)
                # Build set of .pt basenames for name normalization
                pt_name_set = set(
                    os.path.splitext(os.path.basename(p))[0] for p in file_paths
                )
                for ann_key, intervals in raw_gt.items():
                    matched_name = _normalize_xdv_name(ann_key, pt_name_set)
                    if matched_name is None:
                        continue
                    gt_intervals[matched_name] = [
                        (round(s / final_fps, 2), round(e / final_fps, 2))
                        for s, e in intervals
                    ]

        super().__init__(
            file_paths=file_paths,
            labels=labels,
            classes=anomaly_classes,
            mode=mode,
            fps=final_fps,
            clip_size=final_clip_size,
            stride=final_stride,
            overlap=final_overlap,
            num_segments=final_num_segments,
            ground_truth_intervals=gt_intervals,
        )
        self.split = split
        self.dataset_name = "XD-Violence"


class XDV_UCF_Combined_Dataset(BaseVideoAnomalyDataset):
    """Cross-dataset fusion combining UCF-Crime and XD-Violence.

    Supports 3 operational modes:
    1. 'multiclass' (default): Maps raw labels to the 4 Macro Wrapper classes (Violence_Affray,
       Accidents_Disasters, Gun_Violence, Property_Crimes). All dataset samples are included.
    2. 'exact_match' (or 'exact_matching'): Exact semantic intersection between UCF-Crime and XD-Violence.
       Only contains the 5 shared anomaly classes (Abuse, Explosion, Fighting, RoadAccidents, Shooting)
       plus Normal. Samples outside this intersection (e.g. Riot, Burglary, Robbery, etc.) are filtered out.
    3. 'binary': 0.0 for Normal, 1.0 for any Anomaly across all dataset samples.
    """

    def __init__(
        self,
        split: str = "train",
        data_dir: str = DEFAULT_DATA_DIR,
        mode: str = "multiclass",
        fps: Optional[float] = None,
        clip_size: Optional[int] = None,
        stride: Optional[int] = None,
        overlap: Optional[float] = None,
        num_segments: Optional[int] = None,
        annotation_file_ucf: Optional[str] = None,
        annotation_file_xdv: Optional[str] = None,
    ) -> None:
        data_dir = _resolve_data_dir(data_dir)
        # Re-use sub-datasets for clean file, label, and ground truth discovery
        self.ucf_ds = UCF_Dataset(
            split=split,
            data_dir=data_dir,
            mode="multiclass",
            fps=fps,
            clip_size=clip_size,
            stride=stride,
            overlap=overlap,
            num_segments=num_segments,
            annotation_file=annotation_file_ucf,
        )
        self.xdv_ds = XDV_Dataset(
            split=split,
            data_dir=data_dir,
            mode="multiclass",
            fps=fps,
            clip_size=clip_size,
            stride=stride,
            overlap=overlap,
            num_segments=num_segments,
            annotation_file=annotation_file_xdv,
        )

        # 1. Full dataset representation (used for 'multiclass' and 'binary')
        self._all_file_paths = self.ucf_ds.file_paths + self.xdv_ds.file_paths
        self._all_raw_labels = self.ucf_ds.labels + self.xdv_ds.labels
        self._all_wrapper_labels = (
            [map_to_wrapper(lbl, "ucf") for lbl in self.ucf_ds.labels]
            + [map_to_wrapper(lbl, "xdv") for lbl in self.xdv_ds.labels]
        )
        self._all_gt_intervals = {**self.ucf_ds.ground_truth_intervals, **self.xdv_ds.ground_truth_intervals}

        # 2. Exact Match / Intersection subset (only the 5 common anomaly classes + Normal)
        exact_indices = [
            i for i, lbl in enumerate(self._all_raw_labels)
            if lbl in EXACT_MATCH_CLASSES or lbl == "Normal"
        ]
        self._exact_file_paths = [self._all_file_paths[i] for i in exact_indices]
        self._exact_labels = [self._all_raw_labels[i] for i in exact_indices]
        exact_names = {os.path.splitext(os.path.basename(p))[0] for p in self._exact_file_paths}
        self._exact_gt_intervals = {
            k: v for k, v in self._all_gt_intervals.items() if k in exact_names
        }

        # Initial mode resolution
        mode_clean = mode.lower().strip()
        if mode_clean in ("exact_match", "exact_matching", "exact", "intersection"):
            init_classes = list(EXACT_MATCH_CLASSES)
            init_files = self._exact_file_paths
            init_labels = self._exact_labels
            init_gt = self._exact_gt_intervals
            canonical_mode = "exact_match"
        elif mode_clean == "binary":
            init_classes = ["Anomaly"]
            init_files = self._all_file_paths
            init_labels = self._all_raw_labels
            init_gt = self._all_gt_intervals
            canonical_mode = "binary"
        else:
            init_classes = list(WRAPPER_CLASSES)
            init_files = self._all_file_paths
            init_labels = self._all_wrapper_labels
            init_gt = self._all_gt_intervals
            canonical_mode = "multiclass"

        super().__init__(
            file_paths=init_files,
            labels=init_labels,
            classes=init_classes,
            mode=canonical_mode,
            fps=self.ucf_ds.fps,
            clip_size=self.ucf_ds.clip_size,
            stride=self.ucf_ds.stride,
            overlap=self.ucf_ds.overlap,
            num_segments=self.ucf_ds.num_segments,
            ground_truth_intervals=init_gt,
        )
        self.split = split
        self.dataset_name = "XDV_UCF_Combined"

    def _apply_mode(self, mode: str) -> None:
        """Applies mode-specific files, labels, and class taxonomies."""
        mode_clean = mode.lower().strip()
        if mode_clean in ("exact_match", "exact_matching", "exact", "intersection"):
            self.mode = "exact_match"
            self.classes = list(EXACT_MATCH_CLASSES)
            self.file_paths = list(self._exact_file_paths)
            self.labels = list(self._exact_labels)
            self.ground_truth_intervals = dict(self._exact_gt_intervals)
        elif mode_clean == "multiclass":
            self.mode = "multiclass"
            self.classes = list(WRAPPER_CLASSES)
            self.file_paths = list(self._all_file_paths)
            self.labels = list(self._all_wrapper_labels)
            self.ground_truth_intervals = dict(self._all_gt_intervals)
        elif mode_clean == "binary":
            self.mode = "binary"
            self.classes = ["Anomaly"]
            self.file_paths = list(self._all_file_paths)
            self.labels = list(self._all_raw_labels)
            self.ground_truth_intervals = dict(self._all_gt_intervals)
        else:
            raise ValueError(
                f"Invalid mode '{mode}' for XDV_UCF_Combined_Dataset. "
                f"Choose 'multiclass' (4 wrapper classes), 'exact_match' (5 intersection classes), or 'binary'."
            )
        self.class_to_idx = {c: idx for idx, c in enumerate(self.classes)}
        self.num_classes = len(self.classes)

    def set_mode(self, mode: str) -> XDV_UCF_Combined_Dataset:
        """Updates mode on both self and underlying child datasets."""
        self._apply_mode(mode)
        child_mode = "binary" if self.mode == "binary" else "multiclass"
        self.ucf_ds.set_mode(child_mode)
        self.xdv_ds.set_mode(child_mode)
        return self


# Backwards compatibility alias
ClassAwareVideoDataset = XDV_UCF_Combined_Dataset


def get_dataset(
    name: str = "combined",
    split: str = "train",
    mode: str = "multiclass",
    data_dir: str = DEFAULT_DATA_DIR,
    **kwargs: Any,
) -> BaseVideoAnomalyDataset:
    """Factory helper to obtain any dataset with a single function call."""
    data_dir = _resolve_data_dir(data_dir)
    name_clean = name.lower().strip()
    if name_clean in ("ucf", "ucf_crime", "ucf-crime"):
        return UCF_Dataset(split=split, data_dir=data_dir, mode=mode, **kwargs)
    elif name_clean in ("xdv", "xd_violence", "xd-violence"):
        return XDV_Dataset(split=split, data_dir=data_dir, mode=mode, **kwargs)
    elif name_clean in ("combined", "xdv_ucf", "ucf_xdv", "all"):
        return XDV_UCF_Combined_Dataset(split=split, data_dir=data_dir, mode=mode, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name '{name}'. Choose 'ucf', 'xdv', or 'combined'.")


def create_stratified_kfold_splits(
    file_paths: Sequence[str],
    num_folds: int = 5,
    seed: int = 42,
    class_list: Optional[Sequence[str]] = None,
) -> List[Tuple[List[str], List[str]]]:
    """Creates balanced, stratified K-Fold cross-validation splits."""
    classes = list(class_list) if class_list is not None else list(WRAPPER_CLASSES)
    rng = random.Random(seed)

    buckets: Dict[str, List[str]] = {c: [] for c in classes}
    buckets["Normal"] = []

    for path in file_paths:
        parent = os.path.basename(os.path.dirname(path))
        if parent in buckets:
            buckets[parent].append(path)
        elif "Normal" in parent or "label_A" in path:
            buckets["Normal"].append(path)
        else:
            matched = False
            for c in classes:
                if c.lower() in path.lower():
                    buckets[c].append(path)
                    matched = True
                    break
            if not matched:
                buckets["Normal"].append(path)

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

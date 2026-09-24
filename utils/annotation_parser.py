from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

XD_CLASSES: List[str] = ["B1", "B2", "B4", "B5", "B6", "G"]

CLASS_NAMES: Dict[str, str] = {
    "B1": "Fighting",
    "B2": "Shooting",
    "B4": "Riot",
    "B5": "Abuse",
    "B6": "Car Accident",
    "G": "Explosion",
}

from data.taxonomy import (
    DEFAULT_MACRO_CLASSES as MACRO_CLASSES,
    SOURCE_TO_MACRO_MAP as UNIFIED_TAXONOMY_MAP,
)
UNIFIED_CLASSES = MACRO_CLASSES



def strip_video_ext(filename: str) -> str:
    """Strips common video and feature file extensions while preserving inner dots in filenames."""
    for ext in [".mp4", ".avi", ".mkv", ".mov", ".pt", ".npy"]:
        if filename.endswith(ext):
            return filename[:-len(ext)]
    return filename


def is_normal_video(filename_or_path: str) -> bool:
    """Determines whether a video is normal based on naming conventions or directory structure."""
    basename = os.path.basename(filename_or_path)
    if "_label_A" in basename:
        return True
    parent = os.path.basename(os.path.dirname(filename_or_path)).lower()
    if parent == "normal":
        return True
    clean = strip_video_ext(basename).lower()
    return clean == "normal" or clean.startswith("normal_") or clean.startswith("normal-")


def get_video_classes(
    filename_or_path: str,
    valid_classes: Optional[List[str]] = None,
) -> List[str]:
    """Extracts violence class codes (e.g. ['B2', 'G']) from XD-Violence video filename.

    For normal videos ('_label_A'), returns an empty list.
    For anomalous videos, parses codes after 'label_'.
    """
    if is_normal_video(filename_or_path):
        return []

    if valid_classes is None:
        valid_classes = XD_CLASSES

    basename = os.path.basename(filename_or_path)
    match = re.search(r"label_([A-Za-z0-9\-]+)", basename)
    if not match:
        return []

    label_str = match.group(1)
    parts = label_str.split("-")
    found = [p for p in parts if p in valid_classes]
    # Return unique sorted classes
    return sorted(list(set(found)))


def get_multi_hot_label(
    filename_or_path: str,
    valid_classes: Optional[List[str]] = None,
) -> torch.Tensor:
    """Generates a multi-hot binary float tensor of shape (len(valid_classes),).

    Normal videos yield all zeros.
    Anomalous videos yield 1.0 for each detected class.
    """
    if valid_classes is None:
        valid_classes = XD_CLASSES

    classes = get_video_classes(filename_or_path, valid_classes)
    label = torch.zeros(len(valid_classes), dtype=torch.float32)
    for c in classes:
        if c in valid_classes:
            idx = valid_classes.index(c)
            label[idx] = 1.0
    return label


def parse_annotations(ann_file_path: str) -> Dict[str, List[Tuple[int, int]]]:
    """Parses XD-Violence annotations file.

    Format: <video_name> <start1> <end1> [<start2> <end2> ...]
    Returns mapping from cleaned video name (no extension) to list of (start_frame, end_frame).
    """
    if not os.path.isfile(ann_file_path):
        raise FileNotFoundError(f"Annotation file not found: {ann_file_path}")

    annotations: Dict[str, List[Tuple[int, int]]] = {}
    with open(ann_file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            raw_name = parts[0]
            clean_name = strip_video_ext(raw_name)
            frame_nums = parts[1:]
            if len(frame_nums) % 2 != 0:
                raise ValueError(
                    f"Invalid annotation format for {raw_name}: odd number of frame numbers."
                )

            intervals: List[Tuple[int, int]] = []
            for i in range(0, len(frame_nums), 2):
                start = int(frame_nums[i])
                end = int(frame_nums[i + 1])
                if start > end:
                    start, end = end, start
                intervals.append((start, end))

            annotations[clean_name] = intervals

    return annotations


def parse_ucf_annotations(ann_file_path: str) -> Dict[str, List[Tuple[int, int]]]:
    """Parses UCF-Crime test annotations file.

    Format: <video_name> <class> <start1> <end1> <start2> <end2>
    Returns mapping from cleaned video name (no extension) to list of (start_frame, end_frame).
    """
    if not os.path.isfile(ann_file_path):
        raise FileNotFoundError(f"UCF annotation file not found: {ann_file_path}")

    annotations: Dict[str, List[Tuple[int, int]]] = {}
    with open(ann_file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            raw_name = parts[0]
            clean_name = strip_video_ext(raw_name)
            frame_nums = parts[2:]
            intervals: List[Tuple[int, int]] = []
            for i in range(0, len(frame_nums), 2):
                if i + 1 >= len(frame_nums):
                    break
                s = int(frame_nums[i])
                e = int(frame_nums[i + 1])
                if s >= 0 and e >= 0:
                    if s > e:
                        s, e = e, s
                    intervals.append((s, e))

            annotations[clean_name] = intervals

    return annotations



def generate_frame_gt(
    intervals: List[Tuple[int, int]],
    total_frames: int,
) -> np.ndarray:
    """Generates frame-level binary ground truth array of shape (total_frames,).

    1 indicates anomalous frame, 0 indicates normal frame.
    """
    gt = np.zeros(total_frames, dtype=np.uint8)
    for start, end in intervals:
        s = max(0, min(start, total_frames - 1))
        e = max(0, min(end, total_frames))
        if s < e:
            gt[s:e] = 1
    return gt


def generate_segment_gt(
    intervals: List[Tuple[int, int]],
    total_frames: int,
    num_segments: int = 32,
    overlap_threshold: float = 0.0,
) -> torch.Tensor:
    """Generates segment-level binary ground truth tensor of shape (num_segments,).

    If overlap fraction > overlap_threshold, segment is considered anomalous (1.0).
    """
    if total_frames <= 0:
        return torch.zeros(num_segments, dtype=torch.float32)

    frame_gt = generate_frame_gt(intervals, total_frames)
    segment_gt = torch.zeros(num_segments, dtype=torch.float32)

    segment_edges = np.linspace(0, total_frames, num_segments + 1, dtype=int)
    for i in range(num_segments):
        start_idx = segment_edges[i]
        end_idx = segment_edges[i + 1]
        if end_idx <= start_idx:
            continue
        seg_frames = frame_gt[start_idx:end_idx]
        anomaly_ratio = float(np.mean(seg_frames))
        if anomaly_ratio > overlap_threshold:
            segment_gt[i] = 1.0

    return segment_gt


def expand_segment_scores_to_frames(
    segment_scores: Union[torch.Tensor, np.ndarray],
    total_frames: int,
) -> np.ndarray:
    """Expands (num_segments,) scores into a frame-level array of shape (total_frames,)."""
    if isinstance(segment_scores, torch.Tensor):
        scores = segment_scores.detach().cpu().numpy().flatten()
    else:
        scores = np.asarray(segment_scores).flatten()

    num_segments = len(scores)
    frame_scores = np.zeros(total_frames, dtype=np.float32)
    segment_edges = np.linspace(0, total_frames, num_segments + 1, dtype=int)

    for i in range(num_segments):
        start_idx = segment_edges[i]
        end_idx = segment_edges[i + 1]
        if end_idx > start_idx:
            frame_scores[start_idx:end_idx] = scores[i]

    return frame_scores

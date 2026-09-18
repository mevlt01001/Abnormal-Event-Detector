from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import _load_features
from data.multiclass_dataset import MultiClassFeatureDataset
from model.model import FC_head
from utils.annotation_parser import (
    CLASS_NAMES,
    XD_CLASSES,
    expand_segment_scores_to_frames,
    generate_frame_gt,
    get_multi_hot_label,
    is_normal_video,
    parse_annotations,
    strip_video_ext,
)


def compute_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes Area Under ROC Curve (AUC-ROC) using pure NumPy.

    Args:
        y_true: 1D binary ground truth array (0 or 1).
        y_score: 1D predicted confidence scores.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return 0.5
    desc_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_indices]
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    if tps[-1] == 0 or fps[-1] == 0:
        return 0.5
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(tpr, fpr))
    return float(np.trapz(tpr, fpr))


def compute_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes Average Precision (AP) using pure NumPy."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    if np.sum(y_true) == 0:
        return 0.0
    desc_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_indices]
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    recalls = tps / tps[-1]
    precisions = tps / (tps + fps)
    recalls = np.concatenate([[0.0], recalls])
    precisions = np.concatenate([[1.0], precisions])
    return float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]))


def compute_temporal_iou(int1: Tuple[int, int], int2: Tuple[int, int]) -> float:
    """Computes 1D Temporal Intersection over Union (tIoU) between two frame intervals."""
    s1, e1 = int1
    s2, e2 = int2
    inter_start = max(s1, s2)
    inter_end = min(e1, e2)
    inter = max(0, inter_end - inter_start)
    union = max(e1, e2) - min(s1, s2)
    return float(inter / union) if union > 0 else 0.0


def extract_intervals_from_scores(
    scores: np.ndarray,
    threshold: float = 0.5,
    min_length: int = 1,
) -> List[Tuple[int, int]]:
    """Groups consecutive above-threshold elements into [start, end) intervals."""
    binary = (scores >= threshold).astype(int)
    intervals: List[Tuple[int, int]] = []
    in_interval = False
    start = 0

    for i, val in enumerate(binary):
        if val == 1 and not in_interval:
            in_interval = True
            start = i
        elif val == 0 and in_interval:
            in_interval = False
            if i - start >= min_length:
                intervals.append((start, i))

    if in_interval and len(binary) - start >= min_length:
        intervals.append((start, len(binary)))

    return intervals


def get_video_frame_counts(
    video_dir: Optional[str],
    cache_path: str = "data/video_frame_counts.json",
) -> Dict[str, int]:
    """Retrieves or caches total frame counts for all videos in video_dir."""
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except Exception:
            pass

    counts: Dict[str, int] = {}
    if video_dir and os.path.isdir(video_dir):
        files = [f for f in os.listdir(video_dir) if f.endswith((".mp4", ".avi", ".mkv"))]
        for f in files:
            vpath = os.path.join(video_dir, f)
            cap = cv2.VideoCapture(vpath)
            c = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            clean = strip_video_ext(f)
            counts[clean] = max(1, c)

        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(counts, f, indent=2)

    return counts


def evaluate_binary_model(
    model: nn.Module,
    features_dir: str,
    ann_file: str,
    video_dir: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    iou_thresholds: Tuple[float, ...] = (0.3, 0.5, 0.7),
) -> Dict[str, Any]:
    """Evaluates binary MIL model on test set.

    Computes:
    - Frame-level AUC-ROC
    - Frame-level Average Precision (AP)
    - Video-level Accuracy, Precision, Recall, F1
    - Segment-level Temporal IoU (tIoU) & segment mAP at multiple thresholds
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model.to(device)
    model.eval()

    annotations = parse_annotations(ann_file) if os.path.isfile(ann_file) else {}
    frame_counts = get_video_frame_counts(video_dir)

    # Collect all feature files
    dataset = MultiClassFeatureDataset(features_dir=features_dir)
    file_paths = dataset.file_paths

    all_frame_gt: List[np.ndarray] = []
    all_frame_scores: List[np.ndarray] = []

    video_gt: List[int] = []
    video_preds: List[float] = []

    iou_hits: Dict[float, int] = {thresh: 0 for thresh in iou_thresholds}
    total_gt_intervals = 0

    with torch.no_grad():
        for fp in file_paths:
            basename = os.path.basename(fp)
            clean_name = strip_video_ext(basename)
            is_normal = is_normal_video(clean_name)

            feats = _load_features(fp).to(device)  # (32, Dim)
            if feats.ndim == 2:
                feats = feats.unsqueeze(0)  # (1, 32, Dim)

            scores = model(feats)  # (1, 32, 1) or (1, 32)
            if scores.ndim == 3:
                scores = scores.squeeze(-1)
            scores = scores.squeeze(0).cpu().numpy()  # (32,)

            # Total frames for this video
            total_frames = frame_counts.get(clean_name, len(scores) * 16)

            # Ground truth frames
            intervals = annotations.get(clean_name, [])
            frame_gt = generate_frame_gt(intervals, total_frames)
            frame_scores = expand_segment_scores_to_frames(scores, total_frames)

            all_frame_gt.append(frame_gt)
            all_frame_scores.append(frame_scores)

            # Video level
            video_gt.append(0 if is_normal else 1)
            video_preds.append(float(np.max(scores)))

            # Temporal IoU evaluation for anomalous videos
            if not is_normal and intervals:
                total_gt_intervals += len(intervals)
                pred_intervals = extract_intervals_from_scores(frame_scores, threshold=0.5)

                for gt_int in intervals:
                    best_iou = 0.0
                    for pred_int in pred_intervals:
                        best_iou = max(best_iou, compute_temporal_iou(gt_int, pred_int))
                    for thresh in iou_thresholds:
                        if best_iou >= thresh:
                            iou_hits[thresh] += 1

    # Concatenate all frame predictions
    concat_gt = np.concatenate(all_frame_gt) if all_frame_gt else np.array([])
    concat_scores = np.concatenate(all_frame_scores) if all_frame_scores else np.array([])

    frame_auc = compute_auc_roc(concat_gt, concat_scores) if len(concat_gt) > 0 else 0.5
    frame_ap = compute_average_precision(concat_gt, concat_scores) if len(concat_gt) > 0 else 0.0

    # Video-level metrics
    v_gt = np.array(video_gt)
    v_preds = np.array(video_preds)
    v_auc = compute_auc_roc(v_gt, v_preds) if len(v_gt) > 0 else 0.5
    v_ap = compute_average_precision(v_gt, v_preds) if len(v_gt) > 0 else 0.0

    binary_preds = (v_preds >= 0.5).astype(int)
    tp = int(np.sum((binary_preds == 1) & (v_gt == 1)))
    fp = int(np.sum((binary_preds == 1) & (v_gt == 0)))
    tn = int(np.sum((binary_preds == 0) & (v_gt == 0)))
    fn = int(np.sum((binary_preds == 0) & (v_gt == 1)))

    acc = (tp + tn) / max(1, len(v_gt))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-7, prec + rec)

    # tIoU recall rates
    tiou_recalls = {
        f"tIoU_{thresh:.1f}": (iou_hits[thresh] / max(1, total_gt_intervals))
        for thresh in iou_thresholds
    }

    return {
        "frame_auc": float(frame_auc),
        "frame_ap": float(frame_ap),
        "video_auc": float(v_auc),
        "video_ap": float(v_ap),
        "video_accuracy": float(acc),
        "video_precision": float(prec),
        "video_recall": float(rec),
        "video_f1": float(f1),
        "tiou_recalls": tiou_recalls,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def evaluate_multiclass_model(
    model: nn.Module,
    features_dir: str,
    class_list: Optional[List[str]] = None,
    device: Optional[Union[str, torch.device]] = None,
    k_top: int = 1,
) -> Dict[str, Any]:
    """Evaluates multi-class MIL model on test set.

    Computes:
    - Per-class Average Precision (AP)
    - Mean Average Precision (mAP)
    - Multi-label exact match accuracy and subset accuracy
    - Per-class precision, recall, F1
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if class_list is None:
        class_list = XD_CLASSES

    model.to(device)
    model.eval()

    dataset = MultiClassFeatureDataset(features_dir=features_dir, class_list=class_list)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for feats, targets, _ in loader:
            feats = feats.to(device)
            # Model output: (B, 32, C) logits
            scores = model(feats)
            # Top-K pooling
            topk_logits = torch.topk(scores, k=min(k_top, scores.shape[1]), dim=1).values.mean(dim=1)
            probs = torch.sigmoid(topk_logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.cpu().numpy())

    if not all_preds:
        return {"mAP": 0.0, "per_class_ap": {}}

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)

    per_class_ap: Dict[str, float] = {}
    per_class_metrics: Dict[str, Dict[str, float]] = {}

    bin_pred = (y_pred >= 0.5).astype(int)

    for idx, c in enumerate(class_list):
        ap = compute_average_precision(y_true[:, idx], y_pred[:, idx])
        per_class_ap[c] = float(ap)

        tp = int(np.sum((bin_pred[:, idx] == 1) & (y_true[:, idx] == 1)))
        fp = int(np.sum((bin_pred[:, idx] == 1) & (y_true[:, idx] == 0)))
        fn = int(np.sum((bin_pred[:, idx] == 0) & (y_true[:, idx] == 1)))

        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f = 2 * p * r / max(1e-7, p + r)

        per_class_metrics[c] = {
            "name": CLASS_NAMES.get(c, c),
            "ap": float(ap),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(np.sum(y_true[:, idx])),
        }

    m_ap = float(np.mean(list(per_class_ap.values())))

    # Exact match ratio (all labels match exactly)
    exact_matches = np.all(bin_pred == y_true, axis=1)
    exact_match_ratio = float(np.mean(exact_matches))

    return {
        "mAP": m_ap,
        "per_class_ap": per_class_ap,
        "per_class_details": per_class_metrics,
        "exact_match_ratio": exact_match_ratio,
    }


def evaluate_model_pipeline(
    binary_model: nn.Module,
    multiclass_model: nn.Module,
    features_dir: str,
    ann_file: str,
    video_dir: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    class_list: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Runs complete evaluation suite combining binary and multi-class metrics."""
    binary_results = evaluate_binary_model(
        model=binary_model,
        features_dir=features_dir,
        ann_file=ann_file,
        video_dir=video_dir,
        device=device,
    )
    multiclass_results = evaluate_multiclass_model(
        model=multiclass_model,
        features_dir=features_dir,
        class_list=class_list,
        device=device,
    )
    return {
        "binary": binary_results,
        "multiclass": multiclass_results,
    }

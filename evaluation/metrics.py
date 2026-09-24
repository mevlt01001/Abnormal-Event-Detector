"""Evaluation metrics for video anomaly detection powered by scikit-learn.

Provides standard academic benchmarks (ROC-AUC, PR-AUC, mAP, Top-k accuracy)
and realistic production-simulated event detection metrics with threshold sweep analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


def compute_roc_auc(y_true: Sequence[int | float] | np.ndarray, y_score: Sequence[float] | np.ndarray) -> float:
    """Computes Receiver Operating Characteristic Area Under Curve (ROC-AUC) using scikit-learn.

    Args:
        y_true: Ground truth binary indicators (0 or 1).
        y_score: Predicted continuous anomaly scores in [0, 1].

    Returns:
        ROC-AUC scalar float in [0.0, 1.0].
    """
    y_true_arr = np.asarray(y_true, dtype=np.int32).ravel()
    y_score_arr = np.asarray(y_score, dtype=np.float32).ravel()

    # Need at least one positive and one negative sample
    if len(np.unique(y_true_arr)) < 2:
        return 0.5

    try:
        auc = roc_auc_score(y_true_arr, y_score_arr)
        return float(np.clip(auc, 0.0, 1.0))
    except Exception:
        return 0.5


def compute_pr_auc(y_true: Sequence[int | float] | np.ndarray, y_score: Sequence[float] | np.ndarray) -> float:
    """Computes Precision-Recall Area Under Curve (PR-AUC / Average Precision) using scikit-learn.

    Args:
        y_true: Ground truth binary indicators (0 or 1).
        y_score: Predicted continuous anomaly scores.

    Returns:
        Average Precision scalar float in [0.0, 1.0].
    """
    y_true_arr = np.asarray(y_true, dtype=np.int32).ravel()
    y_score_arr = np.asarray(y_score, dtype=np.float32).ravel()

    if int(np.sum(y_true_arr == 1)) == 0 or len(y_true_arr) == 0:
        return 0.0

    try:
        ap = average_precision_score(y_true_arr, y_score_arr)
        return float(np.clip(ap, 0.0, 1.0))
    except Exception:
        return 0.0


def compute_temporal_iou(
    interval_pred: Tuple[float, float],
    interval_gt: Tuple[float, float],
) -> float:
    """Computes 1D Temporal Intersection-over-Union (tIoU) between two time intervals."""
    s1, e1 = interval_pred
    s2, e2 = interval_gt

    intersection = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def compute_multiclass_map(
    y_true: Sequence[Sequence[float]] | np.ndarray,
    y_score: Sequence[Sequence[float]] | np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    iou_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Computes Mean Average Precision (mAP) across classes using scikit-learn."""
    y_true_arr = np.asarray(y_true, dtype=np.float32)
    y_score_arr = np.asarray(y_score, dtype=np.float32)

    if y_true_arr.ndim != 2 or y_score_arr.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got y_true: {y_true_arr.shape}, y_score: {y_score_arr.shape}")

    num_classes = min(y_true_arr.shape[1], y_score_arr.shape[1])
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    else:
        class_names = list(class_names)[:num_classes]

    per_class_ap = {}
    valid_aps = []

    for i, c_name in enumerate(class_names):
        c_true = y_true_arr[:, i]
        c_score = y_score_arr[:, i]
        if np.sum(c_true > 0) > 0:
            ap = compute_pr_auc(c_true, c_score)
            per_class_ap[c_name] = ap
            valid_aps.append(ap)
        else:
            per_class_ap[c_name] = 0.0

    mean_ap = float(np.mean(valid_aps)) if valid_aps else 0.0
    metric_label = f"mAP@IoU={iou_threshold:.2f}" if iou_threshold is not None else "mAP"

    return {
        "mAP": mean_ap,
        "iou_threshold": iou_threshold,
        "metric_name": metric_label,
        "per_class": per_class_ap,
    }


def compute_multiclass_auc(
    y_true: Sequence[Sequence[float]] | np.ndarray,
    y_score: Sequence[Sequence[float]] | np.ndarray,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Computes Macro ROC-AUC and per-class ROC-AUC using scikit-learn."""
    y_true_arr = np.asarray(y_true, dtype=np.float32)
    y_score_arr = np.asarray(y_score, dtype=np.float32)
    num_classes = min(y_true_arr.shape[1], y_score_arr.shape[1])

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    else:
        class_names = list(class_names)[:num_classes]

    per_class_auc = {}
    valid_aucs = []

    for i, c_name in enumerate(class_names):
        c_true = y_true_arr[:, i]
        c_score = y_score_arr[:, i]
        if np.sum(c_true > 0) > 0 and np.sum(c_true == 0) > 0:
            auc = compute_roc_auc(c_true, c_score)
            per_class_auc[c_name] = auc
            valid_aucs.append(auc)
        else:
            per_class_auc[c_name] = 0.5

    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.5
    return {
        "macro_auc": macro_auc,
        "per_class": per_class_auc,
    }


def compute_topk_accuracy(
    y_true: Sequence[int | Sequence[float]] | np.ndarray,
    y_score: Sequence[Sequence[float]] | np.ndarray,
    topk: Sequence[int] = (1, 3, 5),
    anomalous_only: bool = True,
) -> Dict[int, float]:
    """Computes Top-k classification accuracy for multi-class predictions."""
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score, dtype=np.float32)

    if y_score_arr.ndim != 2:
        raise ValueError(f"Expected 2D y_score array [N, C], got {y_score_arr.shape}")

    num_samples, num_classes = y_score_arr.shape
    if num_samples == 0:
        return {int(k): 0.0 for k in topk}

    if y_true_arr.ndim == 2:
        if anomalous_only:
            pos_mask = np.sum(y_true_arr > 0, axis=1) > 0
            if not np.any(pos_mask):
                return {int(k): 0.0 for k in topk}
            y_true_arr = y_true_arr[pos_mask]
            y_score_arr = y_score_arr[pos_mask]

        n_eval = len(y_true_arr)
        sorted_indices = np.argsort(y_score_arr, axis=1)

        results: Dict[int, float] = {}
        for k in topk:
            k_clamped = min(max(1, int(k)), num_classes)
            topk_idx = sorted_indices[:, -k_clamped:]
            hits = np.take_along_axis(y_true_arr, topk_idx, axis=1)
            correct = np.any(hits > 0, axis=1)
            results[int(k)] = float(np.mean(correct)) if n_eval > 0 else 0.0

        return results

    elif y_true_arr.ndim == 1:
        y_true_1d = y_true_arr.astype(np.int64)
        sorted_indices = np.argsort(y_score_arr, axis=1)
        n_eval = len(y_true_1d)

        results = {}
        for k in topk:
            k_clamped = min(max(1, int(k)), num_classes)
            topk_idx = sorted_indices[:, -k_clamped:]
            correct = np.any(topk_idx == y_true_1d[:, None], axis=1)
            results[int(k)] = float(np.mean(correct)) if n_eval > 0 else 0.0

        return results
    else:
        raise ValueError(f"y_true must be 1D or 2D array, got ndim={y_true_arr.ndim}")


def evaluate_event_detection(
    detected_segments: Sequence[Dict[str, Any]],
    gt_intervals: Sequence[Tuple[float, float]],
    iou_threshold: float = 0.30,
) -> Dict[str, Any]:
    """Matches detected segments against ground-truth intervals using 1D Temporal IoU.

    Returns:
        Dictionary with counts: tp, fp, fn.
    """
    matched_gt = set()
    tp = 0
    fp = 0

    for det in detected_segments:
        det_interval = (float(det["start_time"]), float(det["end_time"]))
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_int in enumerate(gt_intervals):
            iou = compute_temporal_iou(det_interval, gt_int)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_intervals) - len(matched_gt)
    return {"tp": tp, "fp": fp, "fn": fn}


def compute_threshold_sweep(
    video_scores_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
    gt_intervals_dict: Dict[str, List[Tuple[float, float]]],
    video_durations_dict: Dict[str, float],
    thresholds: Optional[Sequence[float]] = None,
    tolerance_sec: float = 3.0,
    padding_sec: float = 2.0,
    iou_threshold: float = 0.30,
    multi_iou_thresholds: Sequence[float] = (0.10, 0.20, 0.30, 0.50),
) -> Dict[str, Any]:
    """Performs full threshold sweep from 0.05 to 0.95 in 0.05 increments.

    Calculates real-world event-level TP, FP, FN, Precision, Recall, and F1 at every
    operating point to discover the Optimal Operating Threshold maximizing F1.

    Args:
        video_scores_dict: Mapping from video_name to 1D or 2D score array.
        gt_intervals_dict: Mapping from video_name to list of (start_sec, end_sec) ground truth intervals.
        video_durations_dict: Mapping from video_name to duration in seconds.
        thresholds: Sequence of thresholds to sweep (default: 0.05 to 0.95 with step 0.05).
        tolerance_sec: Gap tolerance in seconds.
        padding_sec: Security padding in seconds.
        iou_threshold: Minimum tIoU for True Positive matching (default: 0.30).
        multi_iou_thresholds: Sequence of tIoU thresholds for multi-level benchmark table.

    Returns:
        Dictionary containing the full sweep table, optimal threshold, best F1, multi-tIoU summary, and markdown report.
    """
    from model.analyzer import extract_anomaly_segments

    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]

    sweep_table: List[Dict[str, Any]] = []
    best_f1 = -1.0
    best_thresh = 0.35
    best_metrics: Dict[str, Any] = {}

    for tau in thresholds:
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for v_name, scores in video_scores_dict.items():
            # Only evaluate videos that have a ground truth entry (even if empty = normal)
            if v_name not in gt_intervals_dict:
                continue
            dur = video_durations_dict.get(v_name, 60.0)
            gt_ints = gt_intervals_dict.get(v_name, [])

            segments, _, _ = extract_anomaly_segments(
                scores=scores,
                video_seconds=dur,
                threshold=tau,
                tolerance_sec=tolerance_sec,
                padding_sec=padding_sec,
            )

            res = evaluate_event_detection(segments, gt_ints, iou_threshold=iou_threshold)
            total_tp += res["tp"]
            total_fp += res["fp"]
            total_fn += res["fn"]

        prec = float(total_tp / max(1, total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
        rec = float(total_tp / max(1, total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
        f1 = float(2 * (prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0

        status = ""
        if tau <= 0.10:
            status = "Yüksek Yanlış Alarm"
        elif tau >= 0.85:
            status = "Aşırı Kaçırma"
        elif 0.30 <= tau <= 0.45:
            status = "Dengeli Bölge"

        row = {
            "threshold": tau,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "status": status,
        }
        sweep_table.append(row)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = tau
            best_metrics = row

    # Tag optimal threshold in table
    for r in sweep_table:
        if r["threshold"] == best_thresh:
            r["status"] = "★ OPTIMAL (EN İYİ F1)"

    # Multi-tIoU Benchmark Summary at Optimal Operating Point
    multi_tiou_table: List[Dict[str, Any]] = []
    tiou_standards = {
        0.10: "Alarm Tetikleme (Point/Hit Level)",
        0.20: "Endüstriyel CCTV / S-THUMOS",
        0.30: "Operasyonel Referans Standart",
        0.40: "Orta-Yüksek Zamansal Uyum",
        0.50: "Katı TAL / Üst Sınır",
    }

    # Extract segments at best_thresh once
    best_segments_cache: List[Tuple[Sequence[Dict[str, Any]], Sequence[Tuple[float, float]]]] = []
    for v_name, scores in video_scores_dict.items():
        if v_name not in gt_intervals_dict:
            continue
        dur = video_durations_dict.get(v_name, 60.0)
        gt_ints = gt_intervals_dict.get(v_name, [])
        segs, _, _ = extract_anomaly_segments(
            scores=scores,
            video_seconds=dur,
            threshold=best_thresh,
            tolerance_sec=tolerance_sec,
            padding_sec=padding_sec,
        )
        best_segments_cache.append((segs, gt_ints))

    for m_iou in multi_iou_thresholds:
        m_tp, m_fp, m_fn = 0, 0, 0
        for segs, gt_ints in best_segments_cache:
            res = evaluate_event_detection(segs, gt_ints, iou_threshold=m_iou)
            m_tp += res["tp"]
            m_fp += res["fp"]
            m_fn += res["fn"]
        m_prec = float(m_tp / max(1, m_tp + m_fp)) if (m_tp + m_fp) > 0 else 0.0
        m_rec = float(m_tp / max(1, m_tp + m_fn)) if (m_tp + m_fn) > 0 else 0.0
        m_f1 = float(2 * (m_prec * m_rec) / (m_prec + m_rec)) if (m_prec + m_rec) > 0 else 0.0
        std_name = tiou_standards.get(round(float(m_iou), 2), "Özel Eşik")
        multi_tiou_table.append({
            "iou_threshold": float(m_iou),
            "standard": std_name,
            "tp": m_tp,
            "fp": m_fp,
            "fn": m_fn,
            "precision": m_prec,
            "recall": m_rec,
            "f1_score": m_f1,
        })

    # Generate Markdown Table Report
    md_lines = [
        f"# Operasyonel Eşik Tarama Analiz Raporu (tIoU >= {iou_threshold:.2f})",
        "",
        f"- **Değerlendirilen Eşikler**: {len(thresholds)} nokta ({thresholds[0]} -> {thresholds[-1]})",
        f"- **Zaman Parametreleri**: `tolerance_sec={tolerance_sec}s`, `padding_sec={padding_sec}s`",
        f"- **★ Önerilen Optimal Eşik**: **`tau = {best_thresh:.2f}`** (En Yüksek F1: **%{best_f1*100:.2f}**)",
        f"- **Optimal Doğruluk**: Alarm Güvenilirliği (Precision)=**%{best_metrics.get('precision', 0)*100:.2f}** | Olay Yakalama (Recall)=**%{best_metrics.get('recall', 0)*100:.2f}**",
        "",
        "| Eşik (tau) | TP (Doğru Alarm) | FP (Yanlış Alarm) | FN (Kaçırılan) | Precision | Recall | F1-Score | Durum |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for r in sweep_table:
        md_lines.append(
            f"| `{r['threshold']:.2f}` | {r['tp']} | {r['fp']} | {r['fn']} | %{r['precision']*100:.2f} | %{r['recall']*100:.2f} | %{r['f1_score']*100:.2f} | {r['status']} |"
        )

    if multi_tiou_table:
        md_lines.extend([
            "",
            f"### Kademeli Zamansal IoU (Multi-tIoU) Başarı Özeti (Optimal Eşik tau = {best_thresh:.2f})",
            "",
            "| tIoU Eşiği | Değerlendirme Standardı | TP | FP | FN | Precision | Recall | F1-Score |",
            "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for row in multi_tiou_table:
            md_lines.append(
                f"| `tIoU >= {row['iou_threshold']:.2f}` | {row['standard']} | {row['tp']} | {row['fp']} | {row['fn']} | %{row['precision']*100:.2f} | %{row['recall']*100:.2f} | %{row['f1_score']*100:.2f} |"
            )

    md_report = "\n".join(md_lines)

    return {
        "best_threshold": best_thresh,
        "best_f1": best_f1,
        "best_precision": best_metrics.get("precision", 0.0),
        "best_recall": best_metrics.get("recall", 0.0),
        "best_tp": best_metrics.get("tp", 0),
        "best_fp": best_metrics.get("fp", 0),
        "best_fn": best_metrics.get("fn", 0),
        "iou_threshold": iou_threshold,
        "tolerance_sec": tolerance_sec,
        "padding_sec": padding_sec,
        "table": sweep_table,
        "multi_tiou_summary": multi_tiou_table,
        "markdown_report": md_report,
    }

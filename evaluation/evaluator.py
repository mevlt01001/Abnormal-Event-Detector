"""Zero-boilerplate Evaluator with Dual-Engine Evaluation (Academic + Production Sweep)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .metrics import (
    compute_multiclass_auc,
    compute_multiclass_map,
    compute_pr_auc,
    compute_roc_auc,
    compute_threshold_sweep,
    compute_topk_accuracy,
)


def _get_dataset_attr(dataset_or_loader: Any, attr: str, default: Any = None) -> Any:
    """Safely extracts attribute even when wrapped in DataLoader or torch.utils.data.Subset."""
    obj = dataset_or_loader
    if isinstance(obj, DataLoader):
        obj = obj.dataset
    val = getattr(obj, attr, None)
    if val is not None:
        return val
    inner = getattr(obj, "dataset", None)
    if inner is not None:
        return getattr(inner, attr, default)
    return default


class Evaluator:
    """Zero-boilerplate Evaluator for Video Anomaly Detection models and checkpoints.

    Features Dual-Engine Evaluation:
    1. Academic Benchmark: ROC-AUC, PR-AUC, mAP@IoU, Top-1/3/5 classification accuracy via scikit-learn.
    2. Production Event Simulation (Threshold Sweep): Evaluates operating points from 0.05 to 0.95
       with tIoU >= 0.30 matching against ground-truth event intervals, discovering the Optimal F1 Threshold.

    Args:
        model: PyTorch nn.Module or string path to a checkpoint .pt file.
        dataset_or_loader: Dataset or DataLoader instance to evaluate on.
        device: 'cuda', 'cpu', or None (auto-detect).
        batch_size: Batch size if dataset was provided instead of a DataLoader (default: 32).
    """

    def __init__(
        self,
        model: Union[nn.Module, str],
        dataset_or_loader: Union[Dataset, DataLoader],
        device: Optional[Union[str, torch.device]] = None,
        batch_size: int = 32,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # 1. Model resolution
        self.checkpoint_metadata: Dict[str, Any] = {}
        if isinstance(model, str):
            if not os.path.isfile(model):
                raise FileNotFoundError(f"Checkpoint file not found: {model}")
            from model.head import AnomalyHead
            loaded_head, ckpt_dict = AnomalyHead.load_from_checkpoint(model, device=self.device)
            self.model = loaded_head
            self.checkpoint_metadata = ckpt_dict
        elif isinstance(model, nn.Module):
            head_module = getattr(model, "head", model)
            self.model = head_module.to(self.device).eval()
        else:
            raise TypeError(f"Expected nn.Module or checkpoint path str, got {type(model)}")

        # 2. DataLoader resolution
        if isinstance(dataset_or_loader, DataLoader):
            self.loader = dataset_or_loader
            self.dataset = dataset_or_loader.dataset
        elif isinstance(dataset_or_loader, Dataset):
            self.dataset = dataset_or_loader
            self.loader = DataLoader(dataset_or_loader, batch_size=batch_size, shuffle=False)
        else:
            raise TypeError(f"Expected Dataset or DataLoader, got {type(dataset_or_loader)}")

    @torch.no_grad()
    def evaluate(
        self,
        mode: str = "auto",
        iou_threshold: float = 0.30,
        k_top_segments: int = 1,
        topk_accuracies: Sequence[int] = (1, 3, 5),
        tolerance_sec: float = 3.0,
        padding_sec: float = 2.0,
        sweep_thresholds: bool = True,
        save_report_dir: Optional[str] = "results",
        print_summary: bool = True,
    ) -> Dict[str, Any]:
        """Runs full dual-engine evaluation and returns comprehensive metrics.

        Args:
            mode: 'auto', 'binary', or 'multiclass'.
            iou_threshold: Temporal IoU threshold for event matching (default: 0.30).
            k_top_segments: Number of top temporal segments to pool for video-level score
                (1 for max-pooling, >1 for top-k snippet average pooling).
            topk_accuracies: Sequence of k values for multi-class Top-k accuracy (default: (1, 3, 5)).
            tolerance_sec: Gap tolerance in seconds for segment merging (default: 3.0).
            padding_sec: Security padding in seconds prepended and appended to segments (default: 2.0).
            sweep_thresholds: If True, sweeps thresholds 0.05 to 0.95 to discover optimal F1 operating point.
            save_report_dir: Directory where markdown and JSON reports will be saved (default: 'results').
            print_summary: If True, prints formatted summary to console.

        Returns:
            Dictionary containing computed academic metrics, threshold sweep table, and optimal operating point.
        """
        self.model.eval()

        all_preds = []
        all_targets = []
        all_names = []
        video_scores_dict: Dict[str, np.ndarray] = {}
        video_durations_dict: Dict[str, float] = {}

        for batch in self.loader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                features = batch[0].to(self.device)
                targets = batch[1].cpu()
                names = batch[2] if len(batch) >= 3 else [f"sample_{i}" for i in range(len(features))]
            elif isinstance(batch, dict):
                features = batch["features"].to(self.device)
                targets = batch["target"].cpu()
                names = batch.get("video_name", [f"sample_{i}" for i in range(len(features))])
            else:
                raise ValueError("Batch must be (features, targets, names) or dict")

            logits = self.model(features)
            probs = torch.sigmoid(logits)

            # Bag-level pooling across temporal segments
            if probs.ndim == 3:
                T = probs.shape[1]
                k_val = min(max(1, int(k_top_segments)), T)
                if probs.shape[-1] == 1:
                    probs_sq = probs.squeeze(-1)
                    bag_scores = probs_sq.max(dim=1).values if k_val == 1 else probs_sq.topk(k_val, dim=1).values.mean(dim=1)
                else:
                    bag_scores = probs.max(dim=1).values if k_val == 1 else probs.topk(k_val, dim=1).values.mean(dim=1)
            elif probs.ndim == 2 and probs.shape[-1] == 1:
                bag_scores = probs.squeeze(-1)
            else:
                bag_scores = probs

            all_preds.append(bag_scores.cpu().numpy())
            all_targets.append(targets.numpy())
            all_names.extend(names)

            # Store per-video temporal scores for event simulation
            probs_cpu = probs.cpu().numpy()
            for v_idx, v_name in enumerate(names):
                v_score = probs_cpu[v_idx]
                video_scores_dict[v_name] = v_score

        # Estimate video durations from dataset spatio-temporal attributes
        ds_num_segments = getattr(self.dataset, "num_segments", 32)
        ds_clip_size = getattr(self.dataset, "clip_size", 16)
        ds_fps = getattr(self.dataset, "fps", 20.0)
        ds_stride = getattr(self.dataset, "stride", ds_clip_size)
        # Duration = first clip covers clip_size/fps seconds, each additional segment
        # advances by stride/fps seconds: clip_size/fps + (num_segments-1) * stride/fps
        base_duration = max(10.0, ds_clip_size / ds_fps + (ds_num_segments - 1) * ds_stride / ds_fps)

        gt_intervals_for_duration = getattr(self.dataset, "ground_truth_intervals", {})
        for v_name in video_scores_dict:
            # Use GT interval end time as lower bound if available (video is at least that long)
            gt_ints = gt_intervals_for_duration.get(v_name, [])
            gt_max_end = max((e for _, e in gt_ints), default=0.0) if gt_ints else 0.0
            video_durations_dict[v_name] = max(base_duration, gt_max_end + 1.0)

        y_score = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)

        # Mode detection
        if mode == "auto":
            model_num_classes = getattr(self.model, "num_classes", None)
            if model_num_classes is None and hasattr(self.model, "head"):
                model_num_classes = getattr(self.model.head, "num_classes", None)

            is_binary = (
                y_true.ndim == 1
                or (y_true.ndim == 2 and y_true.shape[1] == 1)
                or (y_score.ndim == 2 and y_score.shape[1] == 1)
                or getattr(self.dataset, "mode", None) == "binary"
                or model_num_classes == 1
            )
        else:
            is_binary = (mode.lower().strip() == "binary")

        results: Dict[str, Any] = {
            "num_samples": len(y_score),
            "mode": "binary" if is_binary else "multiclass",
            "k_top_segments": k_top_segments,
        }

        classes = _get_dataset_attr(self.dataset, "classes", None)

        # 1. Academic Benchmark Metrics
        if is_binary:
            y_true_1d = y_true.squeeze()
            y_score_1d = y_score.squeeze()
            roc_auc = compute_roc_auc(y_true_1d, y_score_1d)
            pr_auc = compute_pr_auc(y_true_1d, y_score_1d)
            results.update({
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "primary_metric": "roc_auc",
                "primary_value": roc_auc,
            })
        else:
            map_res = compute_multiclass_map(y_true, y_score, class_names=classes, iou_threshold=iou_threshold)
            auc_res = compute_multiclass_auc(y_true, y_score, class_names=classes)
            topk_res = compute_topk_accuracy(y_true, y_score, topk=topk_accuracies, anomalous_only=True)
            results.update({
                "mAP": map_res["mAP"],
                "iou_threshold": iou_threshold,
                "metric_name": map_res["metric_name"],
                "macro_auc": auc_res["macro_auc"],
                "topk_accuracy": topk_res,
                "per_class_ap": map_res["per_class"],
                "per_class_auc": auc_res["per_class"],
                "primary_metric": map_res["metric_name"],
                "primary_value": map_res["mAP"],
            })

        # 2. Production Event Simulation & Threshold Sweep (0.05 to 0.95)
        sweep_res = None
        if sweep_thresholds:
            # Query real ground truth intervals from dataset
            gt_intervals_dict = _get_dataset_attr(self.dataset, "ground_truth_intervals", {})
            if gt_intervals_dict and any(len(v) > 0 for v in gt_intervals_dict.values()):
                sweep_res = compute_threshold_sweep(
                    video_scores_dict=video_scores_dict,
                    gt_intervals_dict=gt_intervals_dict,
                    video_durations_dict=video_durations_dict,
                    tolerance_sec=tolerance_sec,
                    padding_sec=padding_sec,
                    iou_threshold=iou_threshold,
                )
                results["threshold_sweep"] = sweep_res

                # Save report to results/ directory if specified
                if save_report_dir:
                    os.makedirs(save_report_dir, exist_ok=True)
                    md_path = os.path.join(save_report_dir, "threshold_sweep_report.md")
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(sweep_res["markdown_report"])

                    json_path = os.path.join(save_report_dir, "evaluation_summary.json")
                    json_safe = {
                        k: v for k, v in results.items() if k not in ("threshold_sweep",)
                    }
                    if "threshold_sweep" in results and results["threshold_sweep"]:
                        json_safe["best_threshold"] = results["threshold_sweep"]["best_threshold"]
                        json_safe["best_f1"] = results["threshold_sweep"]["best_f1"]
                        json_safe["best_precision"] = results["threshold_sweep"]["best_precision"]
                        json_safe["best_recall"] = results["threshold_sweep"]["best_recall"]
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(json_safe, f, indent=2)
            else:
                if print_summary:
                    print("[Info] No temporal ground truth intervals found for dataset. Threshold sweep skipped.")

        if print_summary:
            self._print_summary(results, sweep_res)

        return results

    def _print_summary(self, results: Dict[str, Any], sweep_res: Optional[Dict[str, Any]] = None) -> None:
        """Pretty-prints evaluation results and threshold sweep table to the console."""
        print("\n" + "=" * 80)
        pooling_info = f" (Segment Pooling: Top-{results.get('k_top_segments', 1)})" if results.get("k_top_segments", 1) > 1 else ""
        print(f"📊 EVALUATION SUMMARY (Mode: {results['mode'].upper()}){pooling_info}")
        print("=" * 80)
        print(f"• Total Evaluated Samples: {results['num_samples']}")

        if results["mode"] == "binary":
            print(f"• ROC-AUC (Academic)     : {results['roc_auc'] * 100:.2f}% (scikit-learn)")
            print(f"• PR-AUC (Average Prec.) : {results['pr_auc'] * 100:.2f}% (scikit-learn)")
        else:
            print(f"• {results['metric_name']:<23}: {results['mAP'] * 100:.2f}% (scikit-learn)")
            print(f"• Macro ROC-AUC          : {results['macro_auc'] * 100:.2f}% (scikit-learn)")
            if "topk_accuracy" in results and results["topk_accuracy"]:
                topk_str = " | ".join(
                    [f"Top-{k}: {acc * 100:.2f}%" for k, acc in sorted(results["topk_accuracy"].items())]
                )
                print(f"• Classification Acc     : {topk_str}")
            if "per_class_ap" in results:
                print("\n  [Class-wise Average Precision]")
                for c_name, ap in results["per_class_ap"].items():
                    auc = results.get("per_class_auc", {}).get(c_name, 0.0)
                    print(f"  - {c_name:<20}: AP={ap * 100:>5.2f}% | AUC={auc * 100:>5.2f}%")

        if sweep_res is not None:
            iou_t = sweep_res.get("iou_threshold", 0.30)
            tol = sweep_res.get("tolerance_sec", 3.0)
            pad = sweep_res.get("padding_sec", 2.0)
            best_t = sweep_res.get("best_threshold", 0.35)
            best_f1 = sweep_res.get("best_f1", 0.0)
            best_p = sweep_res.get("best_precision", 0.0)
            best_r = sweep_res.get("best_recall", 0.0)

            print("\n" + "-" * 80)
            print(f"🎯 PRODUCTION EVENT SIMULATION (tIoU >= {iou_t:.2f}, tol={tol}s, pad={pad}s)")
            print("-" * 80)
            print(f"{'Eşik (tau)':<11} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | Durum")
            print("-" * 80)
            for r in sweep_res.get("table", []):
                t_val = r["threshold"]
                is_opt = (t_val == best_t)
                tag = "★ OPTIMAL" if is_opt else r.get("status", "")
                print(
                    f"{t_val:<11.2f} | {r['tp']:<4} | {r['fp']:<4} | {r['fn']:<4} | %{r['precision']*100:>6.2f}  | %{r['recall']*100:>6.2f}  | %{r['f1_score']*100:>6.2f}  | {tag}"
                )
            print("-" * 80)
            print(f"★ ÖNERİLEN OPTİMAL EŞİK : tau = {best_t:.2f}")
            print(f"• En Yüksek F1-Skoru    : %{best_f1 * 100:.2f}")
            print(f"• Alarm Doğruluğu (Prec): %{best_p * 100:.2f} (Alarmların güvenilirliği)")
            print(f"• Olay Yakalama (Recall): %{best_r * 100:.2f} (Kaçırılmayan olay oranı)")

            if "multi_tiou_summary" in sweep_res and sweep_res["multi_tiou_summary"]:
                print("\n" + "-" * 80)
                print(f"📊 KADEMELİ tIoU PERFORMANSI (Optimal tau = {best_t:.2f})")
                print("-" * 80)
                print(f"{'tIoU Eşiği':<14} | {'Referans / Standart':<32} | {'TP':<4} | {'FP':<4} | {'Precision':<9} | {'Recall':<9} | {'F1-Score'}")
                print("-" * 80)
                for row in sweep_res["multi_tiou_summary"]:
                    lbl = f"tIoU >= {row['iou_threshold']:.2f}"
                    print(
                        f"{lbl:<14} | {row['standard']:<32} | {row['tp']:<4} | {row['fp']:<4} | %{row['precision']*100:>6.2f}  | %{row['recall']*100:>6.2f}  | %{row['f1_score']*100:>6.2f}"
                    )
                print("-" * 80)
        print("=" * 80 + "\n")

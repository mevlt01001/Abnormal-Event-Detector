"""Inference and Timeline Visualization for UCF-Crime Binary Anomaly Model."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Ensure repo root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.feature_extractor import FeatureExtractor
from core.video_processor import VideoProcessor
from model.backbone import create_backbone
from ucf_binary_system.dataset import load_feature_tensor
from ucf_binary_system.model import BinaryAnomalyHead


class UCFBinaryAnalyzer:
    """Inference and timeline visualization engine for binary anomaly detection.

    Args:
        model: Trained BinaryAnomalyHead.
        device: Torch device (cuda or cpu).
        target_fps: Dynamic sampling frame rate (default: 20.0).
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[Union[str, torch.device]] = None,
        target_fps: float = 20.0,
    ) -> None:
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.processor = VideoProcessor(target_fps=target_fps)

    def _smooth_curve(self, scores_1d: np.ndarray, n_points: int) -> np.ndarray:
        """Interpolates discrete segment scores to continuous time and applies moving average."""
        orig_times = np.linspace(0, 1, len(scores_1d))
        target_times = np.linspace(0, 1, n_points)
        interp = np.interp(target_times, orig_times, scores_1d)

        kernel_size = max(5, int(n_points / 32))
        if kernel_size % 2 == 0:
            kernel_size += 1
        pad = kernel_size // 2

        padded = np.pad(interp, pad, mode="edge")
        smoothed = np.convolve(padded, np.ones(kernel_size) / kernel_size, mode="valid")
        return np.clip(smoothed, 0.0, 1.0)

    @torch.no_grad()
    def analyze_features(
        self,
        features: torch.Tensor,
        video_duration_sec: float,
        threshold: float = 0.35,
        video_name: str = "Video",
    ) -> Dict[str, Any]:
        """Runs binary forward inference on segment features.

        Args:
            features: Tensor of shape [32, feature_dim] or [1, 32, feature_dim].
            video_duration_sec: Total duration of video in seconds.
            threshold: Anomaly decision threshold (default: 0.35).
            video_name: Name identifier for reporting and plots.

        Returns:
            Dictionary containing time_axis, anomaly_scores, detected_intervals, peak_score.
        """
        if features.ndim == 2:
            features = features.unsqueeze(0)  # [1, 32, D]
        features = features.to(self.device).float()

        # [1, 32, 1] -> [32]
        probs = self.model.predict_proba(features).squeeze().cpu().numpy()

        n_points = max(100, int(video_duration_sec * 10))
        time_axis = np.linspace(0.0, video_duration_sec, n_points)
        smooth_scores = self._smooth_curve(probs, n_points)

        # Detect contiguous anomaly intervals
        above_thresh = smooth_scores >= threshold
        intervals: List[Dict[str, float]] = []
        in_event = False
        start_idx = 0

        for i, val in enumerate(above_thresh):
            if val and not in_event:
                in_event = True
                start_idx = i
            elif not val and in_event:
                in_event = False
                st = float(time_axis[start_idx])
                et = float(time_axis[i - 1])
                pk = float(np.max(smooth_scores[start_idx:i]))
                if (et - st) >= 0.5:  # Minimum 0.5s duration filter
                    intervals.append({"start_time": st, "end_time": et, "peak_score": pk})

        if in_event:
            st = float(time_axis[start_idx])
            et = float(time_axis[-1])
            pk = float(np.max(smooth_scores[start_idx:]))
            if (et - st) >= 0.5:
                intervals.append({"start_time": st, "end_time": et, "peak_score": pk})

        peak_score = float(np.max(smooth_scores))
        is_anomaly = (peak_score >= threshold) or (len(intervals) > 0)

        return {
            "video_name": video_name,
            "duration_sec": video_duration_sec,
            "threshold": threshold,
            "time_axis": time_axis,
            "anomaly_scores": smooth_scores,
            "raw_segment_scores": probs,
            "detected_intervals": intervals,
            "peak_score": peak_score,
            "is_anomaly": is_anomaly,
        }

    def render_timeline_plot(
        self,
        analysis: Dict[str, Any],
        save_path: str,
    ) -> str:
        """Renders and saves publication-grade binary anomaly timeline plot."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        time_axis = analysis["time_axis"]
        scores = analysis["anomaly_scores"]
        threshold = analysis["threshold"]
        vname = analysis["video_name"]
        duration = analysis["duration_sec"]
        intervals = analysis["detected_intervals"]
        peak = analysis["peak_score"]
        is_anom = analysis["is_anomaly"]

        fig, ax = plt.subplots(figsize=(14, 6.0), dpi=140, facecolor="#0a0e17")
        ax.set_facecolor("#05070c")

        # Color palette: Vibrant Neon Cyan/Mint for normal-to-mid, Neon Coral/Red for anomaly
        curve_color = "#00f5d4" if not is_anom else "#ff3366"

        # 1. Shaded detected anomaly intervals
        for idx, det in enumerate(intervals):
            lbl = f"Tespit Edilen Anomali (>= {threshold:.2f})" if idx == 0 else None
            ax.axvspan(det["start_time"], det["end_time"], color="#ff3366", alpha=0.22, label=lbl, zorder=2)
            ax.text(
                (det["start_time"] + det["end_time"]) / 2.0, 0.05,
                f"ANOMALİ: %{det['peak_score']*100:.1f}\n[{det['start_time']:.1f}s - {det['end_time']:.1f}s]",
                color="#ff6b8b", fontsize=8.5, fontweight="bold", ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e0008", edgecolor="#ff3366", alpha=0.85),
                zorder=5,
            )

        # 2. Main Binary Anomaly Curve (Each plot in its own vibrant color, no white overlay!)
        ax.plot(time_axis, scores, color=curve_color, linewidth=2.6, label=f"Anomali Skoru (Pik: %{peak*100:.1f})", zorder=4)

        # 3. Decision Threshold Line
        ax.axhline(y=threshold, color="#ffcc00", linestyle="--", linewidth=1.8, label=f"Karar Eşiği ({threshold:.2f})", zorder=6)

        # Status badge in upper-left corner
        status_text = f"DURUM: {'[!] ANOMALİ TESPİT EDİLDİ' if is_anom else '[OK] TEMİZ / NORMAL'}"
        status_bg = "#33000d" if is_anom else "#002b18"
        status_fg = "#ff4d6d" if is_anom else "#00ff88"
        ax.text(
            0.02, 0.92, status_text,
            transform=ax.transAxes, color=status_fg, fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=status_bg, edgecolor=status_fg, alpha=0.85),
            zorder=6,
        )

        ax.set_xlim(0, duration)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Zaman (Saniye)", color="#8892b0", fontsize=10)
        ax.set_ylabel("Anomali Olasılığı [Sigmoid: 0.0 - 1.0]", color="#8892b0", fontsize=10)
        ax.set_title(f"UCF-Crime İkili Anomali Analizi (Binary Model): {vname}", color="#ffffff", fontsize=12, fontweight="bold", pad=12)
        ax.grid(color="#1a2233", linestyle="--", alpha=0.5, zorder=1)
        ax.tick_params(colors="#8892b0")
        ax.legend(loc="upper right", facecolor="#0d1322", edgecolor="#2a3b5c", fontsize=8.5, labelcolor="#e2e8f0")

        for spine in ax.spines.values():
            spine.set_color("#1f293d")

        plt.savefig(save_path, dpi=140, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
        plt.close(fig)
        return save_path

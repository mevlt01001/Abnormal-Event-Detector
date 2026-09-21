"""High-level video inference and class-aware anomaly analysis engine."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.video_processor import VideoProcessor
from data.taxonomy import DEFAULT_MACRO_CLASSES, get_class_color_map


class VideoAnalyzer:
    """Class-aware video anomaly inference and localization engine.

    Args:
        model: AnomalyDetector PyTorch module or AnomalyHead.
        class_names: List of class names corresponding to output neurons.
        device: Torch device (cuda or cpu).
        target_fps: Dynamic target frame rate for video sampling.
    """

    def __init__(
        self,
        model: nn.Module,
        class_names: Optional[Sequence[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        target_fps: float = 20.0,
    ) -> None:
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.class_names = list(class_names) if class_names else list(DEFAULT_MACRO_CLASSES)
        self.color_map = get_class_color_map(self.class_names)
        self.processor = VideoProcessor(target_fps=target_fps)

    def _smooth_curve(self, curve_1d: np.ndarray, n_points: int) -> np.ndarray:
        """Interpolates 32 discrete segment scores to continuous time and applies moving average."""
        orig_times = np.linspace(0, 1, len(curve_1d))
        target_times = np.linspace(0, 1, n_points)
        interp = np.interp(target_times, orig_times, curve_1d)

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
        video_name: str = "Video Analysis",
    ) -> Dict[str, Any]:
        """Analyzes pre-extracted segment feature tensor.

        Args:
            features: Tensor of shape [num_segments, feature_dim] or [1, num_segments, feature_dim].
            video_duration_sec: Video length in seconds.
            threshold: Confidence threshold for anomaly detection (default: 0.35).
            video_name: Identifier for reporting and plot title.

        Returns:
            Dictionary containing:
                time_axis: 1D array of seconds
                class_scores: Array of shape [N_points, num_classes] in [0, 1]
                overall_scores: Array of shape [N_points] representing max envelope
                detected_intervals: List of detected anomaly time intervals
                detected_classes: Summary of classes exceeding threshold
                peak_score: Maximum global anomaly confidence
        """
        # Ensure features shape: [1, T, D]
        if features.ndim == 2:
            # [T, D] -> [1, T, D]
            features = features.unsqueeze(0)
        features = features.to(self.device).float()

        # Forward pass through model or head
        if hasattr(self.model, "forward_features"):
            # [1, T, D] -> [1, T, num_classes]
            logits = self.model.forward_features(features)
        else:
            # [1, T, D] -> [1, T, num_classes]
            logits = self.model(features)

        # [1, T, num_classes] -> [T, num_classes] probabilities in [0, 1]
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        num_points = max(100, int(video_duration_sec * 10))
        time_axis = np.linspace(0, video_duration_sec, num_points)

        smooth_class_scores = np.zeros((num_points, probs.shape[1]), dtype=np.float32)
        for c in range(probs.shape[1]):
            smooth_class_scores[:, c] = self._smooth_curve(probs[:, c], num_points)

        overall_envelope = np.max(smooth_class_scores, axis=1)
        peak_score = float(np.max(overall_envelope))

        # Detect continuous temporal intervals where score >= threshold
        detected_intervals = self.extract_intervals(
            time_axis=time_axis,
            overall_scores=overall_envelope,
            class_scores=smooth_class_scores,
            threshold=threshold,
        )

        # Identify classes that exceeded threshold
        detected_classes = []
        for c_idx, c_name in enumerate(self.class_names):
            c_peak = float(np.max(smooth_class_scores[:, c_idx]))
            if c_peak >= threshold:
                detected_classes.append({"class": c_name, "peak_confidence": c_peak})
        detected_classes.sort(key=lambda x: x["peak_confidence"], reverse=True)

        return {
            "video_name": video_name,
            "duration_sec": video_duration_sec,
            "threshold": threshold,
            "time_axis": time_axis,
            "class_scores": smooth_class_scores,
            "overall_scores": overall_envelope,
            "detected_intervals": detected_intervals,
            "detected_classes": detected_classes,
            "peak_score": peak_score,
            "is_anomaly": len(detected_intervals) > 0,
        }

    def extract_intervals(
        self,
        time_axis: np.ndarray,
        overall_scores: np.ndarray,
        class_scores: np.ndarray,
        threshold: float = 0.35,
        tolerance_sec: float = 2.0,
        min_duration_sec: float = 0.4,
    ) -> List[Dict[str, Any]]:
        """Extracts continuous anomaly intervals from temporal score curves."""
        indices = np.where(overall_scores >= threshold)[0]
        if len(indices) == 0:
            return []

        intervals = []
        curr_start = indices[0]
        curr_end = indices[0]

        for idx in indices[1:]:
            gap = time_axis[idx] - time_axis[curr_end]
            if gap <= tolerance_sec:
                curr_end = idx
            else:
                s_t = float(time_axis[curr_start])
                e_t = float(time_axis[curr_end])
                if (e_t - s_t) >= min_duration_sec:
                    span_classes = class_scores[curr_start:curr_end + 1]
                    class_peaks = np.max(span_classes, axis=0)
                    top_c_idx = int(np.argmax(class_peaks))
                    top_class = self.class_names[top_c_idx]
                    peak = float(np.max(overall_scores[curr_start:curr_end + 1]))
                    intervals.append({
                        "start_time": s_t,
                        "end_time": e_t,
                        "duration": e_t - s_t,
                        "top_class": top_class,
                        "peak_score": peak,
                    })
                curr_start = idx
                curr_end = idx

        s_t = float(time_axis[curr_start])
        e_t = float(time_axis[curr_end])
        if (e_t - s_t) >= min_duration_sec:
            span_classes = class_scores[curr_start:curr_end + 1]
            class_peaks = np.max(span_classes, axis=0)
            top_c_idx = int(np.argmax(class_peaks))
            top_class = self.class_names[top_c_idx]
            peak = float(np.max(overall_scores[curr_start:curr_end + 1]))
            intervals.append({
                "start_time": s_t,
                "end_time": e_t,
                "duration": e_t - s_t,
                "top_class": top_class,
                "peak_score": peak,
            })

        return intervals

    def render_timeline_plot(
        self,
        analysis_result: Dict[str, Any],
        save_path: str,
        gt_intervals: Optional[List[Tuple[float, float]]] = None,
    ) -> str:
        """Renders and saves a publication-quality anomaly timeline chart."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        time_axis = analysis_result["time_axis"]
        class_scores = analysis_result["class_scores"]
        overall_scores = analysis_result["overall_scores"]
        threshold = analysis_result["threshold"]
        video_name = analysis_result["video_name"]
        duration = analysis_result["duration_sec"]
        detected_intervals = analysis_result["detected_intervals"]

        fig = plt.figure(figsize=(15, 8.5), dpi=140, facecolor="#0a0e17")
        gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.32)
        ax_time = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1])

        ax_time.set_facecolor("#05070c")
        ax_bar.set_facecolor("#05070c")

        # 1. Ground Truth Span (if provided)
        if gt_intervals:
            for idx, (gt_s, gt_e) in enumerate(gt_intervals):
                lbl = "Gerçek Olay (Ground Truth)" if idx == 0 else None
                ax_time.axvspan(gt_s, gt_e, color="#00ff88", alpha=0.20, label=lbl, zorder=1)
        else:
            ax_time.text(
                0.02, 0.93, "Ground Truth: NORMAL",
                transform=ax_time.transAxes, color="#00ff88", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#002b18", edgecolor="#00ff88", alpha=0.8),
                zorder=5,
            )

        # 2. Detected Anomaly Spans (shaded in detected class's own color)
        for idx, det in enumerate(detected_intervals):
            top_class = det["top_class"]
            span_color = self.color_map.get(top_class, "#ff3366")
            lbl = f"Tespit: {top_class} (>= {threshold})" if idx == 0 else None
            ax_time.axvspan(det["start_time"], det["end_time"], color=span_color, alpha=0.18, label=lbl, zorder=2)
            ax_time.text(
                (det["start_time"] + det["end_time"]) / 2.0, 0.04,
                f"{top_class}\n%{det['peak_score']*100:.1f} [{det['start_time']:.1f}s-{det['end_time']:.1f}s]",
                color=span_color, fontsize=8, fontweight="bold", ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0a0f1d", edgecolor=span_color, alpha=0.85),
                zorder=5,
            )

        # 3. Class curves - Each class in its own distinct color (no white override)
        for c_idx, c_name in enumerate(self.class_names):
            col = self.color_map.get(c_name, "#ffffff")
            c_peak = float(np.max(class_scores[:, c_idx]))
            alpha = 0.95 if c_peak >= threshold else 0.45
            lw = 2.4 if c_peak >= threshold else 1.2
            ax_time.plot(
                time_axis,
                class_scores[:, c_idx],
                color=col,
                alpha=alpha,
                linewidth=lw,
                label=f"{c_name} (%{c_peak*100:.1f})",
                zorder=4 if c_peak >= threshold else 3,
            )

        # 4. Threshold Line (Dashed Gold/Yellow)
        ax_time.axhline(
            y=threshold,
            color="#ffcc00",
            linestyle="--",
            linewidth=1.8,
            label=f"Eşik ({threshold:.2f})",
            zorder=6,
        )

        ax_time.set_xlim(0, duration)
        ax_time.set_ylim(-0.02, 1.05)
        ax_time.set_ylabel("Anomali Olasılığı [0, 1]", color="#8892b0", fontsize=10)
        ax_time.set_title(f"Zaman Serisi Anomali Analizi: {video_name}", color="#ffffff", fontsize=12, fontweight="bold", pad=10)
        ax_time.grid(color="#1a2233", linestyle="--", alpha=0.5, zorder=0)
        ax_time.tick_params(colors="#8892b0")
        ax_time.legend(loc="upper right", facecolor="#0d1322", edgecolor="#2a3b5c", fontsize=8, labelcolor="#e2e8f0", ncol=2)

        # 5. Bottom Peak Confidence Bars
        class_peaks = [float(np.max(class_scores[:, c_idx])) * 100 for c_idx in range(len(self.class_names))]
        colors = [self.color_map.get(c, "#888888") for c in self.class_names]
        bars = ax_bar.barh(self.class_names, class_peaks, color=colors, alpha=0.85, height=0.55, edgecolor="#ffffff", linewidth=0.5)
        ax_bar.axvline(x=threshold * 100, color="#ffcc00", linestyle="--", linewidth=1.5)
        ax_bar.set_xlim(0, 100)
        ax_bar.set_xlabel("Pik Güven Skoru (%)", color="#8892b0", fontsize=9)
        ax_bar.grid(color="#1a2233", linestyle="--", alpha=0.5, axis="x")
        ax_bar.tick_params(colors="#8892b0")

        for bar, val in zip(bars, class_peaks):
            ax_bar.text(val + 1.5, bar.get_y() + bar.get_height() / 2.0, f"%{val:.1f}", color="#e2e8f0", va="center", fontsize=8, fontweight="bold")

        for ax in [ax_time, ax_bar]:
            for spine in ax.spines.values():
                spine.set_color("#1f293d")

        plt.savefig(save_path, dpi=140, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
        plt.close(fig)
        return save_path

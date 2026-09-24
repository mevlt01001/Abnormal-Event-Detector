"""Anomaly segment extraction and clean timeline visualization engine.

Implements exact temporal segment extraction (threshold, tolerance_sec, padding_sec)
and publication-grade anomaly timeline charting.
"""

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

from data.taxonomy import get_class_color_map


def extract_anomaly_segments(
    scores: Union[torch.Tensor, np.ndarray],
    video_seconds: float,
    threshold: float = 0.30,
    tolerance_sec: float = 3.0,
    padding_sec: float = 2.0,
    class_names: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], torch.Tensor, torch.Tensor]:
    """Extracts continuous anomaly segments with tolerance gap merging and security padding.

    Faithfully adopts the proven 1D interpolation, reflection padding, moving-average pooling,
    and interval clustering pipeline.

    Args:
        scores: Continuous anomaly scores. Can be:
            - 1D Tensor/array [T] (binary anomaly scores)
            - 2D Tensor/array [T, C] (multi-class class probabilities)
        video_seconds: Total duration of video in seconds.
        threshold: Minimum score threshold to trigger an anomaly event (default: 0.30).
        tolerance_sec: Maximum time gap (in seconds) between positive frames to merge
            them into the same contiguous event (default: 3.0).
        padding_sec: Security padding (in seconds) prepended and appended to each
            detected segment (default: 2.0).
        class_names: Optional sequence of class names if scores is 2D.

    Returns:
        Tuple of:
            final_segments: List of segment dictionaries with keys:
                'start_time', 'end_time', 'duration', 'score', and 'top_class'.
            scores_smooth: Smoothed 1D envelope curve of length interpolate_size.
            scores_raw: Linearly interpolated raw curve before smoothing.
    """
    if isinstance(scores, np.ndarray):
        scores_t = torch.from_numpy(scores).float()
    else:
        scores_t = scores.clone().detach().float().cpu()

    # Determine 1D envelope and multi-class representation
    is_multiclass = (scores_t.ndim == 2 and scores_t.shape[-1] > 1)
    if is_multiclass:
        # scores_org is max across classes: [T]
        scores_org_1d, top_indices = scores_t.max(dim=-1)
    else:
        scores_org_1d = scores_t.squeeze()
        if scores_org_1d.ndim == 0:
            scores_org_1d = scores_org_1d.unsqueeze(0)

    # Format for 1D convolution: [1, 1, T]
    scores_org = scores_org_1d.unsqueeze(0).unsqueeze(0)

    kernel_size = 11
    video_sec = max(0.1, float(video_seconds))
    interpolate_size = max(len(scores_org_1d), int(video_sec * 10))

    # 1. 1D Linear and Nearest Interpolation
    scores_linear_interpolate = F.interpolate(
        scores_org, size=interpolate_size, mode="linear", align_corners=True
    )
    scores_raw_1d = scores_linear_interpolate.squeeze(0).squeeze(0)

    # 2. Moving average smoothing with reflection padding
    pad_size = kernel_size // 2
    scores_padded = F.pad(scores_linear_interpolate, (pad_size, pad_size), mode="reflect")
    scores_smooth = F.avg_pool1d(scores_padded, kernel_size=kernel_size, stride=1)
    scores_smooth_1d = scores_smooth.squeeze(0).squeeze(0)

    dt = video_sec / float(interpolate_size)
    anomaly_indices = torch.where(scores_smooth_1d >= threshold)[0].tolist()

    final_segments: List[Dict[str, Any]] = []

    if anomaly_indices:
        # Group adjacent or close detections within tolerance_sec
        raw_segments: List[Tuple[int, int]] = []
        current_start = anomaly_indices[0]
        current_end = anomaly_indices[0]

        for idx in anomaly_indices[1:]:
            # Gap between end of current segment and start of next sample:
            time_gap = max(0.0, (idx - current_end - 1) * dt)
            if time_gap <= tolerance_sec:
                current_end = idx
            else:
                raw_segments.append((current_start, current_end))
                current_start = idx
                current_end = idx
        raw_segments.append((current_start, current_end))

        # Add security padding_sec
        padded_segments: List[Tuple[float, float, int, int]] = []
        for s_idx, e_idx in raw_segments:
            s_time = s_idx * dt
            e_time = (e_idx + 1) * dt
            p_start = max(0.0, s_time - padding_sec)
            p_end = min(video_sec, e_time + padding_sec)
            padded_segments.append((p_start, p_end, s_idx, e_idx))

        # Merge overlapping padded segments
        merged_segments: List[Tuple[float, float, int, int]] = []
        curr_p_start, curr_p_end, curr_s_idx, curr_e_idx = padded_segments[0]

        for p_start, p_end, s_idx, e_idx in padded_segments[1:]:
            if p_start <= curr_p_end:
                curr_p_end = max(curr_p_end, p_end)
                curr_e_idx = e_idx
            else:
                merged_segments.append((curr_p_start, curr_p_end, curr_s_idx, curr_e_idx))
                curr_p_start, curr_p_end, curr_s_idx, curr_e_idx = p_start, p_end, s_idx, e_idx
        merged_segments.append((curr_p_start, curr_p_end, curr_s_idx, curr_e_idx))

        # Build segment dictionaries with peak confidence and top class
        for p_start, p_end, s_idx, e_idx in merged_segments:
            seg_scores = scores_smooth_1d[s_idx : e_idx + 1]
            max_score = float(seg_scores.max().item()) if len(seg_scores) > 0 else float(scores_smooth_1d[s_idx].item())

            top_class = "Anomaly"
            if is_multiclass and class_names is not None:
                # Map interpolated indices back to original temporal steps
                orig_len = len(scores_t)
                orig_s = min(orig_len - 1, int(s_idx / interpolate_size * orig_len))
                orig_e = min(orig_len - 1, int(e_idx / interpolate_size * orig_len))
                sub_class_scores = scores_t[orig_s : orig_e + 1]
                if len(sub_class_scores) > 0:
                    class_peaks = sub_class_scores.max(dim=0).values
                    best_c_idx = int(class_peaks.argmax().item())
                    if best_c_idx < len(class_names):
                        top_class = class_names[best_c_idx]

            final_segments.append({
                "start_time": round(float(p_start), 2),
                "end_time": round(float(p_end), 2),
                "duration": round(float(p_end - p_start), 2),
                "score": round(max_score, 4),
                "top_class": top_class,
            })

    return final_segments, scores_smooth_1d, scores_raw_1d


def plot_anomaly_timeline(
    scores_smooth: Union[torch.Tensor, np.ndarray],
    scores_raw: Optional[Union[torch.Tensor, np.ndarray]],
    segments: List[Dict[str, Any]],
    video_seconds: float,
    threshold: float = 0.30,
    save_path: Optional[str] = "anomaly_timeline.png",
    video_name: str = "Video Analysis",
    class_names: Optional[Sequence[str]] = None,
    scores_multiclass: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> str:
    """Renders and saves a clean, publication-grade anomaly timeline chart.

    Adheres strictly to the clear, single-figure layout from visualization_tools.py
    and uses colors mapped directly from data.taxonomy.

    Args:
        scores_smooth: 1D array of smoothed anomaly scores.
        scores_raw: Optional 1D array of raw interpolated anomaly scores.
        segments: List of detected segment dicts with start_time, end_time, score, top_class.
        video_seconds: Total duration of the video.
        threshold: Anomaly confidence threshold line.
        save_path: Filepath where plot image (.png) will be written.
        video_name: Title identifier of the video.
        class_names: Optional list of class names for color assignment.

    Returns:
        Absolute filepath to the saved image.
    """
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    if isinstance(scores_smooth, torch.Tensor):
        s_smooth = scores_smooth.detach().cpu().numpy()
    else:
        s_smooth = np.asarray(scores_smooth)

    time_axis = np.linspace(0, max(0.1, video_seconds), len(s_smooth))

    color_map = get_class_color_map(list(class_names)) if class_names else {}

    fig, ax = plt.subplots(figsize=(15, 6), dpi=130)

    # 1. Raw scores curve (if provided and no multi-class given)
    if scores_raw is not None and scores_multiclass is None:
        if isinstance(scores_raw, torch.Tensor):
            s_raw = scores_raw.detach().cpu().numpy()
        else:
            s_raw = np.asarray(scores_raw)
        ax.plot(time_axis, s_raw, color="#b6b971", alpha=0.35, linewidth=1.5, label="Ham Anomali Skoru")

    # 2. Multi-class specific curves
    if scores_multiclass is not None:
        if isinstance(scores_multiclass, np.ndarray):
            mc_t = torch.from_numpy(scores_multiclass).float()
        else:
            mc_t = scores_multiclass.clone().detach().cpu().float()
        
        # Smooth multi-class scores for plot to match envelope resolution
        kernel_size = 11
        pad_size = kernel_size // 2
        mc_t = mc_t.transpose(0, 1).unsqueeze(0) # [1, C, T]
        mc_interp = F.interpolate(mc_t, size=len(s_smooth), mode="linear", align_corners=True)
        mc_pad = F.pad(mc_interp, (pad_size, pad_size), mode="reflect")
        mc_smooth = F.avg_pool1d(mc_pad, kernel_size=kernel_size, stride=1)
        mc_smooth_np = mc_smooth.squeeze(0).transpose(0, 1).numpy() # [len, C]

        for c in range(mc_smooth_np.shape[1]):
            c_name = class_names[c] if class_names and c < len(class_names) else f"Sınıf_{c}"
            c_color = color_map.get(c_name, None)
            ax.plot(time_axis, mc_smooth_np[:, c], color=c_color, linewidth=2.0, alpha=0.9, label=f"Skor: {c_name}", zorder=4)
            
        # Draw the max envelope slightly faded or distinct
        ax.plot(time_axis, s_smooth, color="black", linestyle=":", linewidth=2.0, alpha=0.6, label="Zarf (Max Sınıf)", zorder=3)
    else:
        # Binary or non-multiclass: just draw the smoothed envelope
        ax.plot(time_axis, s_smooth, color="#1f77b4", linewidth=2.5, label="Anomali Skoru (Yumuşatılmış)", zorder=4)

    # 3. Threshold line
    ax.axhline(
        y=threshold, color="#d62728", linestyle="--", linewidth=1.8, label=f"Eşik Değeri ({threshold:.2f})"
    )

    # 4. Anomaly spans with class-tailored shading
    for idx, seg in enumerate(segments):
        start = seg["start_time"]
        end = seg["end_time"]
        c_label = seg.get("top_class", "Anomali")
        c_color = color_map.get(c_label, "red")

        label = f"Tespit: {c_label}" if idx == 0 else None
        ax.axvspan(start, end, color=c_color, alpha=0.20, label=label)

    ax.set_title(f"Video Anomali Zaman Çizelgesi: {video_name}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Zaman (Saniye)", fontsize=12)
    ax.set_ylabel("Anomali Olasılığı [0, 1]", fontsize=12)

    # 5. Precise and frequent ticks (25 X-ticks, 10 Y-ticks)
    x_ticks = np.linspace(0, max(0.1, video_seconds), 25)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{t:.1f}s" for t in x_ticks], rotation=35, ha="right", fontsize=8)

    y_ticks = np.arange(0.0, 1.05, 0.10)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y:.1f}" for y in y_ticks], fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0, max(0.1, video_seconds))

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=10)
    plt.tight_layout()

    out_path = save_path if save_path else "anomaly_timeline.png"
    plt.savefig(out_path)
    plt.close(fig)
    return os.path.abspath(out_path)


def generate_video_markdown_report(
    video_name: str,
    video_seconds: float,
    threshold: float,
    segments: List[Dict[str, Any]],
    graph_filename: Optional[str] = "anomaly_timeline.png",
    save_path: Optional[str] = None,
) -> str:
    """Generates an executive, publication-grade Markdown report for a predicted video."""
    num_events = len(segments)
    status_emoji = "🚨 Anomali Tespit Edildi" if num_events > 0 else "✅ Normal (Temiz)"

    report_lines = [
        f"# Video Anomali Analiz Raporu: {video_name}",
        "",
        "## 📊 Genel Özet",
        "",
        "| Parametre | Değer |",
        "| :--- | :--- |",
        f"| **Video Adı** | `{video_name}` |",
        f"| **Video Süresi** | `{video_seconds:.2f} saniye` |",
        f"| **Anomali Eşik Değeri** | `{threshold:.2f}` |",
        f"| **Genel Durum** | **{status_emoji}** |",
        f"| **Tespit Edilen Olay Sayısı** | **{num_events}** |",
        "",
    ]

    if graph_filename:
        report_lines.extend([
            "## 📈 Anomali Zaman Çizelgesi Grafiği",
            "",
            f"![{video_name} Zaman Çizelgesi]({graph_filename})",
            "",
        ])

    report_lines.extend([
        "## 🔍 Tespit Edilen Anomali Olayları",
        "",
    ])

    if num_events == 0:
        report_lines.append(f"> [!NOTE]\n> Belirlenen eşik değeri (`{threshold:.2f}`) üzerinde herhangi bir anomali olayı tespit edilmemiştir. Video normal kabul edilmiştir.\n")
    else:
        has_clips = any("clip_path" in s for s in segments)
        if has_clips:
            report_lines.extend([
                "| Olay # | Başlangıç | Bitiş | Süre | Sınıf | Tepe Skoru | Video Klip |",
                "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ])
            for i, seg in enumerate(segments, 1):
                st = seg.get("start_time", 0.0)
                et = seg.get("end_time", 0.0)
                dur = seg.get("duration", et - st)
                cls_name = seg.get("top_class", "Anomali")
                score = seg.get("score", 0.0)
                clip_p = seg.get("clip_path", "-")
                clip_md = f"[{os.path.basename(clip_p)}]({os.path.basename(clip_p)})" if clip_p and clip_p != "-" else "-"
                report_lines.append(f"| {i:02d} | {st:.2f}s | {et:.2f}s | {dur:.2f}s | **{cls_name}** | `{score:.4f}` | {clip_md} |")
        else:
            report_lines.extend([
                "| Olay # | Başlangıç | Bitiş | Süre | Sınıf | Tepe Skoru |",
                "| :---: | :---: | :---: | :---: | :---: | :---: |",
            ])
            for i, seg in enumerate(segments, 1):
                st = seg.get("start_time", 0.0)
                et = seg.get("end_time", 0.0)
                dur = seg.get("duration", et - st)
                cls_name = seg.get("top_class", "Anomali")
                score = seg.get("score", 0.0)
                report_lines.append(f"| {i:02d} | {st:.2f}s | {et:.2f}s | {dur:.2f}s | **{cls_name}** | `{score:.4f}` |")

        report_lines.append("")

    report_content = "\n".join(report_lines)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report_content)

    return report_content


class VideoAnalyzer:
    """Convenience class wrapping model inference and segment visualization."""

    def __init__(
        self,
        model: nn.Module,
        class_names: Optional[Sequence[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.class_names = list(class_names) if class_names else None

    @torch.no_grad()
    def analyze_features(
        self,
        features: torch.Tensor,
        video_duration_sec: float,
        threshold: float = 0.30,
        tolerance_sec: float = 3.0,
        padding_sec: float = 2.0,
        save_plot: Optional[str] = None,
        video_name: str = "Feature Analysis",
    ) -> Dict[str, Any]:
        """Runs forward pass on feature tensor and extracts anomaly segments."""
        if features.ndim == 2:
            features = features.unsqueeze(0)
        features = features.to(self.device).float()

        if hasattr(self.model, "forward_features"):
            logits = self.model.forward_features(features)
        else:
            logits = self.model(features)

        probs = torch.sigmoid(logits).squeeze(0).cpu()

        segments, scores_smooth, scores_raw = extract_anomaly_segments(
            scores=probs,
            video_seconds=video_duration_sec,
            threshold=threshold,
            tolerance_sec=tolerance_sec,
            padding_sec=padding_sec,
            class_names=self.class_names,
        )

        plot_path = None
        if save_plot:
            plot_path = plot_anomaly_timeline(
                scores_smooth=scores_smooth,
                scores_raw=scores_raw,
                segments=segments,
                video_seconds=video_duration_sec,
                threshold=threshold,
                save_path=save_plot,
                video_name=video_name,
                class_names=self.class_names,
                scores_multiclass=probs if (probs.ndim == 2 and probs.shape[-1] > 1) else None,
            )

        return {
            "video_name": video_name,
            "duration_sec": video_duration_sec,
            "threshold": threshold,
            "detected_segments": segments,
            "scores_smooth": scores_smooth.numpy(),
            "plot_path": plot_path,
            "is_anomaly": len(segments) > 0,
        }

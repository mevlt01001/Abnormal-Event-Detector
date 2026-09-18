from __future__ import annotations

import os
from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_anomaly_timeline(
    scores: Union[torch.Tensor, np.ndarray],
    scores_raw: Optional[Union[torch.Tensor, np.ndarray]] = None,
    segments: Optional[List[Dict[str, Any]]] = None,
    video_seconds: float = 100.0,
    threshold: float = 0.3,
    save_path: Optional[str] = "anomaly_segmentation_plot.png",
    title: str = "Video Anomaly Timeline",
) -> None:
    """Plots anomaly score curve over time, highlighting detected anomaly segments.

    Args:
        scores: Smoothed anomaly score array or tensor.
        scores_raw: Optional raw anomaly score array or tensor before smoothing.
        segments: List of anomaly segments with 'start_time', 'end_time', and optionally 'score'.
        video_seconds: Total duration of the video in seconds.
        threshold: Anomaly decision threshold.
        save_path: File path to save the generated plot image.
        title: Title of the plot.
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    scores = np.asarray(scores).squeeze()

    time_axis = np.linspace(0, video_seconds, len(scores))

    plt.figure(figsize=(15, 6), dpi=120)

    if scores_raw is not None:
        if isinstance(scores_raw, torch.Tensor):
            scores_raw = scores_raw.detach().cpu().numpy()
        scores_raw = np.asarray(scores_raw).squeeze()
        plt.plot(
            time_axis,
            scores_raw,
            color="#b6b971",
            alpha=0.45,
            linewidth=1.2,
            label="Raw Anomaly Score",
        )

    plt.plot(
        time_axis,
        scores,
        color="#1f77b4",
        linewidth=2.5,
        label="Smoothed Anomaly Score",
    )
    plt.axhline(
        y=threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({threshold:.2f})",
    )

    if segments:
        for idx, seg in enumerate(segments):
            start = seg.get("start_time", 0.0)
            end = seg.get("end_time", 0.0)
            score_val = seg.get("score")

            label = "Anomaly Segment" if idx == 0 else None
            plt.axvspan(start, end, color="red", alpha=0.2, label=label)

            tag = f"{start:.1f}s - {end:.1f}s"
            if score_val is not None:
                tag += f" ({score_val:.2f})"
            plt.text(
                start,
                1.02,
                f"{start:.1f}s",
                color="darkred",
                fontsize=9,
                ha="right",
                rotation=45,
            )
            plt.text(
                end,
                1.02,
                f"{end:.1f}s",
                color="darkred",
                fontsize=9,
                ha="left",
                rotation=45,
            )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Anomaly Probability", fontsize=12)
    plt.ylim(-0.05, 1.1)
    plt.xlim(0, max(video_seconds, 1.0))
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path)
    plt.close()

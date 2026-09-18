import os
import pytest
import torch
import numpy as np

from utils.video_utils import process_clip_tensor
from utils.visualization import plot_anomaly_timeline


def test_process_clip_tensor_padding():
    # Test shorter sequence than clip_size
    frames = np.random.randint(0, 256, (10, 64, 64, 3), dtype=np.uint8)
    tensor = process_clip_tensor(frames, clip_size=16)
    assert tensor.shape == (1, 3, 16, 64, 64)


def test_process_clip_tensor_truncation():
    # Test longer sequence than clip_size
    frames = np.random.randint(0, 256, (20, 64, 64, 3), dtype=np.uint8)
    tensor = process_clip_tensor(frames, clip_size=16)
    assert tensor.shape == (1, 3, 16, 64, 64)


def test_plot_anomaly_timeline(tmp_path):
    scores = np.linspace(0.1, 0.9, 50)
    scores_raw = np.random.rand(50)
    segments = [
        {"start_time": 5.0, "end_time": 15.0, "duration": 10.0, "score": 0.85},
        {"start_time": 25.0, "end_time": 30.0, "duration": 5.0, "score": 0.72},
    ]
    out_img = str(tmp_path / "plot.png")

    plot_anomaly_timeline(
        scores=scores,
        scores_raw=scores_raw,
        segments=segments,
        video_seconds=40.0,
        threshold=0.3,
        save_path=out_img,
    )

    assert os.path.exists(out_img)
    assert os.path.getsize(out_img) > 1000

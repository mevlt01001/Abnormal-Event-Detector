"""Unit tests for Decord-based VideoProcessor."""

import os
import pytest
import torch

from core.video_processor import VideoProcessor

SAMPLE_VIDEO = "Test_Videos/test_video_0.mp4"


@pytest.mark.skipif(not os.path.isfile(SAMPLE_VIDEO), reason="Sample video not found")
def test_video_processor_metadata():
    processor = VideoProcessor(target_fps=20.0, clip_size=16)
    meta = processor.get_video_metadata(SAMPLE_VIDEO)
    assert "duration_sec" in meta
    assert meta["duration_sec"] > 0.0
    assert meta["height"] > 0
    assert meta["width"] > 0


@pytest.mark.skipif(not os.path.isfile(SAMPLE_VIDEO), reason="Sample video not found")
def test_video_processor_uniform_segments():
    processor = VideoProcessor(target_fps=15.0, clip_size=16)
    gen = processor.extract_uniform_segments(SAMPLE_VIDEO, num_segments=4)

    segments = list(gen)
    assert len(segments) == 4
    for seg in segments:
        # Shape: [K, C=3, clip_size=16, H=224, W=224]
        assert seg.ndim == 5
        assert seg.shape[1] == 3
        assert seg.shape[2] == 16
        assert seg.shape[3] == 224
        assert seg.shape[4] == 224
        assert (seg >= 0.0).all() and (seg <= 1.0).all()


@pytest.mark.skipif(not os.path.isfile(SAMPLE_VIDEO), reason="Sample video not found")
def test_video_processor_sliding_clips():
    processor = VideoProcessor(target_fps=10.0, clip_size=16, stride=16)
    gen, num_clips, duration = processor.extract_sliding_clips(SAMPLE_VIDEO, batch_size=2)
    assert num_clips > 0
    first_batch = next(gen)
    # [B<=2, C=3, clip_size=16, H=224, W=224]
    assert first_batch.ndim == 5
    assert first_batch.shape[1] == 3
    assert first_batch.shape[2] == 16


@pytest.mark.skipif(not os.path.isfile(SAMPLE_VIDEO), reason="Sample video not found")
def test_video_processor_max_clips_per_segment():
    # Force max_clips_per_segment = 3 with high overlap (stride=2)
    processor = VideoProcessor(target_fps=20.0, clip_size=16, stride=2, max_clips_per_segment=3)
    gen = processor.extract_uniform_segments(SAMPLE_VIDEO, num_segments=2)
    segments = list(gen)
    assert len(segments) == 2
    for seg in segments:
        # Number of clips K in each segment must be exactly capped at 3
        assert seg.shape[0] == 3
        assert seg.shape[1] == 3
        assert seg.shape[2] == 16
        assert seg.shape[3] == 224
        assert seg.shape[4] == 224

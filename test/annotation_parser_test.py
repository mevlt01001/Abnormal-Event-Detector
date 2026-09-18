import os
import numpy as np
import pytest
import torch

from utils.annotation_parser import (
    XD_CLASSES,
    CLASS_NAMES,
    strip_video_ext,
    is_normal_video,
    get_video_classes,
    get_multi_hot_label,
    parse_annotations,
    generate_frame_gt,
    generate_segment_gt,
    expand_segment_scores_to_frames,
)


def test_strip_video_ext():
    assert strip_video_ext("video1.mp4") == "video1"
    assert strip_video_ext("Bad.Boys.1995__#01_label_G.mp4") == "Bad.Boys.1995__#01_label_G"
    assert strip_video_ext("feat.pt") == "feat"
    assert strip_video_ext("already_stripped") == "already_stripped"


def test_is_normal_video():
    assert is_normal_video("video_label_A.mp4") is True
    assert is_normal_video("/path/to/movie__label_A.pt") is True
    assert is_normal_video("video_label_B1-0-0.mp4") is False
    assert is_normal_video("Bad.Boys.1995__#01-11-55_label_G-B2-B6.mp4") is False


def test_get_video_classes():
    assert get_video_classes("video_label_A.mp4") == []
    assert get_video_classes("v=BQjKQbYgUBA__#1_label_B1-0-0.mp4") == ["B1"]
    multi = get_video_classes("Bad.Boys.1995__#01-11-55_label_G-B2-B6.mp4")
    assert sorted(multi) == ["B2", "B6", "G"]
    assert get_video_classes("video_without_label.mp4") == []


def test_get_multi_hot_label():
    normal_label = get_multi_hot_label("video_label_A.mp4")
    assert isinstance(normal_label, torch.Tensor)
    assert normal_label.shape == (len(XD_CLASSES),)
    assert torch.all(normal_label == 0)

    anom_label = get_multi_hot_label("Bad.Boys.1995__#01-11-55_label_G-B2-B6.mp4")
    assert anom_label.shape == (len(XD_CLASSES),)
    # XD_CLASSES = ['B1', 'B2', 'B4', 'B5', 'B6', 'G']
    # B2 is index 1, B6 is index 4, G is index 5
    expected = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    assert torch.equal(anom_label, expected)


def test_parse_real_annotations():
    ann_file = "/home/n3uron/workspace/Anomaly_Detection/XDviolanece-test-annotations.txt"
    if os.path.isfile(ann_file):
        annotations = parse_annotations(ann_file)
        assert len(annotations) == 500
        # Check first entry: v=S-7rRLrxnVQ__#1_label_B4-0-0 0 1517 1970 3038
        key = "v=S-7rRLrxnVQ__#1_label_B4-0-0"
        assert key in annotations
        assert annotations[key] == [(0, 1517), (1970, 3038)]


def test_generate_frame_and_segment_gt():
    intervals = [(10, 20), (50, 60)]
    total_frames = 100
    frame_gt = generate_frame_gt(intervals, total_frames)

    assert frame_gt.shape == (100,)
    assert frame_gt.dtype == np.uint8
    assert np.sum(frame_gt) == 20
    assert np.all(frame_gt[10:20] == 1)
    assert np.all(frame_gt[50:60] == 1)
    assert np.all(frame_gt[20:50] == 0)

    segment_gt = generate_segment_gt(intervals, total_frames, num_segments=10, overlap_threshold=0.0)
    assert segment_gt.shape == (10,)
    # Segments are [0,10), [10,20), ..., [50,60), ...
    # segment 1 covers [10, 20), segment 5 covers [50, 60)
    assert segment_gt[1] == 1.0
    assert segment_gt[5] == 1.0
    assert segment_gt[0] == 0.0


def test_expand_segment_scores_to_frames():
    seg_scores = torch.tensor([0.1, 0.9])
    total_frames = 10
    frame_scores = expand_segment_scores_to_frames(seg_scores, total_frames)
    assert frame_scores.shape == (10,)
    assert np.allclose(frame_scores[:5], 0.1)
    assert np.allclose(frame_scores[5:], 0.9)

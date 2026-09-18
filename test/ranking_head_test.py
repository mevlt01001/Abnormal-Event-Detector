import os
import pytest
import torch

from model.ranking_head import SegmentRankingHead
from model.analyzer import VideoAnalyzer


def test_ranking_head_init_and_forward():
    head = SegmentRankingHead(input_dim=512, hidden_dims=[256, 128], dropout_rates=[0.5, 0.4])
    head.eval()

    x = torch.randn(32, 512)
    with torch.no_grad():
        out = head(x)

    assert out.shape == (32, 1), f"Expected (32, 1), got {out.shape}"
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_ranking_head_l2_norm():
    head = SegmentRankingHead(input_dim=512)
    head.eval()

    x = torch.randn(10, 512)
    with torch.no_grad():
        out1 = head(x)
        out2 = head(x * 50.0)

    assert torch.allclose(out1, out2, atol=1e-5), "L2 normalization should ensure scale invariance"


def test_ranking_head_noise_in_training():
    head = SegmentRankingHead(input_dim=512)
    head.train()

    torch.manual_seed(42)
    x = torch.zeros(5, 512)
    out1 = head(x)
    out2 = head(x)

    # In train mode, noise injection makes successive outputs non-identical
    assert not torch.allclose(out1, out2), "Training mode should inject noise augmentation"

    # In eval mode, outputs are strictly deterministic
    head.eval()
    with torch.no_grad():
        eval1 = head(x)
        eval2 = head(x)
    assert torch.allclose(eval1, eval2), "Eval mode must be deterministic"


def test_ranking_head_score_to_segments(tmp_path):
    head = SegmentRankingHead(input_dim=128)
    head.eval()

    feats = torch.randn(32, 128)
    plot_file = str(tmp_path / "test_plot.png")

    segments = head.score_to_segments(
        patch_feats=feats,
        video_seconds=60.0,
        threshold=0.0,  # All segments will exceed threshold
        tolerance_sec=2.0,
        padding_sec=1.0,
        plot_graph=True,
        save_file_name=plot_file,
    )

    assert isinstance(segments, list)
    assert len(segments) > 0
    seg = segments[0]
    for key in ["start_time", "end_time", "duration", "score"]:
        assert key in seg, f"Missing key {key} in segment dict"
    assert seg["duration"] >= 0.0
    assert os.path.exists(plot_file), "Plot file should have been generated"


def test_video_analyzer_init():
    analyzer = VideoAnalyzer(backbone="r3d_18", clip_size=16, overlap=4)
    assert analyzer.feature_dim == 512
    assert analyzer.stride == 12
    assert isinstance(analyzer.ranking_head, SegmentRankingHead)

    # Test forward pass with dummy clip batch: [B, C, T, H, W]
    analyzer.eval()
    dummy_input = torch.randn(2, 3, 16, 224, 224)
    with torch.no_grad():
        scores = analyzer(dummy_input)
    assert scores.shape == (2, 1)


def test_video_analyzer_onnx_export(tmp_path):
    analyzer = VideoAnalyzer(backbone="r3d_18", clip_size=16)
    analyzer.eval()

    onnx_file = str(tmp_path / "analyzer.onnx")
    shape = analyzer.export_onnx(onnx_file, batch_size=1, imgsz=224)
    assert os.path.exists(onnx_file), "ONNX file was not created"
    assert shape == (1, 3, 16, 224, 224)

"""Unit tests for AnomalyDetector and AnomalyHead."""

import os
import tempfile
import numpy as np
import torch
import pytest

from model import AnomalyDetector, AnomalyHead


def test_anomaly_head_defaults():
    head = AnomalyHead(in_features=768)
    assert head.num_classes == 4
    assert head.dropout_rates == (0.6, 0.6)
    assert head.hidden_dims == (512, 256)

    # Forward pass on [B, T, D]
    x = torch.randn(2, 32, 768)
    out = head(x)
    assert out.shape == (2, 32, 4)


def test_anomaly_detector_forward():
    # Head-only detector
    detector = AnomalyDetector(backbone_name=None, num_classes=4, in_features=768)
    assert detector.num_classes == 4

    # 3D feature tensor [B, T, D]
    x_3d = torch.randn(2, 32, 768)
    out_3d = detector(x_3d)
    assert out_3d.shape == (2, 32, 4)

    # 2D feature tensor [B, D]
    x_2d = torch.randn(2, 768)
    out_2d = detector(x_2d)
    assert out_2d.shape == (2, 4)


def test_checkpoint_override():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
        head = AnomalyHead(in_features=768, num_classes=4, dropout_rates=(0.5, 0.3))
        torch.save(
            {
                "model_state_dict": head.state_dict(),
                "head_state_dict": head.state_dict(),
                "config": {"in_features": 768, "num_classes": 4, "dropout_rates": (0.5, 0.3)},
                "dataset_metadata": {"fps": 20.0, "clip_size": 16, "stride": 16, "overlap": 0.5},
            },
            ckpt_path,
        )

        # 1. Standard load
        model, ckpt = AnomalyDetector.load_from_checkpoint(ckpt_path)
        assert model.num_classes == 4
        assert model.video_processor.target_fps == 20.0
        assert model.video_processor.clip_size == 16

        # 2. User override load (user parameters must override checkpoint values)
        model_override, _ = AnomalyDetector.load_from_checkpoint(
            ckpt_path,
            fps=30.0,
            clip_size=32,
            dropout_rates=(0.2, 0.1),
        )
        assert model_override.video_processor.target_fps == 30.0
        assert model_override.video_processor.clip_size == 32
        assert model_override.head.dropout_rates == (0.2, 0.1)

        # 3. Test from_checkpoint alias
        model_alias, _ = AnomalyDetector.from_checkpoint(ckpt_path)
        assert model_alias.num_classes == 4


def test_missing_head_state_dict_raises_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_ckpt = os.path.join(tmpdir, "bad_ckpt.pt")
        torch.save({"model_state_dict": {}}, bad_ckpt)
        with pytest.raises(KeyError, match="head_state_dict"):
            AnomalyHead.load_from_checkpoint(bad_ckpt)


def test_predict_with_feature_dictionary():
    with tempfile.TemporaryDirectory() as tmpdir:
        feat_path = os.path.join(tmpdir, "sample_feat.pt")
        # Save as dictionary format (common in this repository)
        torch.save({"feats": torch.randn(32, 768)}, feat_path)

        detector = AnomalyDetector(backbone_name=None, num_classes=4, in_features=768)
        res = detector.predict(
            input_source=feat_path,
            save_graph=False,
            save_clips=False,
            create_report=False,
            show_progress=False,
        )
        assert "detected_segments" in res
        assert "scores_smooth" in res
        assert res["duration_sec"] > 0
        assert res["report_path"] is None


def test_predict_create_report_flag():
    with tempfile.TemporaryDirectory() as tmpdir:
        feat_path = os.path.join(tmpdir, "test_video_1.pt")
        torch.save({"feats": torch.randn(20, 768)}, feat_path)

        detector = AnomalyDetector(backbone_name=None, num_classes=4, in_features=768)

        # 1. create_report = True (default)
        res_with_report = detector.predict(
            input_source=feat_path,
            save_graph=True,
            create_report=True,
            save_dir=tmpdir,
            show_progress=False,
        )
        assert res_with_report["report_path"] is not None
        assert os.path.exists(res_with_report["report_path"])
        assert res_with_report["graph_path"] is not None
        assert os.path.exists(res_with_report["graph_path"])

        with open(res_with_report["report_path"], "r", encoding="utf-8") as f:
            md_text = f.read()
            assert "test_video_1" in md_text
            assert "anomaly_timeline.png" in md_text

        # 2. create_report = False
        res_no_report = detector.predict(
            input_source=feat_path,
            save_graph=False,
            create_report=False,
            save_dir=tmpdir,
            show_progress=False,
        )
        assert res_no_report["report_path"] is None
        assert res_no_report["graph_path"] is None


def test_predict_5d_clips_tensor():
    # Test Stage 2 prediction with [N, C, T, H, W]
    detector = AnomalyDetector(backbone_name=None, num_classes=4, in_features=768)
    
    # Mock backbone that reduces [B, C, T, H, W] to [B, 768]
    class MockBackbone(torch.nn.Module):
        def forward(self, x):
            B = x.shape[0]
            return torch.randn(B, 768)

    detector.backbone = MockBackbone()

    clips = torch.randn(4, 3, 16, 112, 112)
    res = detector.predict(
        input_source=clips,
        save_graph=False,
        create_report=False,
        show_progress=False,
    )
    assert "detected_segments" in res
    assert res["video_name"] == "Video_Clips_Tensor"
    assert res["duration_sec"] > 0


def test_plot_ticks_and_report_generator():
    from model.analyzer import plot_anomaly_timeline, generate_video_markdown_report

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_path = os.path.join(tmpdir, "timeline.png")
        scores = np.linspace(0.1, 0.8, 100)
        segments = [{"start_time": 2.0, "end_time": 5.0, "score": 0.85, "top_class": "Violence"}]

        saved_path = plot_anomaly_timeline(
            scores_smooth=scores,
            scores_raw=scores,
            segments=segments,
            video_seconds=10.0,
            threshold=0.35,
            save_path=plot_path,
            video_name="TestVideo",
        )
        assert os.path.exists(saved_path)

        md_path = os.path.join(tmpdir, "report.md")
        content = generate_video_markdown_report(
            video_name="TestVideo",
            video_seconds=10.0,
            threshold=0.35,
            segments=segments,
            graph_filename="timeline.png",
            save_path=md_path,
        )
        assert os.path.exists(md_path)
        assert "Violence" in content
        assert "timeline.png" in content


"""Unit tests for VideoAnalyzer inference and timeline plotting."""

import os
import shutil
import tempfile
import pytest
import torch

from core.video_analyzer import VideoAnalyzer
from model.head import AnomalyHead


def test_video_analyzer_analysis_and_plot():
    temp_dir = tempfile.mkdtemp(prefix="analyzer_test_")
    try:
        head = AnomalyHead(in_features=768, num_classes=6)
        analyzer = VideoAnalyzer(model=head, device="cpu", target_fps=20.0)

        # Mock features: [32, 768]
        feats = torch.randn(32, 768)
        result = analyzer.analyze_features(
            features=feats,
            video_duration_sec=60.0,
            threshold=0.35,
            video_name="Test_Analysis",
        )

        assert "time_axis" in result
        assert "class_scores" in result
        assert "overall_scores" in result
        assert "detected_intervals" in result
        assert len(result["time_axis"]) == len(result["overall_scores"])

        # Test plot generation
        plot_file = os.path.join(temp_dir, "test_plot.png")
        analyzer.render_timeline_plot(result, save_path=plot_file)
        assert os.path.isfile(plot_file)
        assert os.path.getsize(plot_file) > 1000

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

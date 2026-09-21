"""Unit tests for ONNX export and FastInferenceEngine."""

import os
import shutil
import tempfile
import pytest
import torch

from deployment.export_onnx import export_head_to_onnx
from deployment.trt_inference import FastInferenceEngine
from model.head import AnomalyHead


def test_onnx_export_and_inference():
    temp_dir = tempfile.mkdtemp(prefix="onnx_test_")
    try:
        head = AnomalyHead(in_features=768, num_classes=6)
        onnx_file = os.path.join(temp_dir, "test_head.onnx")

        # Export and verify with onnx.checker
        exported_path = export_head_to_onnx(head, output_path=onnx_file)
        assert os.path.isfile(exported_path)

        # Fast GPU/CPU engine test
        engine = FastInferenceEngine(head, device="cpu")
        dummy_feats = torch.randn(2, 32, 768)
        probs = engine.predict(dummy_feats)
        assert probs.shape == (2, 32, 6)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

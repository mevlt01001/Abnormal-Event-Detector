"""ONNX Export pipeline for AnomalyHead and AnomalyDetector."""

from __future__ import annotations

import os
from typing import Optional
import torch
import torch.nn as nn
import onnx

from model.head import AnomalyHead
from model.model import AnomalyDetector


def export_head_to_onnx(
    model: nn.Module,
    output_path: str = "deployment/anomaly_head.onnx",
    in_features: Optional[int] = None,
    num_classes: Optional[int] = None,
    opset_version: int = 16,
) -> str:
    """Exports AnomalyHead PyTorch module to ONNX format with dynamic batch and sequence length.

    Args:
        model: AnomalyHead or AnomalyDetector instance (or loaded from checkpoint).
        output_path: Destination .onnx file path.
        in_features: Feature vector size (default: auto-detected from model).
        num_classes: Number of macro anomaly classes (default: auto-detected from model).
        opset_version: ONNX operator set version (default: 16).

    Returns:
        Absolute path to the exported ONNX model file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    head = model.head if hasattr(model, "head") else model
    head.eval()

    feat_dim = in_features or getattr(head, "in_features", 768)

    # Dummy input: [batch_size=1, num_segments=32, feature_dim=feat_dim]
    dummy_input = torch.randn(1, 32, feat_dim, dtype=torch.float32)

    dynamic_axes = {
        "features": {0: "batch_size", 1: "num_segments"},
        "logits": {0: "batch_size", 1: "num_segments"},
    }

    print(f"Exporting AnomalyHead to ONNX: {output_path}...")
    torch.onnx.export(
        head,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )

    # Validate exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"✓ ONNX model successfully verified with onnx.checker!")
    return os.path.abspath(output_path)


if __name__ == "__main__":
    head = AnomalyHead(in_features=768, num_classes=6)
    export_head_to_onnx(head)

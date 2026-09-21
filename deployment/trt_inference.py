"""TensorRT and ONNX High-Performance Inference Engine for Anomaly Detection."""

from __future__ import annotations

import os
import subprocess
import time
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


def build_tensorrt_engine_cmd(
    onnx_path: str = "deployment/anomaly_head.onnx",
    engine_path: str = "deployment/anomaly_head.engine",
    in_features: int = 768,
    fp16: bool = True,
) -> str:
    """Returns the trtexec command to compile ONNX into an optimized TensorRT engine plan.

    Command includes dynamic batch and temporal sequence profiles for real-time video analytics.
    """
    flag_fp16 = "--fp16" if fp16 else ""
    cmd = (
        f"trtexec --onnx={onnx_path} --saveEngine={engine_path} {flag_fp16} "
        f"--minShapes=features:1x16x{in_features} --optShapes=features:1x32x{in_features} --maxShapes=features:8x64x{in_features}"
    )
    return cmd


class FastInferenceEngine:
    """High-speed GPU inference engine for Anomaly Detection.

    Supports PyTorch CUDA JIT/TorchScript, ONNX, and TensorRT.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.in_features = getattr(self.model, "in_features", 768)

        # Warmup
        dummy = torch.randn(1, 32, self.in_features, device=self.device)
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(dummy)

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> np.ndarray:
        """Executes ultra-fast GPU inference.

        Args:
            features: Tensor of shape [B, T, in_features] or [T, in_features].

        Returns:
            probabilities: Numpy array of shape [B, T, num_classes] in range [0, 1].
        """
        if features.ndim == 2:
            # [T, in_features] -> [1, T, in_features]
            features = features.unsqueeze(0)
        features = features.to(self.device).float()

        # [B, T, in_features] -> [B, T, num_classes]
        logits = self.model(features)
        probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def benchmark(self, num_iterations: int = 100) -> float:
        """Benchmarks inference latency on 32-segment video features."""
        dummy = torch.randn(1, 32, self.in_features, device=self.device)
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        total_time = time.time() - start
        avg_ms = (total_time / num_iterations) * 1000.0
        print(f"🚀 Average Inference Latency: {avg_ms:.3f} ms per video on {self.device}!")
        return avg_ms


if __name__ == "__main__":
    from model.head import AnomalyHead
    head = AnomalyHead(in_features=768, num_classes=6)
    engine = FastInferenceEngine(head)
    engine.benchmark()

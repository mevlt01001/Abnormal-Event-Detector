from __future__ import annotations

import gc
import os
from typing import Generator, List, Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from decord import VideoReader, cpu

from model.model import create_base_model, model_feature_dims, get_feature_dim, VideoModel
from model.ranking_head import SegmentRankingHead
from utils.video_utils import get_video_length
from utils.visualization import plot_anomaly_timeline


def unwrap_fc_state_dict(checkpoint: Union[str, os.PathLike, dict]) -> dict:
    """Unwraps state dict from checkpoint dictionary or file path."""
    if isinstance(checkpoint, (str, os.PathLike)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError("FC checkpoint must be a dictionary or path to a .pt file.")
    return checkpoint


class VideoAnalyzer(nn.Module):
    """End-to-end video inference and temporal anomaly localization analyzer."""

    def __init__(
        self,
        backbone: Union[VideoModel, str] = "swin3d_t",
        clip_size: int = 16,
        overlap: int = 0,
        ranking_head: Optional[nn.Module] = None,
        fc_checkpoint: Optional[Union[str, dict]] = None,
    ) -> None:
        super().__init__()
        self.clip_size = clip_size
        self.overlap = overlap
        self.stride = max(1, clip_size - overlap)

        # 1. Setup video backbone
        if isinstance(backbone, str):
            self.backbone_name = backbone.lower()
            self.backbone = create_base_model(self.backbone_name)
            self.feature_dim = model_feature_dims.get(
                self.backbone_name, get_feature_dim(self.backbone)
            )
        elif isinstance(backbone, nn.Module):
            self.backbone = backbone
            self.backbone_name = backbone.__class__.__name__.lower()
            self.feature_dim = get_feature_dim(self.backbone)
        else:
            raise TypeError(f"Unsupported backbone type: {type(backbone)}")

        # 2. Setup ranking head
        if ranking_head is not None:
            self.ranking_head = ranking_head
        else:
            if fc_checkpoint is not None:
                chk = unwrap_fc_state_dict(fc_checkpoint)
                if any(k.startswith("MLP.") for k in chk.keys()):
                    from model.model import FC_head
                    self.ranking_head = FC_head(in_features=self.feature_dim, num_classes=1, use_sigmoid=True)
                else:
                    self.ranking_head = SegmentRankingHead(input_dim=self.feature_dim)
            else:
                from model.model import FC_head
                self.ranking_head = FC_head(in_features=self.feature_dim, num_classes=1, use_sigmoid=True)

        if fc_checkpoint is not None:
            state_dict = unwrap_fc_state_dict(fc_checkpoint)
            self.ranking_head.load_state_dict(state_dict)

    @staticmethod
    @torch.no_grad()
    def clip_generator(
        video_path: str,
        clip_size: int = 16,
        stride: int = 16,
        fps: int = 30,
        width: int = 224,
        height: int = 224,
        max_video_min: int = 45,
    ) -> Tuple[Generator[torch.Tensor, None, None], int]:
        """Decodes sliding clips from video using Decord."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        vr = VideoReader(video_path, ctx=cpu(0), width=width, height=height, num_threads=2)
        total_vr_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        step = avg_fps / fps if (avg_fps > 0 and fps > 0) else 1.0

        frame_indices = torch.arange(0, total_vr_frames, step=step).long()
        total_frames = len(frame_indices)
        duration_minutes = (total_frames / fps / 60) if fps > 0 else 0.0

        if duration_minutes > max_video_min:
            raise ValueError(
                f"Video duration ({duration_minutes:.1f} min) exceeds limit of {max_video_min} min."
            )

        if total_frames < clip_size:
            raise ValueError(
                f"Video too short: {total_frames} sampled frames < clip_size {clip_size}."
            )

        number_of_clips = 1 + (total_frames - clip_size) // stride

        def _generator(video_reader: VideoReader):
            try:
                for end_idx in range(clip_size, total_frames + 1, stride):
                    start_idx = end_idx - clip_size
                    clip_frame_ids = frame_indices[start_idx:end_idx]
                    frames = video_reader.get_batch(clip_frame_ids.tolist()).asnumpy()
                    clip = torch.from_numpy(frames)  # [T, H, W, C]
                    yield clip
            finally:
                del video_reader
                gc.collect()

        return _generator(vr), number_of_clips

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Converts [B, T, H, W, C] uint8 to [B, C, T, H, W] float32 in [0, 1]."""
        if x.dtype == torch.uint8 or x.max() > 1.0:
            x = x.float() / 255.0
        else:
            x = x.float()
        return x.permute(0, 4, 1, 2, 3).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: [B, C, T, H, W] -> [B, 1]."""
        feats = self.backbone(x)
        scores = self.ranking_head(feats)
        return scores

    @torch.no_grad()
    def analyze(
        self,
        video_path: str,
        width: int = 224,
        height: int = 224,
        fps: int = 30,
        batch_size: int = 8,
        threshold: float = 0.3,
        tolerance_sec: float = 3.0,
        padding_sec: float = 2.0,
        save_graph: bool = True,
        save_dir: str = "Video_Analyses",
    ) -> List[Dict[str, Any]]:
        """Analyzes an input video, runs inference on sliding clips, and returns detected anomaly segments."""
        self.eval()
        device = next(self.parameters()).device

        os.makedirs(save_dir, exist_ok=True)
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        video_seconds = get_video_length(video_path)

        clip_gen, total_clips = self.clip_generator(
            video_path=video_path,
            clip_size=self.clip_size,
            stride=self.stride,
            fps=fps,
            width=width,
            height=height,
        )

        mini_batch: List[torch.Tensor] = []
        scores_list: List[torch.Tensor] = []

        for clip in clip_gen:  # [T, H, W, C]
            mini_batch.append(clip)
            if len(mini_batch) == batch_size:
                batch_tensor = torch.stack(mini_batch).to(device)
                batch_tensor = self.preprocess(batch_tensor)  # [B, C, T, H, W]
                batch_scores = self.forward(batch_tensor).detach().cpu().float()
                scores_list.append(batch_scores)
                mini_batch.clear()

        if len(mini_batch) > 0:
            batch_tensor = torch.stack(mini_batch).to(device)
            batch_tensor = self.preprocess(batch_tensor)
            batch_scores = self.forward(batch_tensor).detach().cpu().float()
            scores_list.append(batch_scores)
            mini_batch.clear()

        if not scores_list:
            raise RuntimeError(f"No clips could be generated for video: {video_path}")

        all_scores = torch.cat(scores_list, dim=0).squeeze(-1)  # [Total_Clips]
        scores_org = all_scores.unsqueeze(0).unsqueeze(0)  # [1, 1, Total_Clips]

        # Interpolation and smoothing
        interpolate_size = max(len(all_scores), int(video_seconds * 10), 100)
        scores_linear = F.interpolate(
            scores_org, size=interpolate_size, mode="linear", align_corners=True
        )
        scores_nearest = F.interpolate(
            scores_org, size=interpolate_size, mode="nearest"
        )

        kernel_size = 11
        pad_size = kernel_size // 2
        scores_linear = F.pad(scores_linear, (pad_size, pad_size), mode="reflect")
        scores_linear = F.avg_pool1d(scores_linear, kernel_size=kernel_size, stride=1)
        scores_linear = scores_linear.squeeze(0).squeeze(0)
        scores_nearest = scores_nearest.squeeze(0).squeeze(0)

        dt = video_seconds / interpolate_size if interpolate_size > 0 else 0.0
        anomaly_indices = torch.where(scores_linear >= threshold)[0].tolist()

        final_segments: List[Dict[str, Any]] = []

        if anomaly_indices:
            raw_segments = []
            curr_start = anomaly_indices[0]
            curr_end = anomaly_indices[0]

            for idx in anomaly_indices[1:]:
                time_gap = (idx - curr_end) * dt
                if time_gap <= tolerance_sec:
                    curr_end = idx
                else:
                    raw_segments.append((curr_start, curr_end))
                    curr_start = idx
                    curr_end = idx
            raw_segments.append((curr_start, curr_end))

            padded_segments = []
            for start_idx, end_idx in raw_segments:
                start_time = start_idx * dt
                end_time = end_idx * dt
                p_start = max(0.0, start_time - padding_sec)
                p_end = min(video_seconds, end_time + padding_sec)
                padded_segments.append([p_start, p_end, start_idx, end_idx])

            merged_segments = []
            curr_p_start, curr_p_end, curr_s_idx, curr_e_idx = padded_segments[0]

            for p_start, p_end, s_idx, e_idx in padded_segments[1:]:
                if p_start <= curr_p_end:
                    curr_p_end = max(curr_p_end, p_end)
                    curr_e_idx = max(curr_e_idx, e_idx)
                else:
                    merged_segments.append((curr_p_start, curr_p_end, curr_s_idx, curr_e_idx))
                    curr_p_start, curr_p_end, curr_s_idx, curr_e_idx = p_start, p_end, s_idx, e_idx
            merged_segments.append((curr_p_start, curr_p_end, curr_s_idx, curr_e_idx))

            for p_start, p_end, s_idx, e_idx in merged_segments:
                seg_scores = scores_linear[s_idx : e_idx + 1]
                max_score = seg_scores.max().item() if len(seg_scores) > 0 else 0.0
                final_segments.append({
                    "start_time": round(float(p_start), 2),
                    "end_time": round(float(p_end), 2),
                    "duration": round(float(p_end - p_start), 2),
                    "score": round(float(max_score), 4),
                })

        if save_graph:
            graph_path = os.path.join(save_dir, f"{video_basename}_anomaly_timeline.png")
            plot_anomaly_timeline(
                scores=scores_linear.cpu(),
                scores_raw=scores_nearest.cpu(),
                segments=final_segments,
                video_seconds=video_seconds,
                threshold=threshold,
                save_path=graph_path,
                title=f"Anomaly Timeline - {video_basename}",
            )

        return final_segments

    def export_onnx(
        self,
        file_path: str,
        batch_size: int = 1,
        imgsz: Union[int, Tuple[int, int]] = 224,
    ) -> Tuple[int, ...]:
        """Exports the end-to-end model to ONNX format."""
        import onnx

        self.eval().cpu()
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        H, W = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        shape = (batch_size, 3, self.clip_size, H, W)
        dummy_input = torch.randn(*shape, dtype=torch.float32, device="cpu")

        torch.onnx.export(
            self,
            dummy_input,
            file_path,
            input_names=["video_clip"],
            output_names=["anomaly_score"],
            dynamic_axes={
                "video_clip": {0: "batch_size"},
                "anomaly_score": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        return shape

    def export_trt(
        self,
        model_path: str,
        batch_size: int = 1,
        imgsz: Union[int, Tuple[int, int]] = 224,
    ) -> Optional[str]:
        """Exports to TensorRT plan if NVIDIA CUDA and tensorrt are available."""
        if not torch.cuda.is_available():
            print("[INFO] TensorRT requires NVIDIA CUDA. Skipping TRT export.")
            return None

        try:
            import tensorrt as trt
        except ImportError:
            print("[INFO] tensorrt package not installed. Skipping TRT export.")
            return None

        onnx_file_path = model_path if model_path.endswith(".onnx") else f"{model_path}.onnx"
        plan_file_path = model_path.replace(".onnx", ".plan") if model_path.endswith(".onnx") else f"{model_path}.plan"

        if not os.path.exists(onnx_file_path):
            self.export_onnx(onnx_file_path, batch_size=batch_size, imgsz=imgsz)

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network()
        parser = trt.OnnxParser(network, logger)

        with open(onnx_file_path, "rb") as f:
            if not parser.parse(f.read()):
                for idx in range(parser.num_errors):
                    print(parser.get_error(idx))
                raise RuntimeError(f"Failed to parse ONNX model at {onnx_file_path}")

        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        H, W = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        min_shape = (1, 3, self.clip_size, H, W)
        opt_shape = (batch_size, 3, self.clip_size, H, W)
        max_shape = (max(batch_size, 4), 3, self.clip_size, H, W)
        profile.set_shape("video_clip", min=min_shape, opt=opt_shape, max=max_shape)
        config.add_optimization_profile(profile)

        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError(f"TensorRT plan generation failed for {onnx_file_path}")

        with open(plan_file_path, "wb") as f:
            f.write(engine_bytes)

        return plan_file_path

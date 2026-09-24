"""Unified Anomaly Detector module combining 3D backbone, AnomalyHead, and end-to-end execution.

Single-class interface for:
- Training: model.fit(train_ds, val_ds) or passing to Trainer instances.
- Evaluation: model.evaluate(test_ds) or passing to Evaluator instances.
- TensorRT / ONNX: model.export_trt(...), model.set_inf_mode("trt"), and async CUDA execution.
- Prediction: model.predict(video_or_features) with automated spatio-temporal invariants and tqdm progress.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from model.backbone import create_backbone, get_backbone_dim
from model.head import AnomalyHead
from utils.video_processor import VideoProcessor, save_segment_clips


class AnomalyDetector(nn.Module):
    """End-to-End Class-Aware Video Anomaly Detection PyTorch Module.

    Integrates a 3D spatio-temporal video backbone (e.g. MViT-v2-S) with a pyramidal
    AnomalyHead for segment-level multi-class ranking.

    Args:
        backbone_name: Name of video backbone (default: 'mvit_v2_s').
            Set to None to initialize a lightweight head-only model for precomputed features.
        num_classes: Number of macro anomaly classes (default: 4).
        in_features: Input feature dimension. If None, derived automatically from backbone.
        pretrained_backbone: Whether to load Kinetics-400 pretrained backbone weights.
        hidden_dims: Intermediate layer dimensions for AnomalyHead.
        dropout_rates: Dropout probabilities for AnomalyHead (default: (0.6, 0.6)).
        class_names: Optional human-readable category labels.
    """

    def __init__(
        self,
        backbone_name: Optional[str] = "mvit_v2_s",
        num_classes: int = 4,
        in_features: Optional[int] = None,
        pretrained_backbone: bool = True,
        hidden_dims: Tuple[int, ...] = (512, 256),
        dropout_rates: Tuple[float, ...] = (0.6, 0.6),
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.backbone_name = backbone_name.lower() if backbone_name else None
        self.class_names = list(class_names) if class_names else None

        if self.backbone_name:
            self.backbone = create_backbone(self.backbone_name, pretrained=pretrained_backbone)
            feat_dim = get_backbone_dim(self.backbone_name)
        else:
            self.backbone = None
            feat_dim = in_features if in_features else 768

        self.feature_dim = feat_dim
        self.in_features = feat_dim
        self.hidden_dims = hidden_dims
        self.dropout_rates = dropout_rates
        self.head = AnomalyHead(
            in_features=feat_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout_rates=dropout_rates,
        )


        # Inference engine state
        self._inf_mode: str = "pytorch"
        self.trt_engine: Optional[Any] = None
        self.trt_context: Optional[Any] = None
        self.video_processor: Optional[VideoProcessor] = None

    @property
    def inf_mode(self) -> str:
        """Returns active inference mode ('pytorch' or 'trt')."""
        return self._inf_mode

    def set_inf_mode(self, mode: str) -> AnomalyDetector:
        """Switches inference execution between standard PyTorch and TensorRT engine.

        Args:
            mode: 'pytorch' or 'trt'.
        """
        mode_clean = mode.lower().strip()
        if mode_clean not in ("pytorch", "trt"):
            raise ValueError(f"Invalid inference mode '{mode}'. Choose 'pytorch' or 'trt'.")

        if mode_clean == "trt":
            if self.trt_context is None:
                raise RuntimeError(
                    "TensorRT context not found! Call export_trt() or load_trt() before switching to 'trt' mode."
                )

        self._inf_mode = mode_clean
        return self

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass for pre-extracted segment feature tensors."""
        return self.head(features)

    def forward_video(self, video_clips: torch.Tensor) -> torch.Tensor:
        """Forward pass for spatio-temporal video clips."""
        if self._inf_mode == "trt":
            return self.trt_forward(video_clips)

        if self.backbone is None:
            raise RuntimeError("Backbone is not initialized. Initialize model with backbone_name.")

        feats = self.backbone(video_clips)
        logits = self.head(feats)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Universal forward pass dispatching based on input dimensionality."""
        if self._inf_mode == "trt":
            return self.trt_forward(x)

        if x.ndim == 5:
            return self.forward_video(x)
        elif x.ndim in (2, 3):
            return self.forward_features(x)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}. Expected 2D, 3D, or 5D tensor.")

    # -------------------------------------------------------------------------
    # TensorRT & ONNX Engine Operations (Localized Lazy Import)
    # -------------------------------------------------------------------------

    def export_onnx(
        self,
        file_path: str,
        batch_size: int = 8,
        clip_size: int = 16,
        imgsz: Union[int, Tuple[int, int]] = 224,
        simplify: bool = True,
    ) -> Tuple[int, int, int, int, int]:
        """Exports model to ONNX format with dynamic batch axis and optimizes with onnxsim."""
        import onnx

        self.eval().cpu()
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        B = batch_size
        T = clip_size
        H, W = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        shape = (B, 3, T, H, W)

        dummy_input = torch.rand(*shape, dtype=torch.float32, device="cpu")

        torch.onnx.export(
            self,
            dummy_input,
            file_path,
            input_names=["video_segment"],
            output_names=["logits"],
            dynamic_axes={
                "video_segment": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=19,
            do_constant_folding=True,
        )

        if simplify:
            try:
                import onnxsim
                onnx_model = onnx.load(file_path)
                onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
                onnx_model, check = onnxsim.simplify(onnx_model, test_input_shapes={"video_segment": list(shape)})
                if check:
                    onnx.save(onnx_model, file_path)
            except Exception as exc:
                print(f"[Warn] ONNX simplification skipped or encountered issue: {exc}")

        return shape

    def export_trt(
        self,
        model_path: str,
        batch_size: int = 8,
        clip_size: int = 16,
        imgsz: Union[int, Tuple[int, int]] = 224,
    ) -> None:
        """Compiles model to TensorRT .plan engine with dynamic shape profile and loads context."""
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT requires an active NVIDIA CUDA environment.")

        # Localized lazy import of tensorrt strictly within this method
        import tensorrt as trt

        self.eval().cpu()
        trt_logger = trt.Logger(trt.Logger.WARNING)

        base_path = os.path.splitext(model_path)[0]
        os.makedirs(os.path.dirname(os.path.abspath(base_path)), exist_ok=True)
        onnx_file_path = base_path + ".onnx"
        trt_file_path = base_path + ".plan"

        if os.path.exists(trt_file_path):
            self.load_trt(trt_file_path)
            return

        shape = self.export_onnx(onnx_file_path, batch_size=batch_size, clip_size=clip_size, imgsz=imgsz)
        _, C, T, H, W = shape

        min_shape = (1, C, T, H, W)
        opt_shape = (max(1, batch_size // 2), C, T, H, W)
        max_shape = (batch_size, C, T, H, W)

        builder = trt.Builder(trt_logger)
        network = builder.create_network()
        parser = trt.OnnxParser(network, trt_logger)
        config = builder.create_builder_config()

        with open(onnx_file_path, "rb") as f:
            if not parser.parse(f.read()):
                err_msg = "\n".join([str(parser.get_error(i)) for i in range(parser.num_errors)])
                raise RuntimeError(f"ONNX Parsing Error in {onnx_file_path}:\n{err_msg}")

        profile = builder.create_optimization_profile()
        profile.set_shape("video_segment", min=min_shape, opt=opt_shape, max=max_shape)
        config.add_optimization_profile(profile)

        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError(f"TensorRT plan generation failed for {onnx_file_path}")

        with open(trt_file_path, "wb") as f:
            f.write(engine_bytes)

        self.load_trt(trt_file_path)

    def load_trt(self, plan_file_path: str) -> None:
        """Loads serialized TensorRT .plan engine into memory and initializes CUDA execution context."""
        import tensorrt as trt

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(plan_file_path, "rb") as f:
            self.trt_engine = runtime.deserialize_cuda_engine(f.read())

        self.trt_context = self.trt_engine.create_execution_context()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def trt_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes asynchronous TensorRT CUDA inference."""
        if self.trt_context is None:
            raise RuntimeError("TensorRT context not found! Load engine using load_trt().")

        if x.device.type != "cuda":
            x = x.to("cuda", non_blocking=True)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = x.contiguous()

        B, C, T, H, W = x.shape
        stream = torch.cuda.current_stream()
        output = torch.empty((B, self.num_classes), dtype=torch.float32, device="cuda")

        if hasattr(self.trt_context, "set_tensor_address"):
            self.trt_context.set_input_shape("video_segment", (B, C, T, H, W))
            self.trt_context.set_tensor_address("video_segment", x.data_ptr())
            self.trt_context.set_tensor_address("logits", output.data_ptr())
            self.trt_context.execute_async_v3(stream_handle=stream.cuda_stream)
        elif hasattr(self.trt_context, "execute_async_v2"):
            bindings = [int(x.data_ptr()), int(output.data_ptr())]
            self.trt_context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)

        return output

    @staticmethod
    def _load_feature_tensor(source: Union[str, os.PathLike, torch.Tensor, np.ndarray, Dict[str, Any]]) -> torch.Tensor:
        """Safely extracts a [num_segments, feature_dim] tensor from path, tensor, or dictionary."""
        if isinstance(source, (str, os.PathLike)):
            try:
                data = torch.load(source, weights_only=True, map_location="cpu")
            except Exception:
                data = torch.load(source, weights_only=False, map_location="cpu")
        else:
            data = source

        if isinstance(data, dict):
            if "feats" in data:
                feats = data["feats"]
            elif "features" in data:
                feats = data["features"]
            else:
                first = next((v for v in data.values() if isinstance(v, torch.Tensor)), None)
                if first is not None:
                    feats = first
                else:
                    raise ValueError(f"No tensor found in dictionary loaded from {source}")
        elif isinstance(data, np.ndarray):
            feats = torch.from_numpy(data)
        elif isinstance(data, torch.Tensor):
            feats = data
        else:
            raise TypeError(f"Unexpected feature data type: {type(data)}")

        return feats.float()

    # -------------------------------------------------------------------------
    # End-to-End Prediction with tqdm and Sealed Spatio-Temporal Parameters
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        input_source: Union[str, os.PathLike, torch.Tensor, np.ndarray],
        threshold: float = 0.35,
        tolerance_sec: float = 3.0,
        padding_sec: float = 2.0,
        batch_size: int = 8,
        save_graph: bool = True,
        save_clips: bool = False,
        create_report: bool = True,
        save_dir: str = "results/predictions",
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """Runs hierarchical video, clip-tensor, or feature-tensor anomaly inference.

        Supports 3 distinct entry stages:
        - Stage 1 (Video File): Decodes video -> sliding clips -> backbone features -> head logits.
        - Stage 2 (Clips Tensor 5D): Directly passes [N, C, T, H, W] to backbone -> head logits.
        - Stage 3 (Features Tensor/File 2D/3D): Directly passes [N, D] to head -> logits.
        """
        from model.analyzer import (
            extract_anomaly_segments,
            generate_video_markdown_report,
            plot_anomaly_timeline,
        )

        self.eval()
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        is_path = isinstance(input_source, (str, os.PathLike))
        is_video_file = is_path and not str(input_source).endswith(".pt")
        is_5d_tensor = isinstance(input_source, (torch.Tensor, np.ndarray)) and input_source.ndim == 5

        # ---------------------------------------------------------------------
        # Stage 1: Raw Video File Path -> Decode -> Clips -> Backbone -> Head
        # ---------------------------------------------------------------------
        if is_video_file:
            video_path = str(input_source)
            if not os.path.isfile(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            video_name = os.path.splitext(os.path.basename(video_path))[0]
            if self.video_processor is None:
                self.video_processor = VideoProcessor(target_fps=20.0, clip_size=16)

            clip_gen, total_clips, duration_sec = self.video_processor.extract_sliding_clips(
                video_path=video_path, batch_size=batch_size
            )

            all_batch_scores = []
            pbar = tqdm(total=total_clips, desc=f"Predicting {video_name}", disable=not show_progress)

            for batch_clips in clip_gen:
                batch_tensor = batch_clips.to(device)
                if self._inf_mode == "trt":
                    logits = self.trt_forward(batch_tensor)
                else:
                    if self.backbone is None:
                        raise RuntimeError("Backbone is required to process raw video clips. Initialize model with backbone_name.")
                    feats = self.backbone(batch_tensor)
                    logits = self.head(feats)

                probs = torch.sigmoid(logits).detach().cpu()
                all_batch_scores.append(probs)
                pbar.update(len(batch_clips))

            pbar.close()
            if not all_batch_scores:
                raise RuntimeError(f"No clips could be processed for {video_path}")
            scores_tensor = torch.cat(all_batch_scores, dim=0)

        # ---------------------------------------------------------------------
        # Stage 2: 5D Video Clips Tensor [B, C, T, H, W] -> Backbone -> Head
        # ---------------------------------------------------------------------
        elif is_5d_tensor:
            video_name = "Video_Clips_Tensor"
            clips_t = torch.from_numpy(input_source).float() if isinstance(input_source, np.ndarray) else input_source.float()
            total_clips = len(clips_t)
            fps = getattr(self.video_processor, "target_fps", 20.0) if self.video_processor else 20.0
            clip_sz = getattr(self.video_processor, "clip_size", 16) if self.video_processor else 16
            duration_sec = max(5.0, float(total_clips * (clip_sz / fps)))

            all_batch_scores = []
            pbar = tqdm(total=total_clips, desc="Predicting 5D Clips", disable=not show_progress)

            for i in range(0, total_clips, batch_size):
                batch_tensor = clips_t[i : i + batch_size].to(device)
                if self._inf_mode == "trt":
                    logits = self.trt_forward(batch_tensor)
                else:
                    if self.backbone is None:
                        raise RuntimeError("Backbone is required to process video clips tensor. Initialize model with backbone_name.")
                    feats = self.backbone(batch_tensor)
                    logits = self.head(feats)

                probs = torch.sigmoid(logits).detach().cpu()
                all_batch_scores.append(probs)
                pbar.update(len(batch_tensor))

            pbar.close()
            scores_tensor = torch.cat(all_batch_scores, dim=0)

        # ---------------------------------------------------------------------
        # Stage 3: Feature Tensor or .pt Feature File [N, D] -> Head
        # ---------------------------------------------------------------------
        else:
            video_name = "Feature_Prediction"
            if is_path:
                feat_path = str(input_source)
                video_name = os.path.splitext(os.path.basename(feat_path))[0]

            features = self._load_feature_tensor(input_source)
            if features.ndim == 2:
                features = features.unsqueeze(0)
            features = features.to(device)

            logits = self.forward_features(features)
            scores_tensor = torch.sigmoid(logits).squeeze(0).detach().cpu()
            duration_sec = float(scores_tensor.shape[0] * 1.5)

        # ---------------------------------------------------------------------
        # Post-Processing: Segment Extraction & Event Boundary Clustering
        # ---------------------------------------------------------------------
        segments, scores_smooth, scores_raw = extract_anomaly_segments(
            scores=scores_tensor,
            video_seconds=duration_sec,
            threshold=threshold,
            tolerance_sec=tolerance_sec,
            padding_sec=padding_sec,
            class_names=self.class_names,
        )

        out_root = os.path.join(save_dir, video_name)
        os.makedirs(out_root, exist_ok=True)

        # Optional: Save detected video clips (.mp4)
        if save_clips and is_video_file:
            clip_paths = save_segment_clips(
                video_path=video_path,
                segments=segments,
                save_dir=out_root,
                prefix="segment",
            )
            for seg, c_path in zip(segments, clip_paths):
                seg["clip_path"] = c_path

        # Optional: Save clean timeline chart (25 X-ticks, 10 Y-ticks, zero text clutter)
        graph_path = None
        if save_graph:
            plot_file = os.path.join(out_root, "anomaly_timeline.png")
            graph_path = plot_anomaly_timeline(
                scores_smooth=scores_smooth,
                scores_raw=scores_raw,
                segments=segments,
                video_seconds=duration_sec,
                threshold=threshold,
                save_path=plot_file,
                video_name=video_name,
                class_names=self.class_names,
                scores_multiclass=scores_tensor if (scores_tensor.ndim == 2 and scores_tensor.shape[-1] > 1) else None,
            )

        # Optional: Save executive Markdown report
        report_path = None
        if create_report:
            md_file = os.path.join(out_root, f"{video_name}_report.md")
            generate_video_markdown_report(
                video_name=video_name,
                video_seconds=duration_sec,
                threshold=threshold,
                segments=segments,
                graph_filename="anomaly_timeline.png" if save_graph else None,
                save_path=md_file,
            )
            report_path = os.path.abspath(md_file)

        if show_progress:
            status_tag = "🚨 ANOMALY" if len(segments) > 0 else "✅ CLEAN"
            print(f"[{video_name}] {status_tag} | Detected Events: {len(segments)} | Duration: {duration_sec:.1f}s")

        return {
            "video_name": video_name,
            "duration_sec": duration_sec,
            "threshold": threshold,
            "detected_segments": segments,
            "num_detected_events": len(segments),
            "is_anomaly": len(segments) > 0,
            "scores_smooth": scores_smooth.numpy(),
            "graph_path": graph_path,
            "report_path": report_path,
        }

    # -------------------------------------------------------------------------
    # Convenience Training and Evaluation Wrappers
    # -------------------------------------------------------------------------

    def fit(
        self,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        mode: str = "multiclass",
        tensorboard_dir: Optional[str] = "runs",
        **trainer_kwargs: Any,
    ) -> Dict[str, Any]:
        """Convenience wrapper initiating full model training via BinaryTrainer or MultiClassTrainer."""
        if mode.lower() == "binary":
            from train.trainer import BinaryTrainer
            trainer = BinaryTrainer(model=self, tensorboard_dir=tensorboard_dir, **trainer_kwargs)
        else:
            from train.trainer import MultiClassTrainer
            trainer = MultiClassTrainer(model=self, tensorboard_dir=tensorboard_dir, **trainer_kwargs)

        return trainer.fit(train_dataset, val_dataset)

    def evaluate(
        self,
        dataset_or_loader: Any,
        mode: str = "auto",
        iou_threshold: float = 0.30,
        sweep_thresholds: bool = True,
        **eval_kwargs: Any,
    ) -> Dict[str, Any]:
        """Convenience wrapper running Dual-Engine evaluation (Academic + Threshold Sweep)."""
        from evaluation.evaluator import Evaluator

        evaluator = Evaluator(model=self, dataset_or_loader=dataset_or_loader)
        return evaluator.evaluate(
            mode=mode,
            iou_threshold=iou_threshold,
            sweep_thresholds=sweep_thresholds,
            **eval_kwargs,
        )

    # -------------------------------------------------------------------------
    # Checkpoint Loading with Automatic Provenance Restoration
    # -------------------------------------------------------------------------

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        backbone_name: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        num_classes: Optional[int] = None,
        dropout_rates: Optional[Sequence[float]] = None,
        hidden_dims: Optional[Sequence[int]] = None,
        in_features: Optional[int] = None,
        class_names: Optional[Sequence[str]] = None,
        fps: Optional[float] = None,
        clip_size: Optional[int] = None,
        stride: Optional[int] = None,
        overlap: Optional[float] = None,
    ) -> Tuple[AnomalyDetector, Dict[str, Any]]:
        """Instantiates AnomalyDetector from a trained checkpoint and restores spatio-temporal invariants.

        User-provided arguments explicitly override any configuration stored in the checkpoint.
        """
        dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        head, ckpt = AnomalyHead.load_from_checkpoint(
            checkpoint_path,
            device=dev,
            num_classes=num_classes,
            dropout_rates=dropout_rates,
            hidden_dims=hidden_dims,
            in_features=in_features,
        )

        ds_meta = ckpt.get("dataset_metadata") or {}
        ckpt_classes = ds_meta.get("dataset_classes") or ds_meta.get("classes") or getattr(head, "classes", None)
        classes = list(class_names) if class_names is not None else ckpt_classes

        # Resolve backbone_name from argument or checkpoint config
        resolved_backbone = backbone_name if backbone_name is not None else ckpt.get("config", {}).get("backbone_name")

        detector = cls(
            backbone_name=resolved_backbone,
            num_classes=head.num_classes,
            in_features=head.in_features,
            pretrained_backbone=False,
            hidden_dims=getattr(head, "hidden_dims", (512, 256)),
            dropout_rates=getattr(head, "dropout_rates", (0.6, 0.6)),
            class_names=classes,
        ).to(dev)
        detector.head = head

        # If checkpoint contains full model_state_dict and detector has backbone, restore backbone weights
        if detector.backbone is not None and "model_state_dict" in ckpt:
            detector.load_state_dict(ckpt["model_state_dict"], strict=False)

        # Restore VideoProcessor using spatio-temporal invariants with user override precedence
        final_fps = float(fps) if fps is not None else float(ds_meta.get("fps", 20.0))
        final_clip_size = int(clip_size) if clip_size is not None else int(ds_meta.get("clip_size", 16))
        final_stride = int(stride) if stride is not None else int(ds_meta.get("stride", 16))
        final_overlap = overlap if overlap is not None else ds_meta.get("overlap")

        detector.video_processor = VideoProcessor(
            target_fps=final_fps,
            clip_size=final_clip_size,
            stride=final_stride,
            overlap=final_overlap,
        )

        return detector, ckpt

    @classmethod
    def from_checkpoint(cls, *args: Any, **kwargs: Any) -> Tuple[AnomalyDetector, Dict[str, Any]]:
        """Convenience classmethod alias for load_from_checkpoint."""
        return cls.load_from_checkpoint(*args, **kwargs)


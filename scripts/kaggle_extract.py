"""Multi-Model Feature Extraction Engine for Kaggle and Local Pipelines.

Supports simultaneous extraction for multiple 3D backbones:
- swin3d_t (768-d)
- mvit_v2_s (768-d)
- r3d_18 (512-d)
- mc3_18 (512-d)
- r2plus1d_18 (512-d)
- s3d (1024-d)

Decodes each video once with Decord and passes segment tensors through all requested
models sequentially on GPU, achieving high-throughput extraction with minimal overhead.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from tqdm import tqdm

# Ensure workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.video_processor import VideoProcessor
from model.backbone import (
    BACKBONE_CREATORS,
    BACKBONE_FEATURE_DIMS,
    create_backbone,
    get_backbone_dim,
)

ALL_SUPPORTED_MODELS = ["swin3d_t", "mvit_v2_s", "r3d_18", "mc3_18", "r2plus1d_18", "s3d"]


def strip_video_ext(filename: str) -> str:
    """Strips common video extensions."""
    for ext in (".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv"):
        if filename.lower().endswith(ext):
            return filename[: -len(ext)]
    return os.path.splitext(filename)[0]


def normalize_class_map_input(
    class_video_map: Union[List[List[str]], Dict[str, List[str]], str],
    class_names: Optional[List[str]] = None,
) -> Tuple[List[List[str]], List[str]]:
    """Normalizes class video mappings into uniform list-of-lists and class names."""
    if isinstance(class_video_map, str):
        if os.path.isfile(class_video_map):
            with open(class_video_map, "r", encoding="utf-8") as f:
                class_video_map = json.load(f)
        else:
            class_video_map = json.loads(class_video_map)

    if isinstance(class_video_map, dict):
        c_names = list(class_video_map.keys())
        v_lists = [class_video_map[k] for k in c_names]
        return v_lists, c_names

    if isinstance(class_video_map, list):
        if class_names is None:
            c_names = [f"class_{i}" for i in range(len(class_video_map))]
        else:
            c_names = list(class_names)
        if len(class_video_map) != len(c_names):
            raise ValueError(
                f"Length mismatch: class_video_map has {len(class_video_map)} lists, "
                f"but class_names has {len(c_names)} elements."
            )
        return class_video_map, c_names

    raise TypeError(f"Unsupported type for class_video_map: {type(class_video_map)}")


def extract_features_from_class_map(
    class_video_map: Union[List[List[str]], Dict[str, List[str]], str],
    class_names: Optional[List[str]] = None,
    output_dir: str = "features",
    model_names: Union[str, Sequence[str]] = "swin3d_t",
    num_segments: int = 32,
    clip_size: int = 16,
    overlap_ratio: float = 0.0,
    stride: Optional[int] = None,
    fps: float = 30.0,
    batch_size: int = 8,
    device: Optional[str] = None,
    overwrite: bool = False,
    show_progress: bool = True,
    worker_id: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Extracts segment features for multiple 3D backbones efficiently.

    Args:
        class_video_map: List of lists [[video1, ...], [video2, ...]] or dict {"ClassA": [...]}.
        class_names: Names for class lists if class_video_map is list-of-lists.
        output_dir: Root output directory (e.g. 'features').
        model_names: Single model string or list of models (e.g. ['swin3d_t', 'mvit_v2_s', ...]).
        num_segments: Number of temporal segments per video (default: 32).
        clip_size: Frames per spatio-temporal clip (default: 16).
        overlap_ratio: Overlap ratio between adjacent sliding clips [0.0, 0.95) (default: 0.0).
        fps: Target sampling FPS (default: 30.0).
        batch_size: Sub-batch size of clips passed to model forward pass (default: 8).
        device: 'cuda' or 'cpu' (default: auto-detect).
        overwrite: If True, re-extracts even if .pt file exists.
        show_progress: If True, shows progress bar.
        worker_id: GPU worker identifier for multi-GPU execution.
        num_workers: Total parallel GPU workers.

    Returns:
        Summary dict containing counts, manifests, elapsed time, and status.
    """
    start_time = time.time()
    v_lists, c_names = normalize_class_map_input(class_video_map, class_names)

    tag = f"[GPU {worker_id}]" if worker_id is not None else "[INFO]"

    if isinstance(model_names, str):
        target_models = [model_names.lower()]
    else:
        target_models = [m.lower() for m in model_names]

    for m in target_models:
        if m not in BACKBONE_CREATORS:
            raise ValueError(f"Unsupported model: '{m}'. Supported: {list(BACKBONE_CREATORS.keys())}")

    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    if device_obj.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if worker_id is None:
        print("\n" + "=" * 70)
        print("[PIPELINE] Multi-Model Video Feature Extraction Engine")
        print("=" * 70)
        print(f"[INFO] Target Models ({len(target_models)}): {', '.join(target_models)}")
        print(f"[INFO] Classes ({len(c_names)}):       {', '.join(c_names[:6])}{'...' if len(c_names) > 6 else ''}")
        print(f"[INFO] Configuration:       {num_segments} segments, {clip_size} frames/clip @ {fps:.1f} FPS, overlap={overlap_ratio:.2f}")
        print(f"[INFO] Output Directory:    {output_dir}")
        print(f"[INFO] Compute Device:      {device_obj}")
        print("=" * 70 + "\n")
    else:
        print(f"{tag} Engine initialized on {device_obj} for {sum(len(v) for v in v_lists)} videos | Models: {', '.join(target_models)}")

    # 1. Map each unique video path to list of (class_name, target_pt_path) for each model
    video_targets: Dict[str, Dict[str, List[Tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    total_assignments = 0

    for c_idx, (c_name, v_paths) in enumerate(zip(c_names, v_lists)):
        for v_path in v_paths:
            v_stem = strip_video_ext(os.path.basename(v_path))
            for m_name in target_models:
                m_dir = os.path.join(output_dir, m_name, c_name)
                os.makedirs(m_dir, exist_ok=True)
                target_pt = os.path.join(m_dir, f"{v_stem}.pt")
                video_targets[v_path][m_name].append((c_name, target_pt))
                total_assignments += 1

    unique_videos = list(video_targets.keys())
    if worker_id is None:
        print(f"[INFO] Total Unique Videos: {len(unique_videos)}")
        print(f"[INFO] Total Target Outputs: {total_assignments} ({len(unique_videos)} videos x {len(target_models)} models)\n")

    # Filter videos that actually need extraction
    videos_to_process: List[str] = []
    for v_path in unique_videos:
        needs_work = False
        for m_name in target_models:
            for _, pt_path in video_targets[v_path][m_name]:
                if overwrite or not os.path.isfile(pt_path):
                    needs_work = True
                    break
            if needs_work:
                break
        if needs_work:
            videos_to_process.append(v_path)

    already_done = len(unique_videos) - len(videos_to_process)
    if worker_id is None:
        print(f"[INFO] Existing Videos (Skipping): {already_done}")
        print(f"[INFO] Videos Requiring Extraction: {len(videos_to_process)}\n")
    else:
        print(f"{tag} Videos to extract: {len(videos_to_process)} (Skipping already done: {already_done})")

    if not videos_to_process:
        print(f"{tag} All target features are already extracted and up to date.")
    else:
        # 2. Load all target models onto device
        if worker_id is None:
            print("[INFO] Loading 3D backbone models into memory...")
        loaded_models: Dict[str, nn.Module] = {}
        for m_name in target_models:
            t_load = time.time()
            loaded_models[m_name] = create_backbone(m_name, pretrained=True).to(device_obj).eval()
            print(f"{tag} Loaded '{m_name}' ({get_backbone_dim(m_name)}-d) in {time.time() - t_load:.1f}s")

        if device_obj.type == "cuda":
            allocated_mb = torch.cuda.memory_allocated(device_obj) / (1024**2)
            print(f"{tag} GPU VRAM allocated: {allocated_mb:.1f} MB\n")

        # 3. Process videos
        processor = VideoProcessor(
            target_fps=fps,
            clip_size=clip_size,
            overlap_ratio=overlap_ratio,
            stride=stride,
        )
        success_count = 0
        failed_videos: List[Dict[str, str]] = []

        desc = f"GPU {worker_id}" if worker_id is not None else "Extracting Features"
        pos = worker_id if worker_id is not None else 0
        iterator = tqdm(
            videos_to_process,
            desc=desc,
            position=pos,
            leave=True,
            disable=not show_progress,
            mininterval=1.0,
            dynamic_ncols=True,
        )
        for idx, v_path in enumerate(iterator):
            v_stem = strip_video_ext(os.path.basename(v_path))
            iterator.set_postfix({"vid": v_stem[:14]})

            try:
                # Check which models still need this video
                models_needed = [
                    m_name
                    for m_name in target_models
                    if any(overwrite or not os.path.isfile(pt) for _, pt in video_targets[v_path][m_name])
                ]
                if not models_needed:
                    continue

                # Uniformly segment video into 32 segments with zero black frames
                segment_gen = processor.extract_uniform_segments(
                    v_path,
                    num_segments=num_segments,
                    overlap_ratio=overlap_ratio,
                    stride=stride,
                )

                # Segment features accumulator for each needed model
                model_seg_feats: Dict[str, List[torch.Tensor]] = {m: [] for m in models_needed}

                with torch.inference_mode():
                    for seg_tensor in segment_gen:
                        # seg_tensor: [K, C, clip_size, H, W]
                        K = seg_tensor.shape[0]
                        seg_tensor = seg_tensor.to(device_obj)

                        for m_name in models_needed:
                            model = loaded_models[m_name]

                            if K <= batch_size:
                                out = model(seg_tensor)  # [K, D]
                            else:
                                chunks = torch.split(seg_tensor, batch_size, dim=0)
                                out = torch.cat([model(c) for c in chunks], dim=0)  # [K, D]

                            # Average across clips in this segment -> [1, D]
                            seg_feat = out.mean(dim=0, keepdim=True).cpu()
                            model_seg_feats[m_name].append(seg_feat)

                # Save features for each model
                for m_name in models_needed:
                    if len(model_seg_feats[m_name]) == num_segments:
                        video_feats = torch.cat(model_seg_feats[m_name], dim=0).float()  # [num_segments, D]
                        for c_name, target_pt in video_targets[v_path][m_name]:
                            payload = {
                                "feats": video_feats,
                                "video_name": v_stem,
                                "video_path": v_path,
                                "class_name": c_name,
                                "model": m_name,
                                "num_segments": num_segments,
                                "clip_size": clip_size,
                                "overlap_ratio": float(overlap_ratio),
                                "fps": int(fps),
                                "feature_dim": video_feats.shape[-1],
                            }
                            torch.save(payload, target_pt)

                success_count += 1

            except Exception as e:
                print(f"\n{tag} [ERROR] Extraction failed for video {v_stem}: {e}")
                failed_videos.append({"video": v_path, "error": str(e)})

            # Periodic memory cleanup
            if (idx + 1) % 20 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Clean up models
        del loaded_models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. Generate manifest.json for each model
    manifests: Dict[str, Dict[str, Any]] = {}
    for m_name in target_models:
        m_base = os.path.join(output_dir, m_name)
        counts = {}
        for c_name in c_names:
            c_dir = os.path.join(m_base, c_name)
            if os.path.isdir(c_dir):
                counts[c_name] = len([f for f in os.listdir(c_dir) if f.endswith(".pt")])
            else:
                counts[c_name] = 0

        manifest = {
            "model": m_name,
            "num_segments": num_segments,
            "clip_size": clip_size,
            "overlap_ratio": float(overlap_ratio),
            "fps": int(fps),
            "feature_dim": get_backbone_dim(m_name),
            "classes": c_names,
            "counts": counts,
            "total_files": sum(counts.values()),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        manifest_file = os.path.join(m_base, "manifest.json")
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        manifests[m_name] = manifest

    total_time = time.time() - start_time

    if worker_id is None:
        print("\n" + "=" * 70)
        print("[COMPLETED] Multi-Model Feature Extraction Complete")
        print("=" * 70)
        print(f"[INFO] Elapsed Time: {total_time:.1f}s")
        for m_name in target_models:
            m_info = manifests[m_name]
            print(f"[INFO] Model: {m_name:<12} | Dim: {m_info['feature_dim']} | Files Saved: {m_info['total_files']}")
        print("=" * 70 + "\n")
    else:
        print(f"\n{tag} [COMPLETED] Finished extracting {success_count}/{len(videos_to_process)} videos in {total_time:.1f}s")

    return {
        "status": "completed",
        "output_dir": output_dir,
        "manifests": manifests,
        "elapsed_time": total_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Feature Extraction CLI")
    parser.add_argument("--class-map", type=str, required=True, help="Path to JSON file with class_video_map")
    parser.add_argument("--output-dir", type=str, default="features", help="Output directory")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["swin3d_t"],
        help="List of model names to extract (e.g. swin3d_t mvit_v2_s r3d_18 mc3_18 r2plus1d_18 s3d)",
    )
    parser.add_argument("--num-segments", type=int, default=32, help="Number of segments (default: 32)")
    parser.add_argument("--clip-size", type=int, default=16, help="Frames per clip (default: 16)")
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Overlap ratio between adjacent clips in [0.0, 0.95) (default: 0.0)",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Sampling FPS (default: 30.0)")
    parser.add_argument("--batch-size", type=int, default=8, help="Clip batch size for GPU (default: 8)")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    args = parser.parse_args()

    extract_features_from_class_map(
        class_video_map=args.class_map,
        output_dir=args.output_dir,
        model_names=args.models,
        num_segments=args.num_segments,
        clip_size=args.clip_size,
        overlap_ratio=args.overlap,
        fps=args.fps,
        batch_size=args.batch_size,
        device=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

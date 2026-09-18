#!/usr/bin/env python3
"""Kaggle Feature Extraction Script for Video Anomaly Detection.

Extracts multi-segment spatio-temporal feature representations from arbitrary video lists
and organizes them into class directories for downstream model training.

Designed for Kaggle execution:
- Input format: list of lists [[p1, p2], [p3, p4], ...] or dict {"Fighting": [p1, p2], ...}
- Backbone: Swin3D-T (default, best performing) or any supported 3D backbone
- Output structure: features/{model_name}/{class_name}/{video_stem}.pt
- Multi-label deduplication: videos belonging to multiple classes are only inferred once on GPU
- Manifest: writes lightweight manifest.json with dataset summary and counts
- Dual API: can be executed via CLI or imported directly into Kaggle Python scripts / Jupyter notebooks
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Add repository root to python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from tqdm import tqdm

from model.model import Model
from utils.annotation_parser import strip_video_ext


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract video features on Kaggle for Anomaly Detection"
    )
    parser.add_argument(
        "--class_map",
        type=str,
        required=True,
        help="Path to JSON file or inline JSON string with [[paths], [paths], ...] or {'class': [paths]}",
    )
    parser.add_argument(
        "--class_names",
        nargs="+",
        type=str,
        default=None,
        help="List of class names matching the inner lists of class_map (e.g. Fighting Normal Explosion)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="features",
        help="Root directory where features/{model}/{class} will be saved (default: features)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="swin3d_t",
        help="Video backbone architecture (default: swin3d_t)",
    )
    parser.add_argument(
        "--num_segments",
        type=int,
        default=32,
        help="Number of temporal segments per video (default: 32)",
    )
    parser.add_argument(
        "--clip_size",
        type=int,
        default=16,
        help="Frames per clip (default: 16)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target sampling FPS (default: 30)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for 3D model inference on GPU (default: 4)",
    )
    parser.add_argument(
        "--max_clips_per_segment",
        type=int,
        default=2,
        help="Maximum clips per segment to sample (default: 2)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of background DataLoader workers for parallel video decoding (0 for sequential)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on ('cuda' or 'cpu', default: auto-detect)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted feature files",
    )
    return parser.parse_args()


def normalize_class_map_input(
    class_video_map: Union[List[List[str]], Dict[str, List[str]], str],
    class_names: Optional[List[str]] = None,
) -> Tuple[List[List[str]], List[str]]:
    """Normalizes class_video_map and class_names into standardized lists.

    Returns:
        (normalized_video_lists, normalized_class_names)
    """
    if isinstance(class_video_map, str):
        if os.path.isfile(class_video_map):
            with open(class_video_map, "r", encoding="utf-8") as f:
                parsed = json.load(f)
        else:
            parsed = json.loads(class_video_map)
        class_video_map = parsed

    if isinstance(class_video_map, dict):
        keys = list(class_video_map.keys())
        if class_names is not None:
            # Reorder or filter based on user-provided class_names
            ordered_names = [c for c in class_names if c in class_video_map]
            if len(ordered_names) != len(class_names):
                missing = set(class_names) - set(ordered_names)
                raise ValueError(f"Class names not found in dict keys: {missing}")
            final_names = ordered_names
        else:
            final_names = keys
        final_lists = [class_video_map[k] for k in final_names]
        return final_lists, final_names

    if isinstance(class_video_map, (list, tuple)):
        num_classes = len(class_video_map)
        if class_names is None:
            final_names = [f"class_{i}" for i in range(num_classes)]
        else:
            if len(class_names) != num_classes:
                raise ValueError(
                    f"Mismatch between number of class lists ({num_classes}) "
                    f"and provided class names ({len(class_names)}: {class_names})"
                )
            final_names = list(class_names)
        final_lists = [list(lst) for lst in class_video_map]
        return final_lists, final_names

    raise TypeError(
        f"Unsupported type for class_video_map: {type(class_video_map)}. "
        f"Expected list of lists, dict, or JSON string/filepath."
    )


def extract_features_from_class_map(
    class_video_map: Union[List[List[str]], Dict[str, List[str]], str],
    class_names: Optional[List[str]] = None,
    output_dir: str = "features",
    model_name: str = "swin3d_t",
    num_segments: int = 32,
    clip_size: int = 16,
    fps: int = 30,
    batch_size: int = 4,
    max_clips_per_segment: Optional[int] = 2,
    num_workers: int = 2,
    device: Optional[str] = None,
    overwrite: bool = False,
    model_instance: Optional[Model] = None,
) -> Dict[str, Any]:
    """Main extraction routine for Kaggle-based feature extraction.

    Args:
        class_video_map: List of lists [[path1, path2], ...], dict {"ClassA": [...]}, or JSON.
        class_names: Names for each class list. Required if class_video_map is list-of-lists.
        output_dir: Root output directory (e.g. 'features').
        model_name: Backbone name (default 'swin3d_t').
        num_segments: Number of temporal segments (default 32).
        clip_size: Clip frame length (default 16).
        fps: Resampling FPS (default 30).
        batch_size: 3D inference batch size.
        max_clips_per_segment: Maximum clips sampled per segment (default 2).
        num_workers: Background worker count for video decoding (0 for sequential).
        device: 'cuda' or 'cpu' (default auto-detect).
        overwrite: If True, overwrites existing feature files.
        model_instance: Optional pre-loaded Model instance to avoid re-loading.

    Returns:
        Summary dict containing counts, manifest, elapsed time, and status.
    """
    video_lists, c_names = normalize_class_map_input(class_video_map, class_names)
    start_time = time.time()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    base_out = os.path.join(output_dir, model_name)
    os.makedirs(base_out, exist_ok=True)

    # 1. Create class subdirectories and build mapping: video_path -> list of (class_name, target_file)
    video_to_targets: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for c_name, paths in zip(c_names, video_lists):
        c_dir = os.path.join(base_out, c_name)
        os.makedirs(c_dir, exist_ok=True)
        for p in paths:
            if not p:
                continue
            stem = strip_video_ext(os.path.basename(p))
            target_pt = os.path.join(c_dir, f"{stem}.pt")
            video_to_targets[p].append((c_name, target_pt))

    print(f"==================================================")
    print(f"🎬 Kaggle Feature Extraction")
    print(f"==================================================")
    print(f"• Model Backbone:      {model_name}")
    print(f"• Output Directory:    {os.path.abspath(base_out)}")
    print(f"• Classes ({len(c_names)}):       {', '.join(c_names)}")
    print(f"• Unique Video Files:  {len(video_to_targets)}")
    print(f"• Total Target Files:  {sum(len(targets) for targets in video_to_targets.values())}")
    print(f"• Device:              {device}")
    print(f"• Workers:             {num_workers} | Batch Size: {batch_size}")
    print(f"==================================================\n")

    # 2. Check for already extracted files and fast copies
    videos_to_decode: List[str] = []
    targets_to_extract: Dict[str, List[Tuple[str, str]]] = {}
    skipped_count = 0
    reused_count = 0

    for v_path, targets in video_to_targets.items():
        missing_targets = []
        existing_target_file = None

        for c_name, target_pt in targets:
            if not overwrite and os.path.exists(target_pt):
                existing_target_file = target_pt
                skipped_count += 1
            else:
                missing_targets.append((c_name, target_pt))

        if not missing_targets:
            continue

        # Fast reuse: if video was already extracted for another class, copy without running GPU!
        if existing_target_file is not None and not overwrite:
            try:
                data = torch.load(existing_target_file, map_location="cpu")
                for c_name, target_pt in missing_targets:
                    copy_data = dict(data)
                    copy_data["class_name"] = c_name
                    torch.save(copy_data, target_pt)
                    reused_count += 1
                continue
            except Exception:
                pass  # If file reading fails, re-extract normally

        videos_to_decode.append(v_path)
        targets_to_extract[v_path] = missing_targets

    print(f"📊 Status before run:")
    print(f"   - Already up-to-date: {skipped_count} target files")
    print(f"   - Fast reused (no GPU needed): {reused_count} target files")
    print(f"   - Videos requiring GPU inference: {len(videos_to_decode)}\n")

    if not videos_to_decode:
        print("✅ All target features are already extracted and up-to-date!")
    else:
        # 3. Initialize model
        if model_instance is not None:
            model = model_instance
        else:
            print(f"⏳ Loading backbone model '{model_name}' on {device}...")
            model = Model(base_model_name=model_name).to(device)
            model.eval()

        failed_videos: List[Dict[str, str]] = []
        success_videos = 0

        # Helper to save tensor to all target destinations
        def _save_video_features(v_path: str, feats_tensor: torch.Tensor):
            v_stem = strip_video_ext(os.path.basename(v_path))
            cpu_feats = feats_tensor.cpu().float()
            for c_name, target_pt in targets_to_extract[v_path]:
                payload = {
                    "feats": cpu_feats,
                    "video_name": v_stem,
                    "video_path": v_path,
                    "class_name": c_name,
                    "model": model_name,
                    "num_segments": num_segments,
                    "clip_size": clip_size,
                    "fps": fps,
                    "feature_dim": cpu_feats.shape[-1],
                }
                torch.save(payload, target_pt)

        if num_workers > 0 and len(videos_to_decode) > 1:
            # Parallel pipeline
            from data.video_dataset import VideoExtractionDataset, collate_extraction

            dataset = VideoExtractionDataset(
                video_paths=videos_to_decode,
                num_segments=num_segments,
                clip_size=clip_size,
                fps=fps,
                max_clips_per_segment=max_clips_per_segment,
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=2,
                collate_fn=collate_extraction,
            )

            pbar = tqdm(loader, desc="🚀 Extracting Features", position=0, leave=True)
            for idx, item in enumerate(pbar):
                v_path = item["video_path"]
                v_stem = item["video_name"]
                pbar.set_postfix({"video": v_stem[:20]})

                if item["status"] != "ok":
                    print(f"\n❌ [ERROR] Decode error on {v_stem}: {item.get('error')}")
                    failed_videos.append({"video": v_path, "error": str(item.get("error"))})
                    continue

                try:
                    segment_tensors = item["segments"]
                    segment_features = []

                    with torch.no_grad():
                        for seg_tensor in segment_tensors:
                            seg_tensor = seg_tensor.to(device)
                            if seg_tensor.dtype == torch.uint8 or seg_tensor.max() > 1.0:
                                seg_tensor = seg_tensor.float() / 255.0

                            if seg_tensor.shape[0] <= batch_size:
                                feats = model.video_model(seg_tensor)
                            else:
                                chunks = torch.split(seg_tensor, batch_size, dim=0)
                                feats = torch.cat([model.video_model(c) for c in chunks], dim=0)

                            seg_feature = feats.mean(dim=0, keepdim=True)
                            segment_features.append(seg_feature.cpu())

                    all_features = torch.cat(segment_features, dim=0).float()
                    _save_video_features(v_path, all_features)
                    success_videos += 1

                except Exception as e:
                    print(f"\n❌ [ERROR] Model inference failed on {v_stem}: {e}")
                    failed_videos.append({"video": v_path, "error": str(e)})

                # Periodic GPU memory cleanup
                if (idx + 1) % 25 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        else:
            # Sequential pipeline fallback
            pbar = tqdm(videos_to_decode, desc="🚀 Extracting Features", position=0, leave=True)
            for idx, v_path in enumerate(pbar):
                v_stem = strip_video_ext(os.path.basename(v_path))
                pbar.set_postfix({"video": v_stem[:20]})

                try:
                    feats = model.extract_features(
                        video_path=v_path,
                        num_segments=num_segments,
                        clip_size=clip_size,
                        fps=fps,
                        batch_size=batch_size,
                        show_progress=False,
                    )
                    _save_video_features(v_path, feats)
                    success_videos += 1
                except Exception as e:
                    print(f"\n❌ [ERROR] Failed on {v_stem}: {e}")
                    failed_videos.append({"video": v_path, "error": str(e)})

                if (idx + 1) % 25 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # 4. Generate manifest.json with counts and statistics
    class_counts = {}
    sample_dim = 768
    for c_name in c_names:
        c_dir = os.path.join(base_out, c_name)
        if os.path.isdir(c_dir):
            files = [f for f in os.listdir(c_dir) if f.endswith(".pt")]
            class_counts[c_name] = len(files)
            if files and sample_dim == 768:
                try:
                    sample_pt = os.path.join(c_dir, files[0])
                    s_data = torch.load(sample_pt, map_location="cpu")
                    if isinstance(s_data, dict) and "feats" in s_data:
                        sample_dim = s_data["feats"].shape[-1]
                except Exception:
                    pass
        else:
            class_counts[c_name] = 0

    manifest = {
        "model": model_name,
        "num_segments": num_segments,
        "clip_size": clip_size,
        "fps": fps,
        "feature_dim": sample_dim,
        "classes": c_names,
        "counts": class_counts,
        "total_files": sum(class_counts.values()),
        "unique_videos": len(video_to_targets),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    manifest_file = os.path.join(base_out, "manifest.json")
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total_time = time.time() - start_time

    print(f"\n==================================================")
    print(f"🎉 Feature Extraction Complete!")
    print(f"==================================================")
    print(f"• Total Saved Files:   {manifest['total_files']}")
    for c_name in c_names:
        print(f"    - {c_name:<16}: {class_counts.get(c_name, 0)} files")
    print(f"• Manifest Saved:      {os.path.abspath(manifest_file)}")
    print(f"• Total Time:          {total_time:.1f}s")
    print(f"==================================================\n")

    return {
        "status": "completed",
        "output_dir": base_out,
        "manifest": manifest,
        "manifest_path": manifest_file,
        "elapsed_time": total_time,
    }


def main():
    args = parse_args()
    extract_features_from_class_map(
        class_video_map=args.class_map,
        class_names=args.class_names,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_segments=args.num_segments,
        clip_size=args.clip_size,
        fps=args.fps,
        batch_size=args.batch_size,
        max_clips_per_segment=args.max_clips_per_segment,
        num_workers=args.num_workers,
        device=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

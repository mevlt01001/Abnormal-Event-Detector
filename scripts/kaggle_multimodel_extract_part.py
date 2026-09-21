"""Kaggle Multi-Model Parallel Feature Extraction Script.

Designed for parallel execution across multiple Kaggle GPU sessions.
Supports automatic multi-GPU utilization (e.g. 2x Tesla T4) in pure FP32 (float32).

Features:
- Simultaneous multi-model extraction across 6 3D CNN / Transformer backbones:
  swin3d_t, mvit_v2_s, r3d_18, mc3_18, r2plus1d_18, s3d.
- Pure FP32 precision (guaranteed 100% compatibility with GTX 1650 Ti / TensorRT).
- Automatic 2x T4 multi-GPU worker spawning: splits partition across GPUs for 2x speedup.
- Single-pass video decoding using Decord with zero black-frame guarantee.
- Adjustable spatio-temporal clip overlap ratio.
- Resume capability: skips previously extracted features.
- Automatic manifest generation and zip packaging for direct download.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
import os
import subprocess
import sys
import time
import zipfile
import torch

# Default parallel execution settings (can be overridden via CLI flags or env vars)
DEFAULT_PART = int(os.environ.get("KAGGLE_PART", 1))
DEFAULT_TOTAL_PARTS = int(os.environ.get("KAGGLE_TOTAL_PARTS", 4))

# Models to extract simultaneously in a single decode pass
MODELS = [
    "swin3d_t",
    "mvit_v2_s",
    "r3d_18",
    "mc3_18",
    "r2plus1d_18",
    "s3d",
]

# Extraction hyperparameters
NUM_SEGMENTS = 32
CLIP_SIZE = 16
OVERLAP_RATIO = 0.50  # 0.50 = 50% overlap between adjacent clips (8-frame stride for 16-frame clips)
TARGET_FPS = 30.0
BATCH_SIZE = 8
OVERWRITE = False

# Output paths
OUTPUT_DIR = "/kaggle/working/features" if os.path.exists("/kaggle") else "features"
ZIP_OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle") else "."

# =========================================================================
# Dataset Root Directories
# =========================================================================
# UCF-Crime Dataset Roots
ucf_abuse_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-1/Anomaly-Videos-Part-1/Abuse"
ucf_arrest_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-1/Anomaly-Videos-Part-1/Arrest"
ucf_arson_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-1/Anomaly-Videos-Part-1/Arson"
ucf_assault_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-1/Anomaly-Videos-Part-1/Assault"
ucf_burglary_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-2/Anomaly-Videos-Part-2/Burglary"
ucf_explosion_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-2/Anomaly-Videos-Part-2/Explosion"
ucf_fighting_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-2/Anomaly-Videos-Part-2/Fighting"
ucf_road_accidents_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-3/Anomaly-Videos-Part-3/RoadAccidents"
ucf_robbery_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-3/Anomaly-Videos-Part-3/Robbery"
ucf_shooting_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-3/Anomaly-Videos-Part-3/Shooting"
ucf_shoplifting_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-4/Anomaly-Videos-Part-4/Shoplifting"
ucf_stealing_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-4/Anomaly-Videos-Part-4/Stealing"
ucf_vandalism_videos_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Anomaly-Videos-Part-4/Anomaly-Videos-Part-4/Vandalism"
ucf_normal_videos_1_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Training-Normal-Videos-Part-1/Training-Normal-Videos-Part-1"
ucf_normal_videos_2_root = "/kaggle/input/datasets/minmints/ufc-crime-full-dataset/Training-Normal-Videos-Part-2/Training-Normal-Videos-Part-2"

# XD-Violence Dataset Roots
xdv_abuse_videos_root = "/kaggle/input/datasets/bypktt/xd-violence/train/Abuse"
xdv_road_accidents_videos_root = "/kaggle/input/datasets/bypktt/xd-violence/train/CarAccident"
xdv_explosion_videos_root = "/kaggle/input/datasets/bypktt/xd-violence/train/Explosion"
xdv_fighting_videos_root = "/kaggle/input/datasets/bypktt/xd-violence/train/Fighting"
xdv_normal_videos_root = "/kaggle/input/datasets/bypktt/xd-violence/train/Normal"
xdv_riot_videos_root = "/kaggle/input/datasets/bypktt/xd-violence/train/Riot"
xdv_shooting_videos_root = "/kaggle/input/datasets/bypktt/xd-violence/train/Shooting"

dataset_targets = [
    (ucf_abuse_videos_root, "ucf_abuse"),
    (ucf_arrest_videos_root, "ucf_arrest"),
    (ucf_arson_videos_root, "ucf_arson"),
    (ucf_assault_videos_root, "ucf_assault"),
    (ucf_burglary_videos_root, "ucf_burglary"),
    (ucf_explosion_videos_root, "ucf_explosion"),
    (ucf_fighting_videos_root, "ucf_fighting"),
    (ucf_road_accidents_videos_root, "ucf_road_accidents"),
    (ucf_robbery_videos_root, "ucf_robbery"),
    (ucf_shooting_videos_root, "ucf_shooting"),
    (ucf_shoplifting_videos_root, "ucf_shoplifting"),
    (ucf_stealing_videos_root, "ucf_stealing"),
    (ucf_vandalism_videos_root, "ucf_vandalism"),
    (ucf_normal_videos_1_root, "ucf_normal_1"),
    (ucf_normal_videos_2_root, "ucf_normal_2"),
    (xdv_abuse_videos_root, "xdv_abuse"),
    (xdv_road_accidents_videos_root, "xdv_car_accident"),
    (xdv_explosion_videos_root, "xdv_explosion"),
    (xdv_fighting_videos_root, "xdv_fighting"),
    (xdv_normal_videos_root, "xdv_normal"),
    (xdv_riot_videos_root, "xdv_riot"),
    (xdv_shooting_videos_root, "xdv_shooting"),
]


def zip_directory(source_dir: str, zip_path: str) -> None:
    """Compresses a directory into a zip archive."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, source_dir)
                zf.write(full_path, rel_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle Multi-Model Parallel Extraction")
    parser.add_argument("--part", type=int, default=DEFAULT_PART, help=f"Partition index (1-based, default: {DEFAULT_PART})")
    parser.add_argument("--total-parts", type=int, default=DEFAULT_TOTAL_PARTS, help=f"Total partitions (default: {DEFAULT_TOTAL_PARTS})")
    parser.add_argument("--overlap", type=float, default=OVERLAP_RATIO, help=f"Clip overlap ratio in [0.0, 0.95) (default: {OVERLAP_RATIO})")
    parser.add_argument("--fps", type=float, default=TARGET_FPS, help=f"Target sampling FPS (default: {TARGET_FPS})")
    parser.add_argument("--num-segments", type=int, default=NUM_SEGMENTS, help=f"Number of temporal segments (default: {NUM_SEGMENTS})")
    parser.add_argument("--clip-size", type=int, default=CLIP_SIZE, help=f"Frames per spatiotemporal clip (default: {CLIP_SIZE})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Clip batch size (default: {BATCH_SIZE})")
    parser.add_argument("--models", type=str, nargs="+", default=MODELS, help=f"Models to extract (default: {', '.join(MODELS)})")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite previously extracted features")
    parser.add_argument("--single-gpu", action="store_true", help="Force single GPU execution even if multiple GPUs are available")
    parser.add_argument("--worker-id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num-workers", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    part = args.part
    total_parts = args.total_parts
    overlap_ratio = args.overlap
    target_fps = args.fps
    num_segments = args.num_segments
    clip_size = args.clip_size
    batch_size = args.batch_size
    models = args.models
    overwrite = args.overwrite
    worker_id = args.worker_id
    num_workers = args.num_workers

    # Set repository working directory if cloned in Kaggle
    for candidate_dir in ["Abnormal-Event-Detector", "Anomaly_Detection", "."]:
        if os.path.isdir(candidate_dir) and os.path.isdir(os.path.join(candidate_dir, "core")):
            sys.path.insert(0, os.path.abspath(candidate_dir))
            break

    from scripts.kaggle_extract import extract_features_from_class_map

    # 1. Automatic Multi-GPU Dispatcher (e.g. Kaggle 2x Tesla T4)
    if worker_id is None and not args.single_gpu and torch.cuda.is_available():
        detected_gpus = torch.cuda.device_count()
        if detected_gpus >= 2:
            print("\n" + "=" * 70)
            print(f"[MULTI-GPU] Detected {detected_gpus} GPUs. Spawning parallel workers for 2x speedup...")
            print("=" * 70)
            for gid in range(detected_gpus):
                gpu_name = torch.cuda.get_device_name(gid)
                print(f"[INFO] GPU {gid}: {gpu_name}")
            print("=" * 70 + "\n")

            procs = []
            this_file = os.path.abspath(__file__)
            for gid in range(detected_gpus):
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gid)
                cmd = [
                    sys.executable,
                    this_file,
                    "--part", str(part),
                    "--total-parts", str(total_parts),
                    "--overlap", str(overlap_ratio),
                    "--fps", str(target_fps),
                    "--num-segments", str(num_segments),
                    "--clip-size", str(clip_size),
                    "--batch-size", str(batch_size),
                    "--worker-id", str(gid),
                    "--num-workers", str(detected_gpus),
                ]
                if overwrite:
                    cmd.append("--overwrite")
                if models != MODELS:
                    cmd.append("--models")
                    cmd.extend(models)

                p = subprocess.Popen(cmd, env=env)
                procs.append(p)

            # Wait for all workers to complete
            all_ok = True
            for gid, p in enumerate(procs):
                code = p.wait()
                if code != 0:
                    print(f"[ERROR] Worker {gid} failed with exit code {code}")
                    all_ok = False

            if not all_ok:
                print("[ERROR] One or more workers failed during multi-GPU extraction.")
                sys.exit(1)

            print("\n" + "=" * 70)
            print(f"[MULTI-GPU] All {detected_gpus} GPU workers finished successfully.")
            print("=" * 70)

            # Create zip archive in parent process
            zip_path = os.path.join(ZIP_OUTPUT_DIR, f"features_part_{part}.zip")
            print(f"\n[INFO] Creating compressed archive: {zip_path} ...")
            zip_directory(OUTPUT_DIR, zip_path)
            zip_size_mb = os.path.getsize(zip_path) / (1024**2)

            print("\n" + "=" * 70)
            print(f"[SUCCESS] Partition {part}/{total_parts} Complete (Multi-GPU)")
            print(f"[INFO] Archive Created:  {zip_path} ({zip_size_mb:.1f} MB)")
            print("=" * 70 + "\n")
            return

    # 2. Discover all videos from dataset roots
    valid_exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    all_video_items = []

    if worker_id is None:
        print("\n" + "=" * 70)
        print(f"[CONFIG] Parallel Partition Setting: PART {part} of {total_parts}")
        print("=" * 70)

    for v_root, class_name in dataset_targets:
        if not os.path.isdir(v_root):
            if worker_id is None:
                print(f"[WARNING] Directory not accessible: {v_root}")
            continue

        files = sorted([
            os.path.join(v_root, f)
            for f in os.listdir(v_root)
            if os.path.splitext(f)[1].lower() in valid_exts
        ])
        for f in files:
            all_video_items.append((class_name, f))

    total_videos = len(all_video_items)
    if total_videos == 0:
        print("[ERROR] No input videos found across specified directories. Please verify dataset mounts.")
        return

    # 3. Compute deterministic partition slice
    chunk_size = math.ceil(total_videos / total_parts)
    start_idx = (part - 1) * chunk_size
    end_idx = min(part * chunk_size, total_videos)
    partition_items = all_video_items[start_idx:end_idx]

    # If in multi-GPU worker mode, split partition items across workers
    if worker_id is not None and num_workers is not None:
        worker_items = partition_items[worker_id::num_workers]
        print(f"[GPU {worker_id}] Assigned {len(worker_items)} videos (total part: {len(partition_items)}) on GPU {worker_id}")
        partition_items = worker_items
    else:
        print(f"[INFO] Total Video Count:      {total_videos}")
        print(f"[INFO] Partition Range:        Indices [{start_idx} : {end_idx}]")
        print(f"[INFO] Videos in this Part:    {len(partition_items)}")
        print(f"[INFO] Models to Extract:      {', '.join(models)}")
        print(f"[INFO] Overlap Ratio:          {overlap_ratio:.2f}")
        print(f"[INFO] Target FPS:              {target_fps:.1f}")
        print(f"[INFO] Segments / Clip Size:   {num_segments} segments, {clip_size} frames/clip")
        print(f"[INFO] Batch Size:             {batch_size}")

    # Build class-video mapping for this worker/partition
    partition_dict = defaultdict(list)
    for c_name, v_path in partition_items:
        partition_dict[c_name].append(v_path)

    part_class_names = list(partition_dict.keys())
    part_video_lists = [partition_dict[k] for k in part_class_names]

    # 4. Run multi-model extraction
    t_start = time.time()
    result = extract_features_from_class_map(
        class_video_map=part_video_lists,
        class_names=part_class_names,
        output_dir=OUTPUT_DIR,
        model_names=models,
        num_segments=num_segments,
        clip_size=clip_size,
        overlap_ratio=overlap_ratio,
        fps=target_fps,
        batch_size=batch_size,
        device=None,
        overwrite=overwrite,
        show_progress=True,
        worker_id=worker_id,
        num_workers=num_workers,
    )
    total_elapsed = time.time() - t_start

    # 5. Create downloadable zip archive (only in single-GPU / non-worker mode)
    if worker_id is None:
        zip_path = os.path.join(ZIP_OUTPUT_DIR, f"features_part_{part}.zip")
        print(f"\n[INFO] Creating compressed archive: {zip_path} ...")
        zip_directory(OUTPUT_DIR, zip_path)
        zip_size_mb = os.path.getsize(zip_path) / (1024**2)

        print("\n" + "=" * 70)
        print(f"[SUCCESS] Partition {part}/{total_parts} Extraction Complete")
        print(f"[INFO] Archive Created:  {zip_path} ({zip_size_mb:.1f} MB)")
        print(f"[INFO] Extraction Time:  {total_elapsed:.1f}s")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

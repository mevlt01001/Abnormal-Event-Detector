"""Kaggle Multi-Model Parallel Feature Extraction Script.

Designed for parallel execution across multiple Kaggle GPU sessions.
Partitioning is managed via PART and TOTAL_PARTS environment or configuration variables.

Features:
- Simultaneous multi-model extraction across 6 3D CNN / Transformer backbones:
  swin3d_t, mvit_v2_s, r3d_18, mc3_18, r2plus1d_18, s3d.
- Single-pass video decoding using Decord with zero black-frame guarantee.
- Adjustable spatio-temporal clip overlap ratio.
- Resume capability: skips previously extracted features.
- Automatic manifest generation and zip packaging for direct download.
"""

from __future__ import annotations

from collections import defaultdict
import math
import os
import sys
import time
import zipfile

# =========================================================================
# Parallel Execution Settings
# =========================================================================
# Set PART to 1, 2, 3, or 4 for each parallel Kaggle instance
PART = int(os.environ.get("KAGGLE_PART", 1))
TOTAL_PARTS = int(os.environ.get("KAGGLE_TOTAL_PARTS", 4))

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
BATCH_SIZE = 4
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
    # Set repository working directory if cloned in Kaggle
    for candidate_dir in ["Abnormal-Event-Detector", "Anomaly_Detection", "."]:
        if os.path.isdir(candidate_dir) and os.path.isdir(os.path.join(candidate_dir, "core")):
            sys.path.insert(0, os.path.abspath(candidate_dir))
            break

    from scripts.kaggle_extract import extract_features_from_class_map

    # 1. Discover all videos from dataset roots
    valid_exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    all_video_items = []

    print("\n" + "=" * 70)
    print(f"[CONFIG] Parallel Partition Setting: PART {PART} of {TOTAL_PARTS}")
    print("=" * 70)

    for v_root, class_name in dataset_targets:
        if not os.path.isdir(v_root):
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

    # 2. Compute deterministic partition slice
    chunk_size = math.ceil(total_videos / TOTAL_PARTS)
    start_idx = (PART - 1) * chunk_size
    end_idx = min(PART * chunk_size, total_videos)
    partition_items = all_video_items[start_idx:end_idx]

    print(f"[INFO] Total Video Count:      {total_videos}")
    print(f"[INFO] Partition Range:        Indices [{start_idx} : {end_idx}]")
    print(f"[INFO] Videos in this Part:    {len(partition_items)}")
    print(f"[INFO] Models to Extract:      {', '.join(MODELS)}")
    print(f"[INFO] Overlap Ratio:          {OVERLAP_RATIO:.2f}")

    # Build class-video mapping for this partition
    partition_dict = defaultdict(list)
    for c_name, v_path in partition_items:
        partition_dict[c_name].append(v_path)

    part_class_names = list(partition_dict.keys())
    part_video_lists = [partition_dict[k] for k in part_class_names]

    print("\n[INFO] Class breakdown for this partition:")
    for c_name in part_class_names:
        print(f"  - {c_name:<25}: {len(partition_dict[c_name])} videos")

    # 3. Run multi-model extraction
    t_start = time.time()
    result = extract_features_from_class_map(
        class_video_map=part_video_lists,
        class_names=part_class_names,
        output_dir=OUTPUT_DIR,
        model_names=MODELS,
        num_segments=NUM_SEGMENTS,
        clip_size=CLIP_SIZE,
        overlap_ratio=OVERLAP_RATIO,
        fps=TARGET_FPS,
        batch_size=BATCH_SIZE,
        device=None,
        overwrite=OVERWRITE,
        show_progress=True,
    )
    total_elapsed = time.time() - t_start

    # 4. Create downloadable zip archive
    zip_path = os.path.join(ZIP_OUTPUT_DIR, f"features_part_{PART}.zip")
    print(f"\n[INFO] Creating compressed archive: {zip_path} ...")
    zip_directory(OUTPUT_DIR, zip_path)
    zip_size_mb = os.path.getsize(zip_path) / (1024**2)

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Partition {PART}/{TOTAL_PARTS} Extraction Complete")
    print(f"[INFO] Archive Created:  {zip_path} ({zip_size_mb:.1f} MB)")
    print(f"[INFO] Extraction Time:  {total_elapsed:.1f}s")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Hierarchical Dataset Organizer for Video Anomaly Detection.

Reorganizes raw extracted features (MViT-v2-S / Swin3D) into strict
train/test splits per dataset:

unified_features/
├── test/
│   ├── ucf/
│   │   ├── Normal/
│   │   ├── Abuse/
│   │   ├── Arrest/
│   │   ├── Arson/
│   │   ├── Assault/
│   │   ├── Burglary/
│   │   ├── Explosion/
│   │   ├── Fighting/
│   │   ├── RoadAccidents/
│   │   ├── Robbery/
│   │   ├── Shooting/
│   │   ├── Shoplifting/
│   │   ├── Stealing/
│   │   └── Vandalism/
│   ├── ucf-test-annotatisons.txt
│   ├── xdv/
│   │   ├── Normal/
│   │   ├── Abuse/
│   │   ├── Explosion/
│   │   ├── Fighting/
│   │   ├── Riot/
│   │   ├── RoadAccidents/
│   │   └── Shooting/
│   └── xdv-test-annotations.txt
└── train/
    ├── ucf/
    │   ├── Normal/
    │   └── ... (13 anomaly classes)
    └── xdv/
        ├── Normal/
        ├── Abuse/
        ├── Explosion/
        ├── Fighting/
        ├── Riot/
        ├── RoadAccidents/
        └── Shooting/
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# UCF-Crime Folder to Class Mapping
UCF_FOLDER_MAP: Dict[str, str] = {
    "ucf_abuse_videos_root": "Abuse",
    "ucf_arrest_videos_root": "Arrest",
    "ucf_arson_videos_root": "Arson",
    "ucf_assault_videos_root": "Assault",
    "ucf_burglary_videos_root": "Burglary",
    "ucf_explosion_videos_root": "Explosion",
    "ucf_fighting_videos_root": "Fighting",
    "ucf_road_accidents_videos_root": "RoadAccidents",
    "ucf_robbery_videos_root": "Robbery",
    "ucf_shooting_videos_root": "Shooting",
    "ucf_shoplifting_videos_root": "Shoplifting",
    "ucf_stealing_videos_root": "Stealing",
    "ucf_vandalism_videos_root": "Vandalism",
    "ucf_normal_test_videos_root": "Normal",
    "ucf_normal_videos_1_root": "Normal",
    "ucf_normal_videos_2_root": "Normal",
}

UCF_CLASSES: List[str] = [
    "Normal",
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
]

# XD-Violence Test & Train Folder Mappings
XDV_TEST_MAP: Dict[str, str] = {
    "xdv_test_normal_videos_root": "Normal",
    "xdv_test_abuse_videos_root": "Abuse",
    "xdv_test_explosion_videos_root": "Explosion",
    "xdv_test_fighting_videos_root": "Fighting",
    "xdv_test_riot_videos_root": "Riot",
    "xdv_test_road_accidents_videos_root": "RoadAccidents",
    "xdv_test_shooting_videos_root": "Shooting",
}

XDV_TRAIN_MAP: Dict[str, str] = {
    "xdv_normal_videos_root": "Normal",
    "xdv_abuse_videos_root": "Abuse",
    "xdv_explosion_videos_root": "Explosion",
    "xdv_fighting_videos_root": "Fighting",
    "xdv_riot_videos_root": "Riot",
    "xdv_road_accidents_videos_root": "RoadAccidents",
    "xdv_shooting_videos_root": "Shooting",
}

XDV_CLASSES: List[str] = [
    "Normal",
    "Abuse",
    "Explosion",
    "Fighting",
    "Riot",
    "RoadAccidents",
    "Shooting",
]


def load_ucf_test_annotations(annot_path: str) -> Dict[str, str]:
    """Loads UCF Crime test annotations (filename -> class)."""
    ucf_annots = {}
    if not os.path.exists(annot_path):
        raise FileNotFoundError(f"UCF annotation file not found: {annot_path}")
    with open(annot_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                ucf_annots[parts[0]] = parts[1]
    return ucf_annots


def transfer_file(src: str, dst: str, use_symlinks: bool = False) -> None:
    """Copies or creates a symlink for a file."""
    if os.path.exists(dst):
        os.remove(dst)
    if use_symlinks:
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def organize_hierarchical_dataset(
    source_root: str = "features_clip_16_overlap_8_fps_20_swin3d-t/mvit_v2_s",
    target_root: str = "data/unified_features",
    ucf_annot_file: str = "UCFcrime-test-annotatisons.txt",
    xdv_annot_file: str = "XDviolanece-test-annotations.txt",
    use_symlinks: bool = False,
) -> Dict[str, int]:
    """Organizes features into train/test and ucf/xdv splits."""
    print("=" * 75)
    print("🚀 ORGANIZING DATASET: HIERARCHICAL SPLITS (TRAIN/TEST - UCF/XDV)")
    print("=" * 75)
    print(f"• Source Directory    : {source_root}")
    print(f"• Target Directory    : {target_root}")
    print(f"• UCF Annotations     : {ucf_annot_file}")
    print(f"• XDV Annotations     : {xdv_annot_file}")
    print(f"• Transfer Mode       : {'Symlink' if use_symlinks else 'Copy'}")

    if not os.path.isdir(source_root):
        raise FileNotFoundError(f"Source directory not found: {source_root}")

    # Load UCF test annotations
    ucf_test_map = load_ucf_test_annotations(ucf_annot_file)
    ucf_test_stems = {os.path.splitext(fname)[0]: cls for fname, cls in ucf_test_map.items()}
    print(f"• Loaded UCF test entries: {len(ucf_test_map)} (150 Normal, {len(ucf_test_map)-150} Anomaly)")

    # Prepare Target Directory
    if os.path.exists(target_root):
        print(f"• Cleaning existing target directory: {target_root}")
        shutil.rmtree(target_root)
    os.makedirs(target_root, exist_ok=True)

    # 1. Create Target Directory Tree
    for split in ["train", "test"]:
        # UCF classes
        for cls in UCF_CLASSES:
            os.makedirs(os.path.join(target_root, split, "ucf", cls), exist_ok=True)
        # XDV classes
        for cls in XDV_CLASSES:
            os.makedirs(os.path.join(target_root, split, "xdv", cls), exist_ok=True)

    counts = {
        "ucf_train": defaultdict(int),
        "ucf_test": defaultdict(int),
        "xdv_train": defaultdict(int),
        "xdv_test": defaultdict(int),
    }

    # -------------------------------------------------------------
    # 2. Process UCF-Crime Features
    # -------------------------------------------------------------
    print("\n📦 Processing UCF-Crime features...")
    for folder, target_cls in UCF_FOLDER_MAP.items():
        src_folder_path = os.path.join(source_root, folder)
        if not os.path.isdir(src_folder_path):
            print(f"  ⚠️ Warning: Folder missing: {src_folder_path}")
            continue

        pts = sorted(glob.glob(os.path.join(src_folder_path, "*.pt")))
        for pt_path in pts:
            fname = os.path.basename(pt_path)
            stem = os.path.splitext(fname)[0]

            if folder == "ucf_normal_test_videos_root":
                dst_dir = os.path.join(target_root, "test", "ucf", "Normal")
                transfer_file(pt_path, os.path.join(dst_dir, fname), use_symlinks)
                counts["ucf_test"]["Normal"] += 1
            elif folder in ("ucf_normal_videos_1_root", "ucf_normal_videos_2_root"):
                dst_dir = os.path.join(target_root, "train", "ucf", "Normal")
                transfer_file(pt_path, os.path.join(dst_dir, fname), use_symlinks)
                counts["ucf_train"]["Normal"] += 1
            else:
                # Anomaly folders: check if stem is in ucf_test_stems
                if stem in ucf_test_stems:
                    annot_cls = ucf_test_stems[stem]
                    dst_dir = os.path.join(target_root, "test", "ucf", annot_cls)
                    transfer_file(pt_path, os.path.join(dst_dir, fname), use_symlinks)
                    counts["ucf_test"][annot_cls] += 1
                else:
                    dst_dir = os.path.join(target_root, "train", "ucf", target_cls)
                    transfer_file(pt_path, os.path.join(dst_dir, fname), use_symlinks)
                    counts["ucf_train"][target_cls] += 1

    # -------------------------------------------------------------
    # 3. Process XD-Violence Features
    # -------------------------------------------------------------
    print("📦 Processing XD-Violence features...")
    # Test folders
    for folder, target_cls in XDV_TEST_MAP.items():
        src_folder_path = os.path.join(source_root, folder)
        if not os.path.isdir(src_folder_path):
            print(f"  ⚠️ Warning: Folder missing: {src_folder_path}")
            continue
        pts = sorted(glob.glob(os.path.join(src_folder_path, "*.pt")))
        dst_dir = os.path.join(target_root, "test", "xdv", target_cls)
        for pt_path in pts:
            fname = os.path.basename(pt_path)
            transfer_file(pt_path, os.path.join(dst_dir, fname), use_symlinks)
            counts["xdv_test"][target_cls] += 1

    # Train folders
    for folder, target_cls in XDV_TRAIN_MAP.items():
        src_folder_path = os.path.join(source_root, folder)
        if not os.path.isdir(src_folder_path):
            print(f"  ⚠️ Warning: Folder missing: {src_folder_path}")
            continue
        pts = sorted(glob.glob(os.path.join(src_folder_path, "*.pt")))
        dst_dir = os.path.join(target_root, "train", "xdv", target_cls)
        for pt_path in pts:
            fname = os.path.basename(pt_path)
            transfer_file(pt_path, os.path.join(dst_dir, fname), use_symlinks)
            counts["xdv_train"][target_cls] += 1

    # -------------------------------------------------------------
    # 4. Copy Annotation Files to test/
    # -------------------------------------------------------------
    print("📋 Copying annotation files to test/ ...")
    test_dir = os.path.join(target_root, "test")
    dst_ucf_annot = os.path.join(test_dir, "ucf-test-annotatisons.txt")
    dst_xdv_annot = os.path.join(test_dir, "xdv-test-annotations.txt")

    if os.path.exists(ucf_annot_file):
        shutil.copy2(ucf_annot_file, dst_ucf_annot)
        print(f"  • UCF Annotations: {dst_ucf_annot}")
    if os.path.exists(xdv_annot_file):
        shutil.copy2(xdv_annot_file, dst_xdv_annot)
        print(f"  • XDV Annotations: {dst_xdv_annot}")

    # -------------------------------------------------------------
    # 5. Metadata and Manifest
    # -------------------------------------------------------------
    sample_pt = next(glob.iglob(os.path.join(target_root, "**", "*.pt"), recursive=True), None)
    model_name = "mvit_v2_s"
    feat_dim = 768
    num_segments = 32
    fps = 20.0
    overlap_ratio = 0.50

    if sample_pt:
        try:
            import torch
            sample_data = torch.load(sample_pt, map_location="cpu", weights_only=False)
            if isinstance(sample_data, dict):
                model_name = sample_data.get("model", model_name)
                fps = sample_data.get("fps", fps)
                overlap_ratio = sample_data.get("overlap_ratio", overlap_ratio)
                if "feats" in sample_data:
                    feat_dim = sample_data["feats"].shape[-1]
                    num_segments = sample_data["feats"].shape[0]
            elif isinstance(sample_data, torch.Tensor):
                feat_dim = sample_data.shape[-1]
                num_segments = sample_data.shape[0]
        except Exception:
            pass

    manifest = {
        "dataset_name": "Hierarchical-UCF-XD-Splits",
        "description": "Partitioned train/test splits for UCF-Crime and XD-Violence without cross-contamination",
        "backbone": model_name,
        "feature_dim": feat_dim,
        "num_segments": num_segments,
        "fps": fps,
        "overlap_ratio": overlap_ratio,
        "ucf_classes": UCF_CLASSES,
        "xdv_classes": XDV_CLASSES,
        "counts": {
            "ucf_train": dict(counts["ucf_train"]),
            "ucf_test": dict(counts["ucf_test"]),
            "xdv_train": dict(counts["xdv_train"]),
            "xdv_test": dict(counts["xdv_test"]),
            "totals": {
                "ucf_train": sum(counts["ucf_train"].values()),
                "ucf_test": sum(counts["ucf_test"].values()),
                "xdv_train": sum(counts["xdv_train"].values()),
                "xdv_test": sum(counts["xdv_test"].values()),
                "grand_total": (
                    sum(counts["ucf_train"].values())
                    + sum(counts["ucf_test"].values())
                    + sum(counts["xdv_train"].values())
                    + sum(counts["xdv_test"].values())
                ),
            },
        },
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    manifest_path = os.path.join(target_root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # -------------------------------------------------------------
    # 6. Summary Display
    # -------------------------------------------------------------
    tot_ucf_tr = sum(counts["ucf_train"].values())
    tot_ucf_te = sum(counts["ucf_test"].values())
    tot_xdv_tr = sum(counts["xdv_train"].values())
    tot_xdv_te = sum(counts["xdv_test"].values())

    print("\n" + "=" * 75)
    print("📊 DATASET ORGANIZATION COMPLETE SUMMARY")
    print("=" * 75)
    print("1. UCF-CRIME:")
    print(f"   • Train : {tot_ucf_tr:5d} files (800 Normal, {tot_ucf_tr-800:4d} Anomaly)")
    print(f"   • Test  : {tot_ucf_te:5d} files (150 Normal, {tot_ucf_te-150:4d} Anomaly)")
    print(f"   • Total : {tot_ucf_tr + tot_ucf_te:5d} files (Expected: 1900)")

    print("\n2. XD-VIOLENCE:")
    print(f"   • Train : {tot_xdv_tr:5d} files ({counts['xdv_train']['Normal']} Normal, {tot_xdv_tr-counts['xdv_train']['Normal']} Anomaly)")
    print(f"   • Test  : {tot_xdv_te:5d} files ({counts['xdv_test']['Normal']} Normal, {tot_xdv_te-counts['xdv_test']['Normal']} Anomaly)")
    print(f"   • Total : {tot_xdv_tr + tot_xdv_te:5d} files")

    print("\n3. GRAND TOTAL:")
    print(f"   • Total Train : {tot_ucf_tr + tot_xdv_tr:5d} files")
    print(f"   • Total Test  : {tot_ucf_te + tot_xdv_te:5d} files")
    print(f"   • All Files   : {tot_ucf_tr + tot_ucf_te + tot_xdv_tr + tot_xdv_te:5d} files")
    print(f"   • Manifest    : {manifest_path}")
    print("=" * 75 + "\n")

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical dataset organizer for UCF-Crime and XD-Violence")
    parser.add_argument(
        "--source-root",
        type=str,
        default="features_clip_16_overlap_8_fps_20_swin3d-t/mvit_v2_s",
        help="Path to source extracted features folder",
    )
    parser.add_argument(
        "--target-root",
        type=str,
        default="data/unified_features",
        help="Target folder for organized hierarchy",
    )
    parser.add_argument(
        "--ucf-annot",
        type=str,
        default="UCFcrime-test-annotatisons.txt",
        help="Path to UCF Crime test annotations file",
    )
    parser.add_argument(
        "--xdv-annot",
        type=str,
        default="XDviolanece-test-annotations.txt",
        help="Path to XD-Violence test annotations file",
    )
    parser.add_argument(
        "--symlinks",
        action="store_true",
        help="Use symlinks instead of copying files",
    )
    args = parser.parse_args()

    organize_hierarchical_dataset(
        source_root=args.source_root,
        target_root=args.target_root,
        ucf_annot_file=args.ucf_annot,
        xdv_annot_file=args.xdv_annot,
        use_symlinks=args.symlinks,
    )

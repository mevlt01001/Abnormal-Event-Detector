#!/usr/bin/env python3
"""Consolidates UCF-Crime and XD-Violence feature datasets into 6 High-Level Macro Classes (+ Normal).

6 Consolidated Macro Classes:
1. Violence       (XD Fighting + UCF Fighting + UCF Assault + XD Abuse + UCF Abuse) -> 558 videos
2. Shooting       (XD Shooting + UCF Shooting) -> 282 videos
3. Explosion_Fire (XD Explosion + UCF Explosion + UCF Arson) -> 373 videos
4. Accident       (XD Road Accidents + UCF Road Accidents) -> 525 videos
5. Riot_Vandalism (XD Riot + UCF Vandalism) -> 424 videos
6. Theft_Robbery  (UCF Robbery + UCF Burglary + UCF Stealing + UCF Shoplifting) -> 400 videos
7. Normal         (XD Normal + UCF Normal Part 1 & 2) -> 2,847 videos

Excluded:
- ucf_arrest_videos_root (50 videos) - Law enforcement restraint with no visual violence signature.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.taxonomy import DEFAULT_MACRO_CLASSES, EXCLUDED_SOURCES, MACRO_TO_SOURCES_MAP

MACRO_TAXONOMY = MACRO_TO_SOURCES_MAP



def organize_unified_dataset(
    source_root: str = "extracted_features",
    target_root: str = "data/unified_features",
    use_symlinks: bool = False,
) -> Dict[str, int]:
    """Organizes raw extracted features into structured macro-class folders."""
    print("=" * 70)
    print("🚀 ORGANIZING DATASET INTO 6 MACRO-CLASSES (+ NORMAL)")
    print("=" * 70)
    print(f"• Source Directory: {source_root}")
    print(f"• Target Directory: {target_root}")

    # Remove existing target directory contents if present
    if os.path.exists(target_root):
        print(f"• Cleaning existing directory: {target_root}")
        shutil.rmtree(target_root)
    os.makedirs(target_root, exist_ok=True)

    class_counts: Dict[str, int] = defaultdict(int)
    source_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for macro_class, source_dirs in MACRO_TAXONOMY.items():
        class_target_dir = os.path.join(target_root, macro_class)
        os.makedirs(class_target_dir, exist_ok=True)

        for src_dir_name in source_dirs:
            src_dir_path = os.path.join(source_root, src_dir_name)
            if not os.path.isdir(src_dir_path):
                print(f"  ⚠️ Warning: Source folder not found: {src_dir_path}")
                continue

            pt_files = sorted(glob.glob(os.path.join(src_dir_path, "*.pt")))
            count = 0
            for file_path in pt_files:
                file_name = os.path.basename(file_path)
                target_file_path = os.path.join(class_target_dir, file_name)

                # Avoid duplicate collisions
                if os.path.exists(target_file_path):
                    prefix = "xdv_" if "xdv" in src_dir_name else "ucf_"
                    target_file_path = os.path.join(class_target_dir, f"{prefix}{file_name}")

                if use_symlinks:
                    os.symlink(os.path.abspath(file_path), target_file_path)
                else:
                    shutil.copy2(file_path, target_file_path)
                count += 1

            class_counts[macro_class] += count
            source_counts[macro_class][src_dir_name] = count

    total_files = sum(class_counts.values())

    # Build and save manifest
    manifest = {
        "dataset_name": "Consolidated-UCF-XD-6MacroClasses",
        "taxonomy": "6 High-Level Macro Classes + Normal",
        "total_files": total_files,
        "feature_dim": 768,
        "num_segments": 32,
        "classes": DEFAULT_MACRO_CLASSES + ["Normal"],
        "anomaly_classes": list(DEFAULT_MACRO_CLASSES),
        "class_counts": dict(class_counts),
        "source_breakdown": {k: dict(v) for k, v in source_counts.items()},
        "excluded_sources": EXCLUDED_SOURCES,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    manifest_path = os.path.join(target_root, "taxonomy_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n📊 CONSOLIDATION SUMMARY:")
    for macro_class, count in class_counts.items():
        pct = (count / total_files) * 100
        print(f"  • {macro_class:<16}: {count:>5} files ({pct:>5.1f}%)")
    print("-" * 50)
    print(f"  • Total Dataset : {total_files:>5} files")
    print(f"  • Excluded Arrest : 50 files (Noise reduction)")
    print(f"  • Saved Manifest: {manifest_path}")
    print("=" * 70 + "\n")

    return dict(class_counts)


if __name__ == "__main__":
    organize_unified_dataset()

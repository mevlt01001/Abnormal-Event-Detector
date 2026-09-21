"""Centralized Taxonomy and Single Source of Truth for Video Anomaly Detection.

Defines:
- Canonical macro-classes and hierarchical mappings
- Dynamic manifest and dataset configuration loader (auto-detects feature_dim & classes)
- Dynamic high-contrast neon color palette generator for publication-grade visualization
"""

from __future__ import annotations

import colorsys
import json
import os
from typing import Any, Dict, List, Optional, Sequence
import torch

# Canonical 6 Macro-Classes (ordered consistently across the entire project)
DEFAULT_MACRO_CLASSES: List[str] = [
    "Violence",
    "Shooting",
    "Explosion_Fire",
    "Accident",
    "Riot_Vandalism",
    "Theft_Robbery",
]

# Canonical mapping from UCF-Crime and XD-Violence feature roots to unified macro-classes
SOURCE_TO_MACRO_MAP: Dict[str, str] = {
    # UCF-Crime
    "ucf_abuse_videos_root": "Violence",
    "ucf_assault_videos_root": "Violence",
    "ucf_fighting_videos_root": "Violence",
    "ucf_shooting_videos_root": "Shooting",
    "ucf_explosion_videos_root": "Explosion_Fire",
    "ucf_arson_videos_root": "Explosion_Fire",
    "ucf_road_accidents_videos_root": "Accident",
    "ucf_vandalism_videos_root": "Riot_Vandalism",
    "ucf_burglary_videos_root": "Theft_Robbery",
    "ucf_robbery_videos_root": "Theft_Robbery",
    "ucf_shoplifting_videos_root": "Theft_Robbery",
    "ucf_stealing_videos_root": "Theft_Robbery",
    "ucf_normal_videos_1_root": "Normal",
    "ucf_normal_videos_2_root": "Normal",
    # XD-Violence
    "xdv_fighting_videos_root": "Violence",
    "xdv_abuse_videos_root": "Violence",
    "xdv_shooting_videos_root": "Shooting",
    "xdv_explosion_videos_root": "Explosion_Fire",
    "xdv_road_accidents_videos_root": "Accident",
    "xdv_riot_videos_root": "Riot_Vandalism",
    "xdv_normal_videos_root": "Normal",
}

# Inverted mapping: Macro-class to list of constituent raw sources
MACRO_TO_SOURCES_MAP: Dict[str, List[str]] = {}
for src, macro in SOURCE_TO_MACRO_MAP.items():
    MACRO_TO_SOURCES_MAP.setdefault(macro, []).append(src)

# Excluded sources (e.g. low-violence or ambiguous classes)
EXCLUDED_SOURCES: List[str] = [
    "ucf_arrest_videos_root",
]

# Curated reference color mapping for standard macro classes
BASE_PRESET_COLORS: Dict[str, str] = {
    "Violence": "#e63946",        # Neon Red
    "Shooting": "#9b5de5",        # Vivid Purple / Magenta
    "Explosion_Fire": "#ffb703",  # Radiant Amber / Gold
    "Accident": "#f77f00",        # Deep Orange
    "Riot_Vandalism": "#00f5d4",  # Neon Cyan / Mint
    "Theft_Robbery": "#00b4d8",   # Cerulean Blue
}


def get_class_color_map(class_names: Sequence[str]) -> Dict[str, str]:
    """Generates a dynamic high-contrast color mapping for any arbitrary list of classes.

    Uses curated preset colors where available, and generates equidistant,
    vibrant HSL colors for any novel or custom class names.

    Args:
        class_names: Sequence of target anomaly class names.

    Returns:
        Dict mapping each class name to a hex color string (e.g. '#e63946').
    """
    color_map = {}
    unmapped = []

    for c in class_names:
        if c in BASE_PRESET_COLORS:
            color_map[c] = BASE_PRESET_COLORS[c]
        else:
            unmapped.append(c)

    if unmapped:
        n = len(unmapped)
        for i, c in enumerate(unmapped):
            hue = (i / max(1, n)) * 0.85
            # Generate vibrant color: 85% saturation, 55% lightness
            rgb = colorsys.hls_to_rgb(hue, 0.55, 0.85)
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255),
                int(rgb[1] * 255),
                int(rgb[2] * 255),
            )
            color_map[c] = hex_color

    return color_map


def load_dataset_taxonomy(features_dir: str = "data/unified_features") -> Dict[str, Any]:
    """Dynamically loads or infers dataset taxonomy, classes, and feature dimensions.

    Single source of truth for training and inference pipelines.
    1. Looks for `taxonomy_manifest.json` in `features_dir`.
    2. If absent, dynamically scans directory folders (excluding 'Normal') and
       inspects the first available .pt feature tensor to discover feature_dim and num_segments.

    Args:
        features_dir: Path to unified features directory.

    Returns:
        Dictionary containing:
            - "anomaly_classes": List[str]
            - "classes": List[str] (including Normal)
            - "num_classes": int
            - "feature_dim": int (e.g. 768)
            - "num_segments": int (e.g. 32)
            - "total_files": int
    """
    manifest_path = os.path.join(features_dir, "taxonomy_manifest.json")
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            anomaly_classes = data.get("anomaly_classes", DEFAULT_MACRO_CLASSES)
            # Ensure canonical ordering if matches default classes
            if set(anomaly_classes) == set(DEFAULT_MACRO_CLASSES):
                anomaly_classes = [c for c in DEFAULT_MACRO_CLASSES if c in anomaly_classes]

            return {
                "anomaly_classes": anomaly_classes,
                "classes": data.get("classes", anomaly_classes + ["Normal"]),
                "num_classes": len(anomaly_classes),
                "feature_dim": data.get("feature_dim", 768),
                "num_segments": data.get("num_segments", 32),
                "total_files": data.get("total_files", 0),
                "class_counts": data.get("class_counts", {}),
                "manifest_path": manifest_path,
            }
        except Exception:
            pass

    # Fallback: Dynamic directory inspection
    discovered_classes = []
    sample_tensor_path = None
    total_files = 0

    if os.path.isdir(features_dir):
        for entry in sorted(os.listdir(features_dir)):
            full_sub = os.path.join(features_dir, entry)
            if os.path.isdir(full_sub) and not entry.startswith("."):
                if entry != "Normal":
                    discovered_classes.append(entry)
                # Count files & find sample
                for root, _, files in os.walk(full_sub):
                    for file in files:
                        if file.endswith(".pt"):
                            total_files += 1
                            if sample_tensor_path is None:
                                sample_tensor_path = os.path.join(root, file)

    # Reorder according to canonical DEFAULT_MACRO_CLASSES if present
    canonical_order = [c for c in DEFAULT_MACRO_CLASSES if c in discovered_classes]
    extra_classes = [c for c in discovered_classes if c not in DEFAULT_MACRO_CLASSES]
    anomaly_classes = canonical_order + extra_classes

    if not anomaly_classes:
        anomaly_classes = list(DEFAULT_MACRO_CLASSES)

    # Inspect sample tensor for dimension & segments
    feature_dim = 768
    num_segments = 32
    if sample_tensor_path and os.path.isfile(sample_tensor_path):
        try:
            t = torch.load(sample_tensor_path, weights_only=False, map_location="cpu")
            if isinstance(t, dict):
                t = next((v for v in t.values() if isinstance(v, torch.Tensor)), None)
            if isinstance(t, torch.Tensor) and t.ndim >= 2:
                num_segments = t.shape[0]
                feature_dim = t.shape[-1]
        except Exception:
            pass

    return {
        "anomaly_classes": anomaly_classes,
        "classes": anomaly_classes + ["Normal"],
        "num_classes": len(anomaly_classes),
        "feature_dim": feature_dim,
        "num_segments": num_segments,
        "total_files": total_files,
        "class_counts": {},
        "manifest_path": None,
    }

"""Centralized Taxonomy and Single Source of Truth for Video Anomaly Detection.

Contains:
- Canonical dataset classes: UCF_CLASSES (14) and XDV_CLASSES (7)
- 4 Macro Wrapper Classes for unified cross-dataset learning
- Mapping rules from raw dataset classes to Wrapper classes
- Helper functions for color mapping and manifest loading
"""

from __future__ import annotations

import colorsys
import json
import os
from typing import Any, Dict, List, Optional, Sequence
import torch

# 1. Canonical Dataset Classes
# UCF-Crime: 1 Normal + 13 Anomaly = 14 classes
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

# XD-Violence: 1 Normal + 6 Anomaly = 7 classes
XDV_CLASSES: List[str] = [
    "Normal",
    "Abuse",
    "Explosion",
    "Fighting",
    "Riot",
    "RoadAccidents",
    "Shooting",
]

# 2. Four Unified Wrapper Classes (+ Normal)
WRAPPER_CLASSES: List[str] = [
    "Violence_Affray",     # Wrapper 1: Fiziksel Şiddet ve Arbede
    "Accidents_Disasters", # Wrapper 2: Kazalar ve Felaketler
    "Gun_Violence",        # Wrapper 3: Silahlı Olaylar
    "Property_Crimes",     # Wrapper 4: Mülke Karşı Suçlar
]

# 3. Exact Matching / Intersection Classes between UCF-Crime and XD-Violence
# The 5 common anomaly categories appearing in both datasets (+ Normal).
EXACT_MATCH_CLASSES: List[str] = [
    "Abuse",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Shooting",
]
INTERSECTION_CLASSES: List[str] = EXACT_MATCH_CLASSES

# Backwards compatibility alias
DEFAULT_MACRO_CLASSES: List[str] = WRAPPER_CLASSES
MACRO_CLASSES: List[str] = WRAPPER_CLASSES

# Mapping rules from dataset classes to the 4 Wrapper classes
RAW_TO_WRAPPER_MAP: Dict[str, Dict[str, str]] = {

    "ucf": {
        # Wrapper 1: Fiziksel Şiddet ve Arbede
        "Fighting": "Violence_Affray",
        "Assault": "Violence_Affray",
        "Abuse": "Violence_Affray",
        "Arrest": "Violence_Affray",
        # Wrapper 2: Kazalar ve Felaketler
        "RoadAccidents": "Accidents_Disasters",
        "Explosion": "Accidents_Disasters",
        "Arson": "Accidents_Disasters",
        # Wrapper 3: Silahlı Olaylar
        "Shooting": "Gun_Violence",
        # Wrapper 4: Mülke Karşı Suçlar
        "Burglary": "Property_Crimes",
        "Robbery": "Property_Crimes",
        "Shoplifting": "Property_Crimes",
        "Stealing": "Property_Crimes",
        "Vandalism": "Property_Crimes",
        # Normal
        "Normal": "Normal",
    },
    "xdv": {
        # Wrapper 1: Fiziksel Şiddet ve Arbede
        "Fighting": "Violence_Affray",
        "Abuse": "Violence_Affray",
        "Riot": "Violence_Affray",
        # Wrapper 2: Kazalar ve Felaketler
        "RoadAccidents": "Accidents_Disasters",
        "Explosion": "Accidents_Disasters",
        # Wrapper 3: Silahlı Olaylar
        "Shooting": "Gun_Violence",
        # Normal
        "Normal": "Normal",
    },
}

# Backwards compatibility alias for annotation parsers
SOURCE_TO_MACRO_MAP: Dict[str, Dict[str, str]] = RAW_TO_WRAPPER_MAP
UNIFIED_TAXONOMY_MAP = RAW_TO_WRAPPER_MAP


def map_to_wrapper(raw_class: str, dataset_origin: str) -> str:

    """Maps a raw dataset class (UCF or XDV) to one of the 4 canonical Wrapper classes or Normal."""
    origin = dataset_origin.lower()
    if origin in RAW_TO_WRAPPER_MAP and raw_class in RAW_TO_WRAPPER_MAP[origin]:
        return RAW_TO_WRAPPER_MAP[origin][raw_class]
    if raw_class == "Normal":
        return "Normal"
    raise KeyError(f"Unknown class '{raw_class}' for dataset origin '{dataset_origin}'")


# 3. Visualization Color Palette
BASE_PRESET_COLORS: Dict[str, str] = {
    "Normal": "#2ec4b6",              # Mint / Teal
    "Violence_Affray": "#e63946",     # Neon Red
    "Accidents_Disasters": "#f77f00", # Deep Orange
    "Gun_Violence": "#9b5de5",        # Vivid Purple
    "Property_Crimes": "#00b4d8",     # Cerulean Blue
    # Individual raw classes
    "Fighting": "#e63946",
    "Assault": "#d90429",
    "Abuse": "#ef233c",
    "Arrest": "#6c757d",
    "Riot": "#00f5d4",
    "Explosion": "#ffb703",
    "Arson": "#f48c06",
    "RoadAccidents": "#f77f00",
    "Shooting": "#9b5de5",
    "Burglary": "#0077b6",
    "Robbery": "#00b4d8",
    "Shoplifting": "#48cae4",
    "Stealing": "#90e0ef",
    "Vandalism": "#00f5d4",
}


def get_class_color_map(class_names: Sequence[str]) -> Dict[str, str]:
    """Generates a dynamic high-contrast color mapping for any list of classes."""
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
            rgb = colorsys.hls_to_rgb(hue, 0.55, 0.85)
            color_map[c] = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255),
                int(rgb[1] * 255),
                int(rgb[2] * 255),
            )

    return color_map


def load_dataset_taxonomy(features_dir: str = "data/s3d_features") -> Dict[str, Any]:
    """Loads feature dimensions, segment counts, and manifest metadata."""
    manifest_path = os.path.join(features_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        for fallback in ["data/s3d_features", "data/mvit_v2_s_features", "data/unified_features"]:
            alt_path = os.path.join(fallback, "manifest.json")
            if os.path.isfile(alt_path):
                manifest_path = alt_path
                break

    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "manifest_path": manifest_path,
                "dataset_name": data.get("dataset_name", "Unified-Features"),
                "backbone": data.get("backbone", "mvit_v2_s"),
                "feature_dim": data.get("feature_dim", 768),
                "num_segments": data.get("num_segments", 32),
                "fps": data.get("fps", 20.0),
                "overlap_ratio": data.get("overlap_ratio", 0.50),
                "ucf_classes": data.get("ucf_classes", UCF_CLASSES),
                "xdv_classes": data.get("xdv_classes", XDV_CLASSES),
                "wrapper_classes": WRAPPER_CLASSES,
                "anomaly_classes": WRAPPER_CLASSES,
                "counts": data.get("counts", {}),
            }
        except Exception:
            pass

    return {
        "manifest_path": None,
        "dataset_name": "Unified-Features",
        "backbone": "mvit_v2_s",
        "feature_dim": 768,
        "num_segments": 32,
        "fps": 20.0,
        "overlap_ratio": 0.50,
        "ucf_classes": UCF_CLASSES,
        "xdv_classes": XDV_CLASSES,
        "wrapper_classes": WRAPPER_CLASSES,
        "anomaly_classes": WRAPPER_CLASSES,
        "counts": {},
    }
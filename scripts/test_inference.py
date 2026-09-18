#!/usr/bin/env python3
"""
Interactive Video Anomaly Detection & Multi-Class Classification Tester.
Supports both raw .mp4 videos and pre-extracted .pt feature files.
"""

from __future__ import annotations

import argparse
import os
import sys
import torch
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import FC_head, Model, create_base_model
from utils.annotation_parser import (
    XD_CLASSES,
    CLASS_NAMES,
    is_normal_video,
    get_video_classes,
    parse_annotations,
    strip_video_ext,
)
from utils.video_utils import fetch_video_segments, get_video_length


def load_heads(model_name: str, device: str = "cpu"):
    feature_dims = {
        "swin3d_t": 768,
        "mvit_v1_b": 768,
        "mc3_18": 512,
        "r3d_18": 512,
        "r2plus1d_18": 512,
        "s3d": 1024,
    }
    feat_dim = feature_dims.get(model_name, 768)

    binary_head = FC_head(in_features=feat_dim, num_classes=1, use_sigmoid=True).to(device)
    multiclass_head = FC_head(in_features=feat_dim, num_classes=6, use_sigmoid=False).to(device)

    bin_ckpt_path = f"checkpoints/binary/{model_name}/best_loss_fold_1.pt"
    mul_ckpt_path = f"checkpoints/multiclass/{model_name}/best_loss_fold_1.pt"

    if os.path.exists(bin_ckpt_path):
        b_ckpt = torch.load(bin_ckpt_path, map_location=device)
        s_dict = b_ckpt.get("state_dict", b_ckpt.get("model_state_dict", b_ckpt))
        binary_head.load_state_dict(s_dict)
    else:
        print(f"[WARN] Binary checkpoint not found at {bin_ckpt_path}")

    if os.path.exists(mul_ckpt_path):
        m_ckpt = torch.load(mul_ckpt_path, map_location=device)
        s_dict = m_ckpt.get("model_state_dict", m_ckpt.get("state_dict", m_ckpt))
        multiclass_head.load_state_dict(s_dict)
    else:
        print(f"[WARN] Multi-class checkpoint not found at {mul_ckpt_path}")

    binary_head.eval()
    multiclass_head.eval()
    return binary_head, multiclass_head, feat_dim


def render_ascii_timeline(scores: list[float], num_chars: int = 32, threshold: float = 0.5) -> str:
    """Renders a text-based ASCII sparkline/bar representation of the 32 segments."""
    blocks = "  ▂▃▄▅▆▇█"
    out = []
    for s in scores:
        val = max(0.0, min(1.0, float(s)))
        idx = int(val * (len(blocks) - 1))
        char = blocks[idx]
        if val >= threshold:
            out.append(f"\033[91m{char}\033[0m") # Red for anomaly
        else:
            out.append(f"\033[92m{char}\033[0m") # Green for normal
    return "".join(out)


def test_single_video(
    video_path_or_name: str,
    model_name: str = "swin3d_t",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    binary_threshold: float = 0.5,
    class_threshold: float = 0.25,
    backbone_cache: dict | None = None,
):
    base_name = strip_video_ext(os.path.basename(video_path_or_name))
    binary_head, multiclass_head, feat_dim = load_heads(model_name, device)

    # 1. Check if pre-extracted features exist
    feat_candidates = [
        f"data/features/{model_name}/normal/{base_name}.pt",
        f"data/features/{model_name}/anomal/{base_name}.pt",
        video_path_or_name if video_path_or_name.endswith(".pt") else None,
    ]
    feat_path = None
    for cand in feat_candidates:
        if cand and os.path.exists(cand):
            feat_path = cand
            break

    features = None
    video_sec = 0.0

    # If raw video file exists
    mp4_candidates = [
        video_path_or_name if video_path_or_name.endswith(".mp4") else None,
        os.path.join("XD-Violence-test-videos", f"{base_name}.mp4"),
    ]
    raw_video_path = None
    for cand in mp4_candidates:
        if cand and os.path.exists(cand):
            raw_video_path = cand
            break

    if raw_video_path:
        video_sec = get_video_length(raw_video_path)

    if feat_path:
        loaded = torch.load(feat_path, map_location=device)
        if isinstance(loaded, dict):
            features = loaded.get("feats", loaded).float()
            if not raw_video_path and "video_path" in loaded and os.path.exists(loaded["video_path"]):
                raw_video_path = loaded["video_path"]
                video_sec = get_video_length(raw_video_path)
        else:
            features = loaded.float()
    elif raw_video_path:
        print(f"[*] Extracting features on-the-fly for {base_name} using {model_name}...")
        if backbone_cache is not None and model_name in backbone_cache:
            backbone = backbone_cache[model_name]
        else:
            backbone = create_base_model(model_name).to(device)
            backbone.eval()
            if backbone_cache is not None:
                backbone_cache[model_name] = backbone

        model_wrapper = Model(model_name).to(device)
        model_wrapper.video_model = backbone
        features = model_wrapper.extract_features(
            video_path=raw_video_path,
            num_segments=32,
            clip_size=16,
            fps=30,
            batch_size=4,
            show_progress=True,
        ).to(device)
    else:
        raise FileNotFoundError(f"Could not find video or features for: {video_path_or_name}")

    # 2. Run Inference
    with torch.no_grad():
        features = features.to(device) # [32, D]
        bin_scores = binary_head(features).squeeze(-1).cpu().numpy() # [32]
        multi_logits = multiclass_head(features) # [32, 6]
        multi_probs = torch.sigmoid(multi_logits).cpu().numpy() # [32, 6]

    # Aggregations
    max_bin_score = float(np.max(bin_scores))
    mean_bin_score = float(np.mean(bin_scores))
    is_anomaly = max_bin_score >= binary_threshold

    # Multi-class max probabilities across segments
    max_class_probs = np.max(multi_probs, axis=0) # [6]

    # Ground truth info
    is_normal = is_normal_video(base_name)
    gt_type = "NORMAL" if is_normal else "ANOMALOUS"
    classes_found = get_video_classes(base_name)
    gt_classes = [CLASS_NAMES.get(c, c) for c in classes_found]

    # Find anomaly temporal intervals (assuming 32 uniform segments)
    dt_seg = (video_sec / 32.0) if video_sec > 0 else 1.0
    detected_intervals = []
    in_interval = False
    start_seg = 0
    for i, s in enumerate(bin_scores):
        if s >= binary_threshold and not in_interval:
            in_interval = True
            start_seg = i
        elif s < binary_threshold and in_interval:
            in_interval = False
            detected_intervals.append((start_seg * dt_seg, i * dt_seg, float(np.max(bin_scores[start_seg:i]))))
    if in_interval:
        detected_intervals.append((start_seg * dt_seg, 32 * dt_seg, float(np.max(bin_scores[start_seg:32]))))

    # Print clean formatted report
    print("\n" + "=" * 70)
    print(f"🎬 VIDEO TEST REPORT: {base_name}")
    print("=" * 70)
    if video_sec > 0:
        print(f"⏱️  Duration:        {video_sec:.1f} seconds ({video_sec/60:.2f} min)")
    print(f"🧠 Backbone Model:  {model_name.upper()} (Top Ranked)")
    print(f"🏷️  Ground Truth:    {gt_type} {'[' + ', '.join(gt_classes) + ']' if gt_classes else ''}")
    verdict_str = "🔴 ANOMALY DETECTED" if is_anomaly else "🟢 NORMAL VIDEO"
    print(f"🎯 Binary Verdict:  {verdict_str} (Peak: {max_bin_score*100:.1f}%, Mean: {mean_bin_score*100:.1f}%)")

    timeline_viz = render_ascii_timeline(bin_scores.tolist(), threshold=binary_threshold)
    print(f"📊 32-Seg Timeline: [{timeline_viz}]")

    if detected_intervals and video_sec > 0:
        print(f"🚨 Detected Violent Segments (Time Ranges):")
        for s_t, e_t, peak in detected_intervals:
            print(f"   • {s_t:05.1f}s - {e_t:05.1f}s (Peak Score: {peak*100:.1f}%)")

    print(f"📋 Multi-Class Anomaly Classification:")
    detected_any_class = False
    for code, prob in zip(XD_CLASSES, max_class_probs):
        c_name = CLASS_NAMES.get(code, code)
        is_det = prob >= class_threshold
        marker = " 👈 DETECTED" if is_det else ""
        if is_det:
            detected_any_class = True
        pct = prob * 100
        bar = "█" * int(pct / 5)
        print(f"   - [{code}] {c_name:<14}: {pct:5.1f}% |{bar:<20}|{marker}")

    if not is_anomaly and not detected_any_class:
        print("   -> No abnormal violence activity classified.")

    print("=" * 70)
    return {
        "video": base_name,
        "is_anomaly": is_anomaly,
        "max_score": max_bin_score,
        "class_probs": dict(zip(XD_CLASSES, [float(p) for p in max_class_probs])),
    }


def main():
    parser = argparse.ArgumentParser(description="Test Video Anomaly Detection Models on Videos")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path or name of video (e.g. Bad.Boys.1995__#01-11-55_01-12-40_label_G-B2-B6.mp4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="swin3d_t",
        choices=["swin3d_t", "mvit_v1_b", "s3d", "r3d_18", "mc3_18", "r2plus1d_18"],
        help="Model backbone to use (swin3d_t is #1 overall, mvit_v1_b is #1 localization)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo test on 1 Normal video and 2 Anomalous videos automatically",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    if args.demo or args.video is None:
        print("\n🚀 RUNNING VERIFICATION DEMO ON SAMPLE TEST VIDEOS...")
        samples = [
            # 1. Normal Video
            "A.Beautiful.Mind.2001__#00-25-20_00-29-20_label_A",
            # 2. Violence: Explosion (G), Shooting (B2), Car Accident (B6)
            "Bad.Boys.1995__#01-11-55_01-12-40_label_G-B2-B6",
            # 3. Violence: Riot / Protest (B4)
            "v=m8EkFsaGPzU__#1_label_B4-0-0",
        ]
        backbone_cache = {}
        for s in samples:
            test_single_video(s, model_name=args.model, device=args.device, backbone_cache=backbone_cache)
    else:
        test_single_video(args.video, model_name=args.model, device=args.device)


if __name__ == "__main__":
    main()

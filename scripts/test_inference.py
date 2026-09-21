#!/usr/bin/env python3
"""Unified Interactive and Batch Video Anomaly Inference Script.

Supports both raw .mp4 video files and pre-extracted .pt feature tensors.
Equipped with dynamic target FPS configuration (no hardcoding), timeline plotting,
and temporal anomaly segment extraction.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import List, Optional, Tuple

# Ensure repository root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from core.feature_extractor import FeatureExtractor
from core.video_analyzer import VideoAnalyzer
from data.dataset import load_feature_tensor
from data.taxonomy import DEFAULT_MACRO_CLASSES, load_dataset_taxonomy
from model.backbone import create_backbone
from model.head import AnomalyHead
from model.model import AnomalyDetector


def load_trained_head(
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[AnomalyHead, List[str]]:
    """Loads AnomalyHead dynamically from checkpoint, discovering classes and architecture."""
    device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        # Auto-discover best checkpoint
        candidates = sorted(glob.glob("checkpoints/class_aware_8fold/best_map_fold_*.pt"))
        if not candidates:
            candidates = sorted(glob.glob("checkpoints/multiclass_8fold/best_map_fold_*.pt"))
        if candidates:
            checkpoint_path = candidates[0]
            print(f"ℹ️  No checkpoint specified. Auto-discovered best: {checkpoint_path}")
        else:
            tax = load_dataset_taxonomy()
            print(f"⚠️ No trained checkpoint found. Initializing initialized AnomalyHead for testing.")
            head = AnomalyHead(in_features=tax["feature_dim"], num_classes=tax["num_classes"]).to(device)
            return head, tax["anomaly_classes"]

    head, ckpt = AnomalyHead.load_from_checkpoint(checkpoint_path, device=device)
    class_list = ckpt.get("config", {}).get("class_list", ckpt.get("class_list", DEFAULT_MACRO_CLASSES))
    print(f"✓ Successfully loaded weights from {checkpoint_path} ({len(class_list)} classes, {head.in_features} dims)")
    return head, list(class_list)


def main():
    parser = argparse.ArgumentParser(description="Class-Aware Video Anomaly Inference Engine")
    parser.add_argument("input_path", type=str, help="Path to raw .mp4 video file, .pt feature file, or directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/class_aware_8fold/best_map_fold_3.pt", help="Path to trained checkpoint (.pt)")
    parser.add_argument("--threshold", type=float, default=0.35, help="Anomaly decision threshold (default: 0.35)")
    parser.add_argument("--target-fps", type=float, default=30.0, help="Sampling frame rate for raw video (default: 30.0)")
    parser.add_argument("--output-dir", type=str, default="results/test_analyses", help="Output directory for plots")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"❌ Input path not found: {args.input_path}")
        sys.exit(1)

    if os.path.isdir(args.input_path):
        print(f"📁 Dizin tespit edildi: {args.input_path}. Toplu test çalıştırıcısına yönlendiriliyor...")
        from scripts.run_all_test_videos import main as run_batch_main
        sys.argv = [
            "run_all_test_videos.py",
            "--test-dir", args.input_path,
            "--checkpoint", args.checkpoint,
            "--threshold", str(args.threshold),
            "--target-fps", str(args.target_fps),
            "--output-dir", args.output_dir,
            "--device", args.device,
        ]
        run_batch_main()
        return

    device = torch.device(args.device)
    head, class_names = load_trained_head(args.checkpoint, device=device)
    analyzer = VideoAnalyzer(
        model=head,
        class_names=class_names,
        device=device,
        target_fps=args.target_fps,
    )


    is_pt = args.input_path.endswith(".pt")
    video_stem = os.path.splitext(os.path.basename(args.input_path))[0]
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print(f"🎬 ANALYZING: {os.path.basename(args.input_path)}")
    print(f"• Input Type:  {'Precomputed Feature Tensor (.pt)' if is_pt else 'Raw Video (.mp4)'}")
    print(f"• Target FPS:  {args.target_fps:.1f} FPS")
    print(f"• Threshold:   {args.threshold:.2f}")
    print(f"• Device:      {device}")
    print("=" * 65)

    start_t = time.time()

    if is_pt:
        features = load_feature_tensor(args.input_path)  # [32, 768]
        duration = 60.0  # Nominal duration if not specified
    else:
        print("• Extracting 32-segment features using Swin3D-T backbone...")
        backbone = create_backbone("swin3d_t", pretrained=True).to(device)
        extractor = FeatureExtractor(backbone=backbone, device=device, target_fps=args.target_fps)
        features = extractor.extract_video(args.input_path, num_segments=32, show_progress=True)
        meta = analyzer.processor.get_video_metadata(args.input_path)
        duration = meta["duration_sec"]

    # Run Class-Aware Analysis
    result = analyzer.analyze_features(
        features=features,
        video_duration_sec=duration,
        threshold=args.threshold,
        video_name=video_stem,
    )

    elapsed = time.time() - start_t

    # Render Plot
    plot_path = os.path.join(args.output_dir, f"{video_stem}_timeline.png")
    analyzer.render_timeline_plot(result, save_path=plot_path)

    # Print Summary
    print("\n📊 İNCELEME SONUCU:")
    print(f"  • Durum:           {'🚨 ANOMALİ TESPİT EDİLDİ!' if result['is_anomaly'] else '✅ TEMİZ / NORMAL'}")
    print(f"  • Tepe Skor (Max): %{result['peak_score']*100:.1f}")
    print(f"  • Analiz Süresi:   {elapsed:.2f} saniye")
    print(f"  • Grafik:          {plot_path}")

    if result["detected_classes"]:
        print("\n🏷️  Eşiği Aşan Sınıflar:")
        for dc in result["detected_classes"]:
            print(f"    - {dc['class']:<16}: %{dc['peak_confidence']*100:.1f}")

    if result["detected_intervals"]:
        print("\n⏱️  Tespit Edilen Olay Zaman Aralıkları:")
        for idx, inter in enumerate(result["detected_intervals"], 1):
            print(f"    {idx}. [{inter['start_time']:.1f}s - {inter['end_time']:.1f}s] {inter['top_class']} (Pik: %{inter['peak_score']*100:.1f})")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
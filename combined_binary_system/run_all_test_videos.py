#!/usr/bin/env python3
"""Batch Anomaly Inference Runner on all Test_Videos using Combined UCF & XD Binary Model with Feature Caching."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

# Ensure repo root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from core.feature_extractor import FeatureExtractor
from model.backbone import create_backbone
from combined_binary_system.infer import CombinedBinaryAnalyzer
from combined_binary_system.model import DeepBinaryAnomalyHead


def main():
    parser = argparse.ArgumentParser(description="Run Combined UCF & XD Binary Model on all Test_Videos with Feature Caching")
    parser.add_argument("--test-dir", type=str, default="Test_Videos", help="Directory containing test .mp4 videos")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="combined_binary_system/checkpoints/best_combined_binary_model.pt",
        help="Model checkpoint path",
    )
    parser.add_argument("--target-fps", type=float, default=30.0, help="Frame rate for video clip extraction")
    parser.add_argument("--threshold", type=float, default=0.35, help="Anomaly decision threshold (default: 0.35)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/combined_binary_analyses",
        help="Output directory for plots and reports",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="features_cache/test_videos",
        help="Directory to cache extracted features so extraction is only done once",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Force re-extraction of video features even if already cached",
    )
    parser.add_argument(
        "--show-intervals",
        action="store_true",
        help="Show shaded red anomaly interval highlights on timeline plots (default: False)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("Please train the model first with: python combined_binary_system/train.py")
        sys.exit(1)

    device = torch.device(args.device)
    head, ckpt = DeepBinaryAnomalyHead.load_from_checkpoint(args.checkpoint, device=device)
    analyzer = CombinedBinaryAnalyzer(model=head, device=device, target_fps=args.target_fps)

    video_files = sorted(glob.glob(os.path.join(args.test_dir, "*.mp4")))
    if not video_files:
        print(f"❌ No .mp4 videos found in {args.test_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    print("\n" + "=" * 75)
    print("🎬 COMBINED UCF-CRIME & XD-VIOLENCE BINARY ANOMALY DETECTION: BATCH RUN")
    print("=" * 75)
    print(f"• Model Checkpoint: {args.checkpoint} (Epoch {ckpt.get('epoch', '?')}, Val AUC: {ckpt.get('val_roc_auc', 0.0):.4f})")
    print(f"• Test Videos:      {len(video_files)} files in '{args.test_dir}'")
    print(f"• Target FPS:       {args.target_fps:.1f} FPS")
    print(f"• Threshold:        {args.threshold:.2f}")
    print(f"• Cache Directory:  {args.cache_dir}")
    print(f"• Output Directory: {args.output_dir}")
    print(f"• Device:           {device}")
    print("=" * 75 + "\n")

    # Check which videos need feature extraction
    fps_tag = int(args.target_fps) if args.target_fps.is_integer() else f"{args.target_fps:.1f}"
    videos_needing_extraction = []

    if not args.force_extract:
        for vpath in video_files:
            stem = os.path.splitext(os.path.basename(vpath))[0]
            cache_file = os.path.join(args.cache_dir, f"{stem}_swin3d_fps{fps_tag}.pt")
            if not os.path.isfile(cache_file):
                videos_needing_extraction.append(vpath)
    else:
        videos_needing_extraction = list(video_files)

    # Lazy load backbone only if some videos need extraction
    extractor = None
    if videos_needing_extraction:
        print(f"• {len(videos_needing_extraction)} video için öznitelik çıkarımı gerekiyor. Swin3D-T omurgası yükleniyor...")
        backbone = create_backbone("swin3d_t", pretrained=True).to(device)
        extractor = FeatureExtractor(backbone=backbone, device=device, target_fps=args.target_fps)
    else:
        print("⚡ Tüm test videolarının öznitelikleri önbellekte (cache) hazır! Ağır omurga yüklenmeden doğrudan test ediliyor...")

    results = []
    overall_start = time.time()

    for i, vpath in enumerate(video_files, 1):
        vname = os.path.basename(vpath)
        stem = os.path.splitext(vname)[0]
        cache_file = os.path.join(args.cache_dir, f"{stem}_swin3d_fps{fps_tag}.pt")

        t0 = time.time()

        if os.path.isfile(cache_file) and not args.force_extract:
            cached_data = torch.load(cache_file, map_location="cpu", weights_only=False)
            if isinstance(cached_data, dict) and "feats" in cached_data:
                features = cached_data["feats"]
                duration_sec = cached_data.get("duration_sec", None)
            else:
                features = cached_data
                duration_sec = None

            if duration_sec is None:
                meta = analyzer.processor.get_video_metadata(vpath)
                duration_sec = meta["duration_sec"]

            load_source = "⚡ Önbellek"
        else:
            meta = analyzer.processor.get_video_metadata(vpath)
            duration_sec = meta["duration_sec"]
            features = extractor.extract_video(vpath, num_segments=32, show_progress=False)

            # Save to cache for future runs
            torch.save(
                {
                    "feats": features.cpu(),
                    "duration_sec": duration_sec,
                    "target_fps": args.target_fps,
                    "model": "swin3d_t",
                },
                cache_file,
            )
            load_source = "🚀 Yeni Çıkarıldı"

        analysis = analyzer.analyze_features(
            features=features,
            video_duration_sec=duration_sec,
            threshold=args.threshold,
            video_name=stem,
        )

        plot_path = os.path.join(args.output_dir, f"{stem}_combined_binary_timeline.png")
        analyzer.render_timeline_plot(analysis, save_path=plot_path, show_intervals=args.show_intervals)
        el_time = time.time() - t0

        status_str = "🚨 ANOMALİ" if analysis["is_anomaly"] else "✅ NORMAL"
        print(f"[{i:02d}/{len(video_files):02d}] {vname:<20} ({load_source}) -> {status_str} | Tepe: %{analysis['peak_score']*100:.1f} | Süre: {duration_sec:.1f}s ({el_time:.2f}s)")
        if analysis["detected_intervals"]:
            for det in analysis["detected_intervals"]:
                print(f"        • [{det['start_time']:.1f}s - {det['end_time']:.1f}s] Pik: %{det['peak_score']*100:.1f}")

        results.append({
            "video_name": vname,
            "stem": stem,
            "duration_sec": duration_sec,
            "peak_score": analysis["peak_score"],
            "is_anomaly": analysis["is_anomaly"],
            "intervals": analysis["detected_intervals"],
            "plot_path": plot_path,
        })

    total_elapsed = time.time() - overall_start

    # Save summary report in json
    summary_json_path = os.path.join(args.output_dir, "batch_inference_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 75)
    print("📋 BATCH TEST INFERENCE SUMMARY TABLE")
    print("=" * 75)
    print(f"{'Video Adı':<20} | {'Süre (s)':<10} | {'Tepe Skor':<12} | {'Durum':<12} | {'Anomali Aralık Sayısı'}")
    print("-" * 75)
    for r in results:
        status_txt = "ANOMALİ 🚨" if r["is_anomaly"] else "NORMAL ✅"
        print(f"{r['video_name']:<20} | {r['duration_sec']:<10.1f} | %{r['peak_score']*100:<11.1f} | {status_txt:<12} | {len(r['intervals'])} aralık")
    print("=" * 75)
    print(f"✓ Tüm test videoları {total_elapsed:.2f}s içinde başarıyla analiz edildi.")
    print(f"★ Zaman çizelgesi grafikleri: {args.output_dir} dizinine kaydedildi.\n")


if __name__ == "__main__":
    main()

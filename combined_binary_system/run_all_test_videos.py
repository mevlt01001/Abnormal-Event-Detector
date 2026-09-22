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
    parser.add_argument("--target-fps", type=float, default=20.0, help="Frame rate for video clip extraction (default: 20.0)")
    parser.add_argument("--clip-size", type=int, default=16, help="Frames per spatio-temporal clip (default: 16)")
    parser.add_argument(
        "--overlap",
        type=float,
        default=8.0,
        help="Overlap between adjacent clips. If >= 1, frame count (e.g. 8); if < 1.0, overlap ratio (e.g. 0.5) (default: 8.0)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Explicit temporal stride between clips in frames. If specified, overrides --overlap",
    )
    parser.add_argument(
        "--max-clips-per-segment",
        type=int,
        default=24,
        help="Maximum clips sampled per segment to prevent memory blowups on long videos (default: 16)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="mvit_v2_s",
        help="3D backbone model for feature extraction (default: mvit_v2_s, options: mvit_v2_s, swin3d_t, r3d_18, etc.)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Clip batch size passed to GPU during feature extraction (default: 8)",
    )
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
    analyzer = CombinedBinaryAnalyzer(
        model=head,
        device=device,
        target_fps=args.target_fps,
        clip_size=args.clip_size,
        overlap=args.overlap,
        stride=args.stride,
    )

    video_files = sorted(glob.glob(os.path.join(args.test_dir, "*.mp4")))
    if not video_files:
        print(f"❌ No .mp4 videos found in {args.test_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # Determine effective overlap and stride representation
    if args.stride is not None:
        eff_stride = max(1, int(args.stride))
        eff_overlap = max(0, args.clip_size - eff_stride)
        overlap_desc = f"stride={eff_stride}f (overlap={eff_overlap}f)"
    elif args.overlap >= 1.0:
        eff_overlap = int(args.overlap)
        eff_stride = max(1, args.clip_size - eff_overlap)
        overlap_desc = f"{eff_overlap} frames ({(eff_overlap / args.clip_size)*100:.1f}%) [stride={eff_stride}f]"
    else:
        eff_ratio = max(0.0, min(0.95, float(args.overlap)))
        eff_stride = max(1, int(round(args.clip_size * (1.0 - eff_ratio))))
        eff_overlap = max(0, args.clip_size - eff_stride)
        overlap_desc = f"{eff_ratio*100:.1f}% ({eff_overlap} frames) [stride={eff_stride}f]"

    fps_tag = int(args.target_fps) if args.target_fps.is_integer() else f"{args.target_fps:.1f}"
    clip_tag = f"c{args.clip_size}"
    overlap_tag = f"ov{eff_overlap}"
    backbone_tag = args.backbone.replace("-", "_").lower()

    def get_cache_file(vpath: str) -> str:
        stem = os.path.splitext(os.path.basename(vpath))[0]
        primary_cache = os.path.join(
            args.cache_dir, f"{stem}_{backbone_tag}_{clip_tag}_{overlap_tag}_fps{fps_tag}.pt"
        )
        if os.path.isfile(primary_cache):
            return primary_cache
        # Legacy fallback if default swin3d_t without overlap was previously cached
        legacy_cache = os.path.join(args.cache_dir, f"{stem}_swin3d_fps{fps_tag}.pt")
        if (
            os.path.isfile(legacy_cache)
            and backbone_tag == "swin3d_t"
            and args.clip_size == 16
            and eff_overlap == 0
        ):
            return legacy_cache
        return primary_cache

    print("\n" + "=" * 75)
    print("🎬 COMBINED UCF-CRIME & XD-VIOLENCE BINARY ANOMALY DETECTION: BATCH RUN")
    print("=" * 75)
    print(f"• Model Checkpoint: {args.checkpoint} (Epoch {ckpt.get('epoch', '?')}, Val AUC: {ckpt.get('val_roc_auc', 0.0):.4f})")
    print(f"• Test Videos:      {len(video_files)} files in '{args.test_dir}'")
    print(f"• Backbone Model:   {args.backbone}")
    print(f"• Target FPS:       {args.target_fps:.1f} FPS")
    print(f"• Clip Size:        {args.clip_size} frames")
    print(f"• Overlap / Stride: {overlap_desc}")
    print(f"• Max Clips / Seg:  {args.max_clips_per_segment}")
    print(f"• Threshold:        {args.threshold:.2f}")
    print(f"• Cache Directory:  {args.cache_dir}")
    print(f"• Output Directory: {args.output_dir}")
    print(f"• Device:           {device}")
    print("=" * 75 + "\n")

    # Check which videos need feature extraction
    videos_needing_extraction = []
    if not args.force_extract:
        for vpath in video_files:
            cfile = get_cache_file(vpath)
            if not os.path.isfile(cfile):
                videos_needing_extraction.append(vpath)
    else:
        videos_needing_extraction = list(video_files)

    # Lazy load backbone only if some videos need extraction
    extractor = None
    if videos_needing_extraction:
        print(f"• {len(videos_needing_extraction)} video için öznitelik çıkarımı gerekiyor. {args.backbone} omurgası yükleniyor...")
        backbone = create_backbone(args.backbone, pretrained=True).to(device)
        extractor = FeatureExtractor(
            backbone=backbone,
            device=device,
            target_fps=args.target_fps,
            clip_size=args.clip_size,
            overlap=args.overlap,
            stride=args.stride,
            max_clips_per_segment=args.max_clips_per_segment,
            batch_size=args.batch_size,
        )
    else:
        print("⚡ Tüm test videolarının öznitelikleri önbellekte (cache) hazır! Ağır omurga yüklenmeden doğrudan test ediliyor...")

    results = []
    overall_start = time.time()

    for i, vpath in enumerate(video_files, 1):
        vname = os.path.basename(vpath)
        stem = os.path.splitext(vname)[0]
        cache_file = get_cache_file(vpath)

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
                    "clip_size": args.clip_size,
                    "overlap": eff_overlap,
                    "stride": eff_stride,
                    "model": args.backbone,
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

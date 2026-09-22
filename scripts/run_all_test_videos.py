#!/usr/bin/env python3
"""Batch Class-Aware Video Anomaly Inference Script for Test_Videos.

Evaluates trained multi-class AnomalyHead checkpoints (such as checkpoints/class_aware_8fold/best_map_fold_3.pt)
across all videos in the Test_Videos directory.
Utilizes cached features when available for near-instant inference, with automatic fallback to Swin3D-T extraction.
Generates timeline plots, a JSON metrics summary, and an executive Markdown report.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from core.feature_extractor import FeatureExtractor
from core.video_analyzer import VideoAnalyzer
from data.taxonomy import DEFAULT_MACRO_CLASSES
from model.backbone import create_backbone
from model.head import AnomalyHead


def main():
    parser = argparse.ArgumentParser(
        description="Batch Class-Aware Anomaly Inference on Test Videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/class_aware_8fold/best_map_fold_3.pt",
        help="Path to trained AnomalyHead checkpoint (.pt)",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="Test_Videos",
        help="Directory containing test .mp4 videos",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="features_cache/test_videos",
        help="Directory containing pre-extracted feature tensors (.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/class_aware_test_analyses",
        help="Directory to save timeline plots, JSON summary, and Markdown report",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Anomaly decision threshold (0.0 to 1.0)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=20.0,
        help="Target frame rate for video sampling if feature extraction is needed (default: 20.0)",
    )
    parser.add_argument(
        "--clip-size",
        type=int,
        default=16,
        help="Number of frames per spatio-temporal clip (default: 16)",
    )
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
        default=16,
        help="Maximum clips sampled per segment to prevent memory blowups on long videos (default: 16)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="swin3d_t",
        help="3D backbone model for feature extraction (default: swin3d_t, options: swin3d_t, mvit_v2_s, r3d_18, etc.)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Clip batch size passed to GPU during feature extraction (default: 8)",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Force re-extraction of video features even if cache exists",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (cuda or cpu)",
    )

    args = parser.parse_args()

    # Verify checkpoint exists
    if not os.path.isfile(args.checkpoint):
        print(f"❌ Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    # Verify test videos directory
    video_files = sorted(glob.glob(os.path.join(args.test_dir, "*.mp4")))
    if not video_files:
        print(f"❌ No .mp4 files found in: {args.test_dir}")
        sys.exit(1)

    device = torch.device(args.device)

    # Load trained model dynamically
    head, ckpt = AnomalyHead.load_from_checkpoint(args.checkpoint, device=device)
    class_list = ckpt.get("config", {}).get("class_list", ckpt.get("class_list", DEFAULT_MACRO_CLASSES))
    val_map = ckpt.get("val_map", None)
    val_auc = ckpt.get("val_auc", None)
    epoch = ckpt.get("epoch", None)
    fold = ckpt.get("fold", None)

    # Initialize VideoAnalyzer
    analyzer = VideoAnalyzer(
        model=head,
        class_names=class_list,
        device=device,
        target_fps=args.target_fps,
    )

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
        legacy_cache = os.path.join(args.cache_dir, f"{stem}_swin3d_fps{fps_tag}.pt")
        if (
            os.path.isfile(legacy_cache)
            and backbone_tag == "swin3d_t"
            and args.clip_size == 16
            and eff_overlap == 0
        ):
            return legacy_cache
        return primary_cache

    print("\n" + "=" * 78)
    print("🎬 TOPLU SINIF-FARKINDA (CLASS-AWARE) ANOMALİ TEST ÇALIŞTIRICISI")
    print("=" * 78)
    print(f"• Model Checkpoint:  {args.checkpoint}")
    if fold is not None and epoch is not None:
        info_str = f"Fold {fold} | Epoch {epoch}"
        if val_map is not None:
            info_str += f" | Val mAP: {val_map:.4f}"
        if val_auc is not None:
            info_str += f" | Val AUC: {val_auc:.4f}"
        print(f"• Model Eğitimi:     {info_str}")
    print(f"• Omurga Modeli:     {args.backbone}")
    print(f"• Hedef Sınıflar:    {len(class_list)} adet ({', '.join(class_list)})")
    print(f"• Test Videoları:    {len(video_files)} video ({args.test_dir})")
    print(f"• Target FPS:        {args.target_fps:.1f} FPS")
    print(f"• Clip Size:         {args.clip_size} frames")
    print(f"• Overlap / Stride:  {overlap_desc}")
    print(f"• Max Clips / Seg:   {args.max_clips_per_segment}")
    print(f"• Karar Eşiği:       {args.threshold:.2f}")
    print(f"• Cihaz (Device):    {device}")
    print(f"• Çıktı Dizini:      {args.output_dir}")
    print("=" * 78 + "\n")

    # Check which videos need feature extraction
    videos_needing_extraction = []
    if not args.force_extract:
        for vpath in video_files:
            cfile = get_cache_file(vpath)
            if not os.path.isfile(cfile):
                videos_needing_extraction.append(vpath)
    else:
        videos_needing_extraction = list(video_files)

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

    results: List[Dict[str, Any]] = []
    overall_start = time.time()

    for idx, vpath in enumerate(video_files, 1):
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

        # Run Class-Aware Analysis
        analysis = analyzer.analyze_features(
            features=features,
            video_duration_sec=duration_sec,
            threshold=args.threshold,
            video_name=stem,
        )

        plot_path = os.path.join(args.output_dir, f"{stem}_class_aware_timeline.png")
        analyzer.render_timeline_plot(analysis, save_path=plot_path)
        elapsed_sec = time.time() - t0

        status_str = "🚨 ANOMALİ" if analysis["is_anomaly"] else "✅ NORMAL"
        top_cls = analysis["detected_classes"][0]["class"] if analysis["detected_classes"] else "None"

        print(
            f"[{idx:02d}/{len(video_files):02d}] {vname:<20} ({load_source}) -> {status_str} "
            f"| Tepe: %{analysis['peak_score']*100:.1f} ({top_cls}) "
            f"| Süre: {duration_sec:.1f}s ({elapsed_sec:.2f}s)"
        )

        if analysis["detected_intervals"]:
            for det in analysis["detected_intervals"]:
                print(
                    f"        • [{det['start_time']:.1f}s - {det['end_time']:.1f}s] "
                    f"{det['top_class']} (Pik: %{det['peak_score']*100:.1f})"
                )

        results.append({
            "video_name": vname,
            "stem": stem,
            "duration_sec": round(duration_sec, 2),
            "is_anomaly": analysis["is_anomaly"],
            "peak_score": round(float(analysis["peak_score"]), 4),
            "top_class": top_cls,
            "detected_classes": analysis["detected_classes"],
            "intervals": [
                {
                    "start_time": round(float(inter["start_time"]), 2),
                    "end_time": round(float(inter["end_time"]), 2),
                    "duration": round(float(inter["duration"]), 2),
                    "top_class": inter["top_class"],
                    "peak_score": round(float(inter["peak_score"]), 4),
                }
                for inter in analysis["detected_intervals"]
            ],
            "plot_path": plot_path,
        })

    total_time = time.time() - overall_start

    # Save JSON summary
    json_path = os.path.join(args.output_dir, "batch_test_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "fold": fold,
                "epoch": epoch,
                "val_map": val_map,
                "val_auc": val_auc,
                "threshold": args.threshold,
                "total_videos": len(video_files),
                "total_time_sec": round(total_time, 2),
                "results": results,
            },
            f,
            indent=2,
        )

    # Save Markdown report
    md_path = os.path.join(args.output_dir, "batch_test_report.md")
    anomaly_count = sum(1 for r in results if r["is_anomaly"])
    normal_count = len(results) - anomaly_count

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 🧪 Toplu Test_Videos Sınıf-Farkında Anomali Tespit Raporu\n\n")
        f.write(f"- **Tarih:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Test Edilen Model:** `{args.checkpoint}`\n")
        if fold is not None and epoch is not None:
            f.write(f"- **Fold / Epoch:** Fold {fold} / Epoch {epoch}\n")
            if val_map is not None:
                f.write(f"- **Model Val mAP:** `{val_map:.4f}`\n")
            if val_auc is not None:
                f.write(f"- **Model Val ROC-AUC:** `{val_auc:.4f}`\n")
        f.write(f"- **Eşik Değeri (Threshold):** `{args.threshold:.2f}`\n")
        f.write(f"- **Toplam Video:** {len(results)} adet ({anomaly_count} Anomali, {normal_count} Normal)\n")
        f.write(f"- **Toplam Test Süresi:** {total_time:.2f} saniye\n\n")

        f.write("## 1. Toplu Test Sonuçları Tablosu\n\n")
        f.write("| Video Adı | Süre (s) | Durum | Baskın Sınıf | Tepe Skor (%) | Anomali Aralıkları | Grafik |\n")
        f.write("| :--- | :---: | :---: | :--- | :---: | :--- | :---: |\n")
        for r in results:
            status_badge = "🚨 **ANOMALİ**" if r["is_anomaly"] else "✅ **NORMAL**"
            top_c = r["top_class"] if r["is_anomaly"] else "-"
            inter_strs = [
                f"`{it['start_time']}s-{it['end_time']}s` ({it['top_class']} %{it['peak_score']*100:.0f})"
                for it in r["intervals"]
            ]
            inter_disp = "<br>".join(inter_strs) if inter_strs else "-"
            plot_rel = os.path.basename(r["plot_path"])
            f.write(
                f"| `{r['video_name']}` | {r['duration_sec']:.1f}s | {status_badge} | {top_c} | "
                f"%{r['peak_score']*100:.1f} | {inter_disp} | [Grafik]({plot_rel}) |\n"
            )

        f.write("\n## 2. Tespit Edilen Olay Detayları\n\n")
        for r in results:
            if r["is_anomaly"]:
                f.write(f"### 🎬 {r['video_name']}\n")
                f.write(f"- **Toplam Süre:** {r['duration_sec']}s\n")
                f.write(f"- **Maksimum Skor:** %{r['peak_score']*100:.1f} ({r['top_class']})\n")
                f.write("- **Olay Aralıkları:**\n")
                for it in r["intervals"]:
                    f.write(f"  - `[{it['start_time']}s - {it['end_time']}s]` ({it['duration']}s): **{it['top_class']}** (Pik: %{it['peak_score']*100:.1f})\n")
                f.write(f"- **Zaman Çizelgesi Grafiği:** `results/class_aware_test_analyses/{os.path.basename(r['plot_path'])}`\n\n")

    # Terminal summary
    print("\n" + "=" * 78)
    print("📋 TOPLU TEST SONUÇ TABLOSU")
    print("=" * 78)
    print(f"{'Video Adı':<20} | {'Süre':<8} | {'Durum':<12} | {'Baskın Sınıf':<16} | {'Tepe Skor':<10} | {'Aralıklar'}")
    print("-" * 78)
    for r in results:
        status_txt = "🚨 ANOMALİ" if r["is_anomaly"] else "✅ NORMAL"
        top_c = r["top_class"] if r["is_anomaly"] else "-"
        num_it = f"{len(r['intervals'])} aralık" if r["intervals"] else "-"
        print(f"{r['video_name']:<20} | {r['duration_sec']:>6.1f}s | {status_txt:<12} | {top_c:<16} | %{r['peak_score']*100:<8.1f} | {num_it}")
    print("=" * 78)
    print(f"✓ {len(video_files)} video toplam {total_time:.2f} saniyede test edildi.")
    print(f"📊 JSON Raporu:     {json_path}")
    print(f"📄 Markdown Raporu: {md_path}")
    print(f"📈 Grafik Dizini:   {args.output_dir}\n")


if __name__ == "__main__":
    main()

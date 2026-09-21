#!/usr/bin/env python3
"""Batch Anomaly Inference Runner on all Test_Videos using UCF-Crime Binary Model."""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

# Ensure repo root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from core.feature_extractor import FeatureExtractor
from model.backbone import create_backbone
from ucf_binary_system.infer import UCFBinaryAnalyzer
from ucf_binary_system.model import BinaryAnomalyHead


def main():
    parser = argparse.ArgumentParser(description="Run UCF-Crime Binary Model on all Test_Videos")
    parser.add_argument("--test-dir", type=str, default="Test_Videos", help="Directory containing test .mp4 videos")
    parser.add_argument("--checkpoint", type=str, default="ucf_binary_system/checkpoints/best_binary_model.pt", help="Model checkpoint path")
    parser.add_argument("--target-fps", type=float, default=20.0, help="Frame rate for video clip extraction")
    parser.add_argument("--threshold", type=float, default=0.35, help="Anomaly decision threshold (default: 0.35)")
    parser.add_argument("--output-dir", type=str, default="results/ucf_binary_analyses", help="Output directory for plots")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")

    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("Please train the model first with: python ucf_binary_system/train.py")
        sys.exit(1)

    device = torch.device(args.device)
    head, ckpt = BinaryAnomalyHead.load_from_checkpoint(args.checkpoint, device=device)
    analyzer = UCFBinaryAnalyzer(model=head, device=device, target_fps=args.target_fps)

    video_files = sorted(glob.glob(os.path.join(args.test_dir, "*.mp4")))
    if not video_files:
        print(f"❌ No .mp4 videos found in {args.test_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("🎬 UCF-CRIME BINARY ANOMALY DETECTION: BATCH TEST RUN")
    print("=" * 70)
    print(f"• Model Checkpoint: {args.checkpoint} (Epoch {ckpt.get('epoch', '?')}, Val AUC: {ckpt.get('val_auc', 0.0):.4f})")
    print(f"• Test Videos:      {len(video_files)} files in '{args.test_dir}'")
    print(f"• Target FPS:       {args.target_fps:.1f} FPS")
    print(f"• Threshold:        {args.threshold:.2f}")
    print(f"• Output Directory: {args.output_dir}")
    print(f"• Device:           {device}")
    print("=" * 70 + "\n")

    # Initialize Swin3D-T backbone feature extractor
    print("• Loading Swin3D-T 32-segment video feature extractor...")
    backbone = create_backbone("swin3d_t", pretrained=True).to(device)
    extractor = FeatureExtractor(backbone=backbone, device=device, target_fps=args.target_fps)

    results = []
    overall_start = time.time()

    for i, vpath in enumerate(video_files, 1):
        vname = os.path.basename(vpath)
        stem = os.path.splitext(vname)[0]
        print(f"[{i:02d}/{len(video_files):02d}] İşleniyor: {vname}...")

        t0 = time.time()
        meta = analyzer.processor.get_video_metadata(vpath)
        features = extractor.extract_video(vpath, num_segments=32, show_progress=False)
        analysis = analyzer.analyze_features(
            features=features,
            video_duration_sec=meta["duration_sec"],
            threshold=args.threshold,
            video_name=stem,
        )

        plot_path = os.path.join(args.output_dir, f"{stem}_binary_timeline.png")
        analyzer.render_timeline_plot(analysis, save_path=plot_path)
        el_time = time.time() - t0

        status_str = "🚨 ANOMALİ" if analysis["is_anomaly"] else "✅ NORMAL"
        print(f"     -> {status_str} | Tepe Skor: %{analysis['peak_score']*100:.1f} | Süre: {meta['duration_sec']:.1f}s ({el_time:.2f}s)")
        if analysis["detected_intervals"]:
            for det in analysis["detected_intervals"]:
                print(f"        • [{det['start_time']:.1f}s - {det['end_time']:.1f}s] Pik: %{det['peak_score']*100:.1f}")

        results.append({
            "video_name": vname,
            "stem": stem,
            "duration_sec": meta["duration_sec"],
            "peak_score": analysis["peak_score"],
            "is_anomaly": analysis["is_anomaly"],
            "intervals": analysis["detected_intervals"],
            "plot_path": plot_path,
        })

    total_time = time.time() - overall_start

    print("\n" + "=" * 70)
    print(f"🎉 TÜM TEST VİDEOLARI TAMAMLANDI ({total_time:.1f} saniye)")
    print("=" * 70)

    # Save summary report markdown
    report_path = os.path.join(args.output_dir, "ucf_binary_test_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# UCF-Crime Binary Model: Test Videos Değerlendirme Raporu\n\n")
        f.write(f"- **Tarih:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Model:** `{args.checkpoint}`\n")
        f.write(f"- **Eşik:** {args.threshold:.2f}\n")
        f.write(f"- **Target FPS:** {args.target_fps:.1f}\n")
        f.write(f"- **Toplam Video:** {len(video_files)}\n\n")
        f.write("| Video | Süre | Durum | Tepe Skor (%) | Tespit Edilen Aralıklar |\n")
        f.write("| :--- | :---: | :---: | :---: | :--- |\n")
        for r in results:
            stat = "**🚨 ANOMALİ**" if r["is_anomaly"] else "✅ NORMAL"
            int_str = ", ".join([f"[{d['start_time']:.1f}s-{d['end_time']:.1f}s] (%{d['peak_score']*100:.1f})" for d in r["intervals"]]) if r["intervals"] else "-"
            f.write(f"| `{r['video_name']}` | {r['duration_sec']:.1f}s | {stat} | **%{r['peak_score']*100:.1f}** | {int_str} |\n")

    print(f"📄 Özet rapor kaydedildi: {report_path}")
    print(f"📊 Tüm çizelgeler kaydedildi: {args.output_dir}/")


if __name__ == "__main__":
    main()

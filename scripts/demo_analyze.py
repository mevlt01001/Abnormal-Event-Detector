#!/usr/bin/env python3
"""
Interactive demo script showcasing `VideoAnalyzer.analyze()` method.
Runs end-to-end inference directly on video files with sliding window clips,
produces temporal anomaly segments, and renders timeline PNG charts.
"""

from __future__ import annotations

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.analyzer import VideoAnalyzer
from utils.annotation_parser import parse_annotations, is_normal_video, get_video_classes, CLASS_NAMES


def run_analyzer_demo(
    video_path: str,
    backbone_name: str = "swin3d_t",
    threshold: float = 0.3,
    save_dir: str = "Video_Analyses",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("\n" + "=" * 75)
    print(f"🔬 INITIALIZING VideoAnalyzer({backbone_name.upper()}) ON [{device.upper()}]")
    print("=" * 75)

    ckpt_path = f"checkpoints/binary/{backbone_name}/best_loss_fold_1.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 1. Initialize analyzer with trained checkpoint
    analyzer = VideoAnalyzer(
        backbone=backbone_name,
        fc_checkpoint=ckpt_path,
    ).to(device)

    # 2. Extract Ground Truth info from filename
    is_normal = is_normal_video(video_path)
    gt_classes = [CLASS_NAMES.get(c, c) for c in get_video_classes(video_path)]
    gt_str = "NORMAL" if is_normal else f"ANOMALOUS ({', '.join(gt_classes)})"

    print(f"🎬 Video:        {os.path.basename(video_path)}")
    print(f"🏷️  Ground Truth: {gt_str}")
    print(f"⚙️  Threshold:    {threshold}")
    print(f"🚀 Calling analyzer.analyze(video_path, threshold={threshold}, save_graph=True)...")

    # 3. Call the core .analyze() method
    segments = analyzer.analyze(
        video_path=video_path,
        batch_size=4,
        threshold=threshold,
        tolerance_sec=3.0,
        padding_sec=2.0,
        save_graph=True,
        save_dir=save_dir,
    )

    # 4. Display Results
    basename = os.path.splitext(os.path.basename(video_path))[0]
    chart_file = os.path.join(save_dir, f"{basename}_anomaly_timeline.png")

    print("\n" + "-" * 75)
    print(f"📋 ANALYZER RESULTS ({len(segments)} anomaly segment(s) detected):")
    print("-" * 75)

    if not segments:
        print("🟢 No anomalies detected! The video is classified as completely NORMAL.")
    else:
        print(f"🔴 ANOMALIES DETECTED! Violence intervals identified:")
        for idx, seg in enumerate(segments, 1):
            s_t = seg["start_time"]
            e_t = seg["end_time"]
            dur = seg["duration"]
            sc = seg["score"]
            print(f"   [{idx}] {s_t:05.1f}s - {e_t:05.1f}s  (Süre: {dur:4.1f}s | Tepe Güven Skoru: {sc*100:5.1f}%)")

    if os.path.exists(chart_file):
        print(f"\n📊 Görsel Zaman Çizelgesi Grafiği Kaydedildi:")
        print(f"   👉 {chart_file}")
    print("=" * 75 + "\n")
    return segments


def main():
    parser = argparse.ArgumentParser(description="Demo of VideoAnalyzer.analyze()")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to .mp4 video file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="swin3d_t",
        choices=["swin3d_t", "mvit_v1_b", "s3d", "r3d_18", "mc3_18", "r2plus1d_18"],
        help="Model backbone to use",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Anomaly detection sensitivity threshold (default 0.3)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="Video_Analyses",
        help="Output folder for timeline plots",
    )
    args = parser.parse_args()

    if args.video:
        run_analyzer_demo(args.video, backbone_name=args.model, threshold=args.threshold, save_dir=args.save_dir)
    else:
        # Run comparison demo on 1 Violent video and 1 Normal video
        print("\n⚡ RUNNING COMPARISON DEMO WITH VideoAnalyzer.analyze()...\n")
        vids = [
            "XD-Violence-test-videos/Bad.Boys.1995__#01-11-55_01-12-40_label_G-B2-B6.mp4",
            "XD-Violence-test-videos/About.Time.2013__#00-23-50_00-24-31_label_A.mp4",
        ]
        for v in vids:
            run_analyzer_demo(v, backbone_name=args.model, threshold=args.threshold, save_dir=args.save_dir)


if __name__ == "__main__":
    main()

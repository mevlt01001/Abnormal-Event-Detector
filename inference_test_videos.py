#!/usr/bin/env python3
"""Batch Video Anomaly Inference Script for Test_Videos across Trained Models.

Evaluates trained AnomalyDetector checkpoints (from checkpoints/) on all .mp4 videos
in the Test_Videos directory.
- Features are extracted once using Swin3D-T and cached in features_cache/test_videos/.
- For each model, generates:
    1. Timeline plots in results/{model_name}/plots/
    2. Anomaly video clips in results/{model_name}/clips/{video_stem}/
    3. JSON summary in results/{model_name}/test_videos_summary.json
    4. Executive Markdown report in results/{model_name}/test_videos_report.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from model.analyzer import extract_anomaly_segments, plot_anomaly_timeline
from model.backbone import create_backbone
from model.head import AnomalyHead
from utils.feature_extractor import FeatureExtractor
from utils.video_processor import VideoProcessor, save_segment_clips


ALL_MODEL_NAMES = [
    "UCF_binary",
    "UCF_multiclass",
    "XDV_binary",
    "XDV_multiclass",
    "UCF_XDV_Combined_binary",
    "UCF_XDV_Combined_multiclass",
    "UCF_XDV_Combined_exact_match",
]


def extract_and_cache_features(
    video_files: List[str],
    cache_dir: str = "features_cache/test_videos",
    backbone_name: str = "mvit_v2_s",
    target_fps: float = 20.0,
    clip_size: int = 16,
    overlap_ratio: float = 0.50,
    max_clips_per_segment: int = 24,
    batch_size: int = 8,
    num_segments: int = 32,
    device: str = "cuda",
    force_extract: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Extracts 3D video features once and caches them to disk matching Kaggle extraction.

    Returns:
        Mapping from video_stem to dict containing:
            'feats': Tensor of shape [num_segments, feature_dim]
            'duration_sec': float
            'fps': float
            'video_path': str
    """
    os.makedirs(cache_dir, exist_ok=True)
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    cached_features: Dict[str, Dict[str, Any]] = {}
    videos_to_extract: List[str] = []

    for vpath in video_files:
        stem = os.path.splitext(os.path.basename(vpath))[0]
        cpath = os.path.join(cache_dir, f"{stem}_{backbone_name}_fps{int(target_fps)}_c{clip_size}.pt")

        if os.path.isfile(cpath) and not force_extract:
            data = torch.load(cpath, map_location="cpu", weights_only=False)
            cached_features[stem] = {
                "feats": data["feats"],
                "duration_sec": data.get("duration_sec", 60.0),
                "fps": data.get("fps", target_fps),
                "video_path": vpath,
            }
        else:
            videos_to_extract.append(vpath)

    if videos_to_extract:
        print(f"\n📦 {len(videos_to_extract)}/{len(video_files)} video için öznitelik çıkarılıyor ({backbone_name}, {device})...")
        print(f"• Parametreler: fps={target_fps}, clip_size={clip_size}, overlap_ratio={overlap_ratio}, max_clips={max_clips_per_segment}")
        backbone = create_backbone(backbone_name, pretrained=True).to(device_obj).eval()
        extractor = FeatureExtractor(
            backbone=backbone,
            device=device_obj,
            target_fps=target_fps,
            clip_size=clip_size,
            overlap_ratio=overlap_ratio,
            max_clips_per_segment=max_clips_per_segment,
            batch_size=batch_size,
        )

        for vpath in tqdm(videos_to_extract, desc="Öznitelik Çıkarımı"):
            stem = os.path.splitext(os.path.basename(vpath))[0]
            cpath = os.path.join(cache_dir, f"{stem}_{backbone_name}_fps{int(target_fps)}_c{clip_size}.pt")

            meta = extractor.processor.get_video_metadata(vpath)
            duration_sec = meta["duration_sec"]
            fps = meta["original_fps"]

            feats = extractor.extract_video(vpath, num_segments=num_segments, show_progress=False)

            torch.save(
                {
                    "feats": feats.cpu(),
                    "duration_sec": duration_sec,
                    "fps": fps,
                    "stem": stem,
                    "backbone": backbone_name,
                },
                cpath,
            )

            cached_features[stem] = {
                "feats": feats.cpu(),
                "duration_sec": duration_sec,
                "fps": fps,
                "video_path": vpath,
            }

        del extractor, backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"⚡ Tüm {len(video_files)} test videosunun öznitelikleri önbellekte hazır! ({cache_dir})")

    return cached_features


def get_model_optimal_threshold(model_name: str, default: float = 0.30) -> float:
    """Retrieves the optimal threshold from evaluation_summary.json if present."""
    summary_path = os.path.join("results", model_name, "evaluation_summary.json")
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
                if "best_threshold" in summary and summary["best_threshold"] is not None:
                    return float(summary["best_threshold"])
        except Exception:
            pass
    return default


def run_inference_for_model(
    model_name: str,
    checkpoint_path: str,
    cached_features: Dict[str, Dict[str, Any]],
    output_base_dir: str = "results",
    threshold: Optional[float] = None,
    tolerance_sec: float = 1.5,
    padding_sec: float = 1.0,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Runs inference with a single model checkpoint on all cached video features."""
    if not os.path.isfile(checkpoint_path):
        print(f"⚠️ Checkpoint bulunamadı, atlanıyor: {checkpoint_path}")
        return {}

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    head, ckpt = AnomalyHead.load_from_checkpoint(checkpoint_path, device=device_obj)
    head.eval()

    num_classes = getattr(head, "num_classes", 1)
    ds_meta = ckpt.get("dataset_metadata", {})
    class_names = ds_meta.get("dataset_classes", None)

    if not class_names or len(class_names) != num_classes:
        if num_classes == 1:
            class_names = ["Anomaly"]
        else:
            class_names = [f"Class_{i}" for i in range(num_classes)]

    # Determine threshold
    optimal_tau = get_model_optimal_threshold(model_name, default=0.30)
    tau = float(threshold) if threshold is not None else optimal_tau

    model_dir = os.path.join(output_base_dir, model_name)
    plots_dir = os.path.join(model_dir, "plots")
    clips_dir = os.path.join(model_dir, "clips")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(clips_dir, exist_ok=True)

    print(f"\n{'='*75}")
    print(f"🔍 MODEL İNCELEMESİ: {model_name}")
    print(f"• Checkpoint:  {checkpoint_path}")
    print(f"• Sınıf Sayısı: {num_classes} ({', '.join(class_names[:5])}{'...' if len(class_names)>5 else ''})")
    print(f"• Karar Eşiği: tau = {tau:.2f} (Optimal referans: {optimal_tau:.2f})")
    print(f"• Çıktı Dizini: {model_dir}")
    print(f"{'='*75}")

    video_results: List[Dict[str, Any]] = []

    for stem, vinfo in cached_features.items():
        vpath = vinfo["video_path"]
        vname = os.path.basename(vpath)
        duration_sec = vinfo["duration_sec"]
        feats = vinfo["feats"]  # [T, D]

        # 1. Forward Pass through AnomalyHead
        with torch.no_grad():
            inp = feats.unsqueeze(0).to(device_obj)  # [1, T, D]
            logits = head(inp)                      # [1, T, C]
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # [T, C] or [T, 1]

        # 2. Segment Extraction
        segments, scores_smooth, scores_raw = extract_anomaly_segments(
            scores=probs,
            video_seconds=duration_sec,
            threshold=tau,
            tolerance_sec=tolerance_sec,
            padding_sec=padding_sec,
            class_names=class_names if num_classes > 1 else None,
        )

        peak_score = float(scores_smooth.max().item()) if len(scores_smooth) > 0 else 0.0
        is_anomaly = bool(len(segments) > 0)

        # 3. Timeline Plot Rendering
        plot_path = os.path.join(plots_dir, f"{stem}_timeline.png")
        plot_anomaly_timeline(
            scores_smooth=scores_smooth,
            scores_raw=scores_raw,
            segments=segments,
            video_seconds=duration_sec,
            threshold=tau,
            save_path=plot_path,
            video_name=f"{stem} — {model_name}",
            class_names=class_names if num_classes > 1 else ["Anomaly"],
        )

        # 4. Anomaly Clips Extraction
        clip_paths: List[str] = []
        if is_anomaly:
            video_clip_dir = os.path.join(clips_dir, stem)
            saved_clips = save_segment_clips(
                video_path=vpath,
                segments=segments,
                save_dir=video_clip_dir,
                prefix=stem,
            )
            clip_paths = [p for p in saved_clips if p is not None]

        # Top detected category
        top_detected_class = "Normal"
        if is_anomaly:
            if num_classes == 1:
                top_detected_class = "Anomaly"
            else:
                top_detected_class = segments[0].get("top_class", "Anomaly")

        status_icon = "🚨 ANOMALİ" if is_anomaly else "✅ NORMAL"
        print(
            f"  [{status_icon}] {vname:<18} | Süre: {duration_sec:5.1f}s | "
            f"Pik Skor: %{peak_score*100:5.1f} | "
            f"Tespit: {len(segments)} segment ({top_detected_class}) | "
            f"Klip: {len(clip_paths)} adet"
        )

        video_results.append({
            "video_name": vname,
            "stem": stem,
            "duration_sec": round(duration_sec, 2),
            "is_anomaly": is_anomaly,
            "peak_score": round(peak_score, 4),
            "top_class": top_detected_class,
            "segments": segments,
            "plot_path": os.path.relpath(plot_path, output_base_dir),
            "clips": [os.path.relpath(p, output_base_dir) for p in clip_paths],
        })

    # 5. Save JSON Summary
    summary_data = {
        "model_name": model_name,
        "checkpoint": checkpoint_path,
        "num_classes": num_classes,
        "classes": class_names,
        "threshold": tau,
        "tolerance_sec": tolerance_sec,
        "padding_sec": padding_sec,
        "total_videos": len(video_results),
        "anomaly_videos": sum(1 for v in video_results if v["is_anomaly"]),
        "normal_videos": sum(1 for v in video_results if not v["is_anomaly"]),
        "total_clips_generated": sum(len(v["clips"]) for v in video_results),
        "results": video_results,
    }

    json_path = os.path.join(model_dir, "test_videos_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    # 6. Save Executive Markdown Report
    md_path = os.path.join(model_dir, "test_videos_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# 🎬 Test_Videos Çıkarım Raporu: `{model_name}`\n\n")
        f.write(f"- **Model Checkpoint**: `{checkpoint_path}`\n")
        f.write(f"- **Model Tipi**: {'Binary (1 Skor)' if num_classes == 1 else f'Çok Sınıflı ({num_classes} Sınıf)'}\n")
        f.write(f"- **Uygulanan Eşik (tau)**: **`{tau:.2f}`**\n")
        f.write(f"- **Zaman Parametreleri**: `tolerance_sec={tolerance_sec}s`, `padding_sec={padding_sec}s`\n")
        f.write(f"- **Toplam Test Videosu**: {len(video_results)}\n")
        f.write(f"- **Anomali Tespit Edilen**: **{summary_data['anomaly_videos']}** video\n")
        f.write(f"- **Üretilen Anomali Klibi**: **{summary_data['total_clips_generated']}** adet MP4\n\n")

        f.write("## 📊 Video Çıkarım Özet Tablosu\n\n")
        f.write("| Video | Durum | Süre | Pik Skor | Tespit Sayısı | Başlıca Sınıf | Üretilen Klipler | Grafik |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for r in video_results:
            icon = "🚨 **Anomali**" if r["is_anomaly"] else "✅ Normal"
            clips_str = f"{len(r['clips'])} klip" if r["clips"] else "—"
            plot_link = f"[Grafik](plots/{r['stem']}_timeline.png)"
            f.write(
                f"| `{r['video_name']}` | {icon} | {r['duration_sec']:.1f}s | "
                f"%{r['peak_score']*100:.1f} | {len(r['segments'])} | "
                f"`{r['top_class']}` | {clips_str} | {plot_link} |\n"
            )

        f.write("\n## 🔍 Tespit Detayları ve Anomali Aralıkları\n\n")
        for r in video_results:
            if not r["is_anomaly"]:
                continue
            f.write(f"### 📹 `{r['video_name']}`\n")
            f.write(f"- **Pik Anomali Güveni**: %{r['peak_score']*100:.1f}\n")
            f.write(f"- **Zaman Grafiği**: ![{r['stem']}](plots/{r['stem']}_timeline.png)\n")
            f.write("- **Tespit Edilen Olay Segmentleri**:\n")
            for idx, seg in enumerate(r["segments"], start=1):
                f.write(
                    f"  {idx}. `[{seg['start_time']:.1f}s - {seg['end_time']:.1f}s]` "
                    f"({seg['duration']:.1f}s) — **{seg.get('top_class', 'Anomaly')}** "
                    f"(Skor: %{seg['score']*100:.1f})\n"
                )
            if r["clips"]:
                f.write("- **Kaydedilen Klipler**:\n")
                for c in r["clips"]:
                    f.write(f"  - `{c}`\n")
            f.write("\n---\n")

    print(f"📄 Raporlar kaydedildi: {json_path} ve {md_path}")
    return summary_data


def main():
    parser = argparse.ArgumentParser(
        description="Batch Video Anomaly Inference for Test_Videos on all trained models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="Test_Videos",
        help="Directory containing .mp4 test videos",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="features_cache/test_videos",
        help="Directory to save/load pre-extracted features",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Base results directory where per-model folders exist",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="List of model names to run inference for, or 'all' for all 7 models",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Explicit decision threshold (if None, model optimal threshold from sweep is used)",
    )
    parser.add_argument(
        "--tolerance-sec",
        type=float,
        default=1.5,
        help="Gap tolerance for merging adjacent anomaly moments (seconds)",
    )
    parser.add_argument(
        "--padding-sec",
        type=float,
        default=1.0,
        help="Context padding around detected anomaly segment (seconds)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda or cpu)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="s3d",
        help="Feature extraction backbone model (default: mvit_v2_s)",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Force re-extraction of video features even if cache exists",
    )

    args = parser.parse_args()

    # Find test video files
    video_files = sorted(glob.glob(os.path.join(args.test_dir, "*.mp4")))
    if not video_files:
        print(f"❌ '{args.test_dir}' klasöründe .mp4 dosyası bulunamadı!")
        sys.exit(1)

    print(f"🎬 Bulunan test videoları: {len(video_files)} adet ({args.test_dir})")

    # Determine which models to evaluate
    if "all" in args.models:
        target_models = ALL_MODEL_NAMES
    else:
        target_models = args.models

    # Step 1: Pre-extract and cache features once for all test videos
    cached_features = extract_and_cache_features(
        video_files=video_files,
        cache_dir=args.cache_dir,
        backbone_name=args.backbone,
        device=args.device,
        force_extract=args.force_extract,
    )

    # Step 2: Run inference for each target model
    start_time = time.time()
    completed_models = 0

    for m_name in target_models:
        ckpt_path = os.path.join("checkpoints", m_name, "best_model.pt")
        if not os.path.isfile(ckpt_path):
            print(f"⚠️ Model checkpoint bulunamadı: {ckpt_path}, atlanıyor...")
            continue

        run_inference_for_model(
            model_name=m_name,
            checkpoint_path=ckpt_path,
            cached_features=cached_features,
            output_base_dir=args.output_dir,
            threshold=args.threshold,
            tolerance_sec=args.tolerance_sec,
            padding_sec=args.padding_sec,
            device=args.device,
        )
        completed_models += 1

    total_sec = time.time() - start_time
    print(f"\n{'='*75}")
    print(f"🎉 Toplu Çıkarım Tamamlandı! {completed_models} model işlendi (Toplam süre: {total_sec:.1f}s)")
    print(f"📁 Çıktılar: results/<model_name>/plots/ ve results/<model_name>/clips/")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()

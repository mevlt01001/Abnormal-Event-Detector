#!/usr/bin/env python3
"""Master Pipeline Orchestrator for Multi-Model Video Anomaly Detection on XD-Violence.

Runs the full end-to-end benchmark across 6 3D backbone models:
1. Feature Extraction (ordered fastest to slowest: s3d -> swin3d_t -> mvit_v1_b -> r3d_18 -> mc3_18 -> r2plus1d_18)
2. Binary MIL Training (Sultani Ranking Loss)
3. Multi-Class MIL Training (Top-K Multi-Label BCE)
4. Comprehensive Evaluation (Frame-level AUC, tIoU, Video-level F1, Per-class AP, mAP)
5. Comparison Reporting
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Add workspace to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.model import FC_head, model_feature_dims
from training.trainer import MILTrainer
from training.multiclass_trainer import MultiClassMILTrainer
from scripts.evaluator import evaluate_model_pipeline

# 6 models ordered by extraction throughput (fastest to slowest)
DEFAULT_MODELS: List[str] = [
    "s3d",
    "swin3d_t",
    "mvit_v1_b",
    "r3d_18",
    "mc3_18",
    "r2plus1d_18",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run complete Multi-Model Anomaly Detection Pipeline"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="List of backbone model names to run",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="/home/n3uron/Videos/XD-Violence-test-videos",
        help="Directory containing XD-Violence test videos",
    )
    parser.add_argument(
        "--ann_file",
        type=str,
        default="XDviolanece-test-annotations.txt",
        help="Path to XD-Violence annotations file",
    )
    parser.add_argument(
        "--features_root",
        type=str,
        default="data/features",
        help="Root directory for extracted features",
    )
    parser.add_argument(
        "--checkpoints_root",
        type=str,
        default="checkpoints",
        help="Root directory for saved checkpoints",
    )
    parser.add_argument(
        "--runs_root",
        type=str,
        default="runs",
        help="Root directory for TensorBoard logs",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="results",
        help="Directory to save evaluation results and summaries",
    )
    parser.add_argument(
        "--epochs_binary",
        type=int,
        default=15,
        help="Number of epochs for binary MIL training",
    )
    parser.add_argument(
        "--epochs_multiclass",
        type=int,
        default=15,
        help="Number of epochs for multi-class MIL training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--k_fold",
        type=int,
        default=5,
        help="Number of cross validation folds",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers for feature extraction",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--skip_extraction",
        action="store_true",
        help="Skip feature extraction if already extracted",
    )
    parser.add_argument(
        "--skip_binary",
        action="store_true",
        help="Skip binary training phase",
    )
    parser.add_argument(
        "--skip_multiclass",
        action="store_true",
        help="Skip multi-class training phase",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation phase",
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=-1,
        help="Number of videos to process per model (-1 for all 800)",
    )
    return parser.parse_args()


def is_extraction_complete(features_dir: str, target_count: int = 800) -> bool:
    """Checks whether extraction has already been completed for this model."""
    norm_dir = os.path.join(features_dir, "normal")
    anom_dir = os.path.join(features_dir, "anomal")
    if not (os.path.isdir(norm_dir) and os.path.isdir(anom_dir)):
        return False

    norm_pts = len([f for f in os.listdir(norm_dir) if f.endswith(".pt")])
    anom_pts = len([f for f in os.listdir(anom_dir) if f.endswith(".pt")])
    total = norm_pts + anom_pts
    return total >= target_count and norm_pts > 0 and anom_pts > 0


def run_extraction_for_model(
    model_name: str,
    video_dir: str,
    output_dir: str,
    num_workers: int = 4,
    device: str = "cuda",
    num_videos: int = -1,
) -> float:
    """Executes extract_features.py for a single model backbone and returns elapsed seconds."""
    from scripts.extract_features import main as extract_main

    print(f"\n{'#'*60}")
    print(f"STAGE 1: FEATURE EXTRACTION FOR [{model_name}]")
    print(f"{'#'*60}")

    start_t = time.time()
    sys_argv_backup = list(sys.argv)
    try:
        sys.argv = [
            "extract_features.py",
            "--model_name", model_name,
            "--video_dir", video_dir,
            "--output_dir", output_dir,
            "--num_workers", str(num_workers),
            "--device", device,
            "--num_videos", str(num_videos),
        ]
        extract_main()
    finally:
        sys.argv = sys_argv_backup

    elapsed = time.time() - start_t
    print(f"Feature extraction for {model_name} finished in {elapsed:.1f}s ({elapsed/3600:.2f}h)")
    return elapsed


def run_pipeline():
    args = parse_args()
    print("=" * 70)
    print("STARTING FULL ANOMALY DETECTION PIPELINE")
    print(f"Models: {args.models}")
    print(f"Device: {args.device}")
    print(f"Video Directory: {args.video_dir}")
    print(f"Features Root: {args.features_root}")
    print(f"Checkpoints Root: {args.checkpoints_root}")
    print("=" * 70)

    os.makedirs(args.features_root, exist_ok=True)
    os.makedirs(args.checkpoints_root, exist_ok=True)
    os.makedirs(args.runs_root, exist_ok=True)
    os.makedirs(args.results_root, exist_ok=True)

    pipeline_summary: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {},
        "rankings": {},
    }

    total_pipeline_start = time.time()

    for model_idx, model_name in enumerate(args.models, 1):
        print(f"\n\n{'*'*70}")
        print(f"PROCESSING MODEL {model_idx}/{len(args.models)}: {model_name.upper()}")
        print(f"{'*'*70}")

        model_start_time = time.time()
        model_feat_dir = os.path.join(args.features_root, model_name)
        norm_dir = os.path.join(model_feat_dir, "normal")
        anom_dir = os.path.join(model_feat_dir, "anomal")
        feat_dim = model_feature_dims.get(model_name, 512)

        model_results: Dict[str, Any] = {
            "model_name": model_name,
            "feature_dim": feat_dim,
            "extraction_time_s": 0.0,
            "binary_training": {},
            "multiclass_training": {},
            "evaluation": {},
        }

        # -------------------------------------------------------------
        # STEP 1: Feature Extraction
        # -------------------------------------------------------------
        target_vids = 800 if args.num_videos <= 0 else args.num_videos
        if not args.skip_extraction:
            if is_extraction_complete(model_feat_dir, target_vids):
                print(f"[CACHE] Features for {model_name} already extracted ({target_vids} videos). Skipping extraction.")
            else:
                ext_time = run_extraction_for_model(
                    model_name=model_name,
                    video_dir=args.video_dir,
                    output_dir=model_feat_dir,
                    num_workers=args.num_workers,
                    device=args.device,
                    num_videos=args.num_videos,
                )
                model_results["extraction_time_s"] = ext_time
        else:
            print(f"[SKIP] Feature extraction skipped by user flag.")

        # Verify features exist
        if not (os.path.isdir(norm_dir) and os.path.isdir(anom_dir)):
            print(f"[WARN] Feature directories for {model_name} missing. Skipping training/eval.")
            continue

        # -------------------------------------------------------------
        # STEP 2: Binary MIL Training
        # -------------------------------------------------------------
        binary_save_dir = os.path.join(args.checkpoints_root, "binary", model_name)
        binary_log_dir = os.path.join(args.runs_root, "binary", model_name)
        binary_head = FC_head(in_features=feat_dim, num_classes=1, use_sigmoid=True)

        if not args.skip_binary:
            print(f"\n{'#'*60}")
            print(f"STAGE 2: BINARY MIL TRAINING FOR [{model_name}]")
            print(f"{'#'*60}")

            trainer_binary = MILTrainer(
                model=binary_head,
                anomal_dir=anom_dir,
                normal_dir=norm_dir,
                epochs=args.epochs_binary,
                batch_size=args.batch_size,
                k_fold=args.k_fold,
                save_dir=binary_save_dir,
                log_dir=binary_log_dir,
                device=args.device,
            )
            binary_res = trainer_binary.train()
            model_results["binary_training"] = binary_res
        else:
            print(f"[SKIP] Binary training skipped by user flag.")

        # -------------------------------------------------------------
        # STEP 3: Multi-Class MIL Training
        # -------------------------------------------------------------
        multi_save_dir = os.path.join(args.checkpoints_root, "multiclass", model_name)
        multi_log_dir = os.path.join(args.runs_root, "multiclass", model_name)
        multi_head = FC_head(in_features=feat_dim, num_classes=6, use_sigmoid=False)

        if not args.skip_multiclass:
            print(f"\n{'#'*60}")
            print(f"STAGE 3: MULTI-CLASS MIL TRAINING FOR [{model_name}]")
            print(f"{'#'*60}")

            trainer_multi = MultiClassMILTrainer(
                model=multi_head,
                features_dir=model_feat_dir,
                epochs=args.epochs_multiclass,
                batch_size=args.batch_size,
                k_fold=args.k_fold,
                save_dir=multi_save_dir,
                log_dir=multi_log_dir,
                device=args.device,
            )
            multi_res = trainer_multi.train()
            model_results["multiclass_training"] = multi_res
        else:
            print(f"[SKIP] Multi-class training skipped by user flag.")

        # -------------------------------------------------------------
        # STEP 4: Evaluation
        # -------------------------------------------------------------
        if not args.skip_eval:
            print(f"\n{'#'*60}")
            print(f"STAGE 4: EVALUATION FOR [{model_name}]")
            print(f"{'#'*60}")

            # Load best checkpoints if available
            bin_best = os.path.join(binary_save_dir, "best_loss_fold_1.pt")
            if os.path.isfile(bin_best):
                ckpt = torch.load(bin_best, map_location=args.device)
                s_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
                binary_head.load_state_dict(s_dict)

            mul_best = os.path.join(multi_save_dir, "best_loss_fold_1.pt")
            if os.path.isfile(mul_best):
                ckpt = torch.load(mul_best, map_location=args.device)
                s_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
                multi_head.load_state_dict(s_dict)


            eval_res = evaluate_model_pipeline(
                binary_model=binary_head,
                multiclass_model=multi_head,
                features_dir=model_feat_dir,
                ann_file=args.ann_file,
                video_dir=args.video_dir,
                device=args.device,
            )
            model_results["evaluation"] = eval_res

            # Print concise metrics
            bin_eval = eval_res.get("binary", {})
            mul_eval = eval_res.get("multiclass", {})
            print(f"\n--- {model_name.upper()} Evaluation Results ---")
            print(f"Frame AUC-ROC:      {bin_eval.get('frame_auc', 0.0):.4f}")
            print(f"Frame AP:           {bin_eval.get('frame_ap', 0.0):.4f}")
            print(f"Video AUC:          {bin_eval.get('video_auc', 0.0):.4f}")
            print(f"Video F1:           {bin_eval.get('video_f1', 0.0):.4f}")
            print(f"tIoU @ 0.5:         {bin_eval.get('tiou_recalls', {}).get('tIoU_0.5', 0.0):.4f}")
            print(f"Multi-Class mAP:    {mul_eval.get('mAP', 0.0):.4f}")
            print(f"Exact Match Ratio:  {mul_eval.get('exact_match_ratio', 0.0):.4f}")

            # Save individual model eval file
            eval_out_path = os.path.join(args.results_root, f"{model_name}_eval.json")
            with open(eval_out_path, "w") as f:
                json.dump(model_results, f, indent=2)

        model_elapsed = time.time() - model_start_time
        model_results["total_elapsed_s"] = model_elapsed
        pipeline_summary["models"][model_name] = model_results

        # Progress update
        print(f"\nModel {model_name} completed in {model_elapsed/60:.1f} minutes.")

    # -------------------------------------------------------------
    # STEP 5: Aggregate Benchmark Summary & Rankings
    # -------------------------------------------------------------
    total_time_h = (time.time() - total_pipeline_start) / 3600
    print(f"\n\n{'='*70}")
    print(f"ALL MODELS PROCESSED! Total time: {total_time_h:.2f} hours")
    print(f"{'='*70}")

    # Compute rankings based on Frame AUC and Multi-Class mAP
    rankings = []
    for m_name, m_data in pipeline_summary["models"].items():
        ev = m_data.get("evaluation", {})
        frame_auc = ev.get("binary", {}).get("frame_auc", 0.0)
        m_ap = ev.get("multiclass", {}).get("mAP", 0.0)
        tiou_5 = ev.get("binary", {}).get("tiou_recalls", {}).get("tIoU_0.5", 0.0)
        rankings.append({
            "model": m_name,
            "frame_auc": frame_auc,
            "multiclass_mAP": m_ap,
            "tiou_0.5": tiou_5,
            "extraction_s": m_data.get("extraction_time_s", 0.0),
        })

    rankings.sort(key=lambda x: x["frame_auc"], reverse=True)
    pipeline_summary["rankings"] = rankings

    summary_file = os.path.join(args.results_root, "benchmark_summary.json")
    with open(summary_file, "w") as f:
        json.dump(pipeline_summary, f, indent=2)

    print(f"Benchmark summary saved to {summary_file}")
    print("\n--- FINAL BENCHMARK RANKINGS (by Frame AUC) ---")
    print(f"{'Rank':<5} {'Model':<15} {'Frame AUC':<12} {'Multi mAP':<12} {'tIoU@0.5':<10}")
    print("-" * 55)
    for i, r in enumerate(rankings, 1):
        print(f"{i:<5} {r['model']:<15} {r['frame_auc']:<12.4f} {r['multiclass_mAP']:<12.4f} {r['tiou_0.5']:<10.4f}")


if __name__ == "__main__":
    run_pipeline()

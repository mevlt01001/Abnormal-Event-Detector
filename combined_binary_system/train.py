#!/usr/bin/env python3
"""CLI Training Script for Combined UCF-Crime & XD-Violence Binary Anomaly Detector."""

from __future__ import annotations

import argparse
import os
import sys

# Ensure repo root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from combined_binary_system.model import DeepBinaryAnomalyHead
from combined_binary_system.trainer import CombinedBinaryTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Combined UCF & XD-Violence Binary Video Anomaly Detector")
    parser.add_argument(
        "--features-root",
        type=str,
        default="data/unified_features",
        help="Path to directory containing feature folders",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds (1 for single split)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs per fold")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.001, help="Weight decay L2 penalty")
    parser.add_argument("--noise-std", type=float, default=0.075, help="Training noise scale")
    parser.add_argument("--k-top", type=int, default=3, help="Top-k anomalous segments for pooling")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="Validation partition ratio if folds=1")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="combined_binary_system/checkpoints",
        help="Output checkpoint directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cuda/cpu)",
    )

    args = parser.parse_args()

    # Discover feature dimension from first sample
    feat_dim = 768
    from combined_binary_system.dataset import scan_combined_features, load_feature_tensor
    samples = scan_combined_features(args.features_root)
    if samples:
        sample_tensor = load_feature_tensor(samples[0][0])
        feat_dim = sample_tensor.shape[-1]

    # Deep Pyramidal Binary Head: in_features -> 512 -> 256 -> 128 -> 1 (with Sigmoid output)
    model = DeepBinaryAnomalyHead(
        in_features=feat_dim,
        hidden_dims=(512, 256, 128),
        dropout_rates=(0.50, 0.40, 0.30),
        noise_std=args.noise_std,
        enable_noise=True,
    )

    trainer = CombinedBinaryTrainer(
        model=model,
        features_root=args.features_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        k_top=args.k_top,
        k_fold=args.folds,
        val_ratio=args.val_ratio,
        save_dir=args.save_dir,
        device=args.device,
    )

    results = trainer.train()

    # Generate Markdown Report
    os.makedirs("results", exist_ok=True)
    report_file = os.path.join("results", f"{args.folds}fold_binary_training_report.md" if args.folds > 1 else "binary_training_report.md")
    import time
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# {args.folds}-Fold Binary Video Anomali Tespiti Eğitim Raporu\n\n")
        f.write(f"- **Tarih:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Model Mimarisi:** Deep Pyramidal Binary Anomaly Head ({feat_dim} -> 512 -> 256 -> 128 -> 1, Sigmoid)\n")
        f.write(f"- **Veri Seti:** `{args.features_root}` (Toplam: {len(samples)} video)\n")
        f.write(f"- **Fold Sayısı:** {args.folds}\n")
        f.write(f"- **Epoch / Fold:** {args.epochs}\n")
        f.write(f"- **Batch Size:** {args.batch_size}\n")
        f.write(f"- **Toplam Süre:** {results.get('total_duration_sec', 0.0)} saniye\n\n")

        if args.folds > 1:
            f.write("## 1. Genel Performans Özeti (5-Fold Stratified Cross-Validation)\n\n")
            f.write("| Metrik | Ortalama (Mean) | Standart Sapma (Std) |\n")
            f.write("| :--- | :---: | :---: |\n")
            f.write(f"| **Doğrulama ROC-AUC** | **{results['mean_best_auc']:.4f}** | ± {results['std_best_auc']:.4f} |\n")
            f.write(f"| **Doğrulama PR-AUC (Average Precision)** | **{results['mean_best_pr_auc']:.4f}** | ± {results['std_best_pr_auc']:.4f} |\n")
            f.write(f"| **Doğrulama Kaybı (Val Loss)** | **{results['mean_best_loss']:.4f}** | ± {results['std_best_loss']:.4f} |\n\n")

            f.write("## 2. Fold Bazlı Detaylar\n\n")
            f.write("| Fold | En İyi ROC-AUC | En İyi PR-AUC | En İyi Val Loss | En İyi Epoch |\n")
            f.write("| :---: | :---: | :---: | :---: | :---: |\n")
            for fr in results.get("fold_results", []):
                f.write(f"| Fold {fr['fold']} | {fr['best_val_auc']:.4f} | {fr['best_val_pr_auc']:.4f} | {fr['best_val_loss']:.4f} | Epoch {fr['best_epoch']} |\n")

    print(f"📄 Binary eğitim raporu kaydedildi: {report_file}")


if __name__ == "__main__":
    main()

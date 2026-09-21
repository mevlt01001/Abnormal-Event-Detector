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
        default="swin3d_t_extracted_features",
        help="Path to directory containing ucf_* and xdv_* feature folders",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.001, help="Weight decay L2 penalty")
    parser.add_argument("--noise-std", type=float, default=0.075, help="Training noise scale")
    parser.add_argument("--k-top", type=int, default=3, help="Top-k anomalous segments for pooling")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation partition ratio")
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

    # Deep Pyramidal Binary Head: 768 -> 512 -> 256 -> 128 -> 1
    # Dropout: 0.70 -> 0.60 -> 0.50
    # Noise: noise_std = 0.10
    model = DeepBinaryAnomalyHead(
        in_features=768,
        hidden_dims=(512, 256, 128),
        dropout_rates=(0.5, 0.4, 0.3),
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
        val_ratio=args.val_ratio,
        save_dir=args.save_dir,
        device=args.device,
    )

    trainer.train()


if __name__ == "__main__":
    main()

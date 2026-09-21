#!/usr/bin/env python3
"""CLI Training Script for Isolated UCF-Crime Binary Anomaly Detector."""

from __future__ import annotations

import argparse
import os
import sys

# Ensure repo root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from ucf_binary_system.model import BinaryAnomalyHead
from ucf_binary_system.trainer import BinaryMILTrainer


def main():
    parser = argparse.ArgumentParser(description="Train UCF-Crime Binary Video Anomaly Detector")
    parser.add_argument("--features-root", type=str, default="extracted_features", help="Path to extracted_features directory")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.001, help="Weight decay")
    parser.add_argument("--k-top", type=int, default=3, help="Top-k anomalous segments for pooling")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="Validation partition ratio")
    parser.add_argument("--save-dir", type=str, default="ucf_binary_system/checkpoints", help="Output checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Pyramidal Binary Head: 768 -> 512 -> 256 -> 1
    model = BinaryAnomalyHead(in_features=768, hidden_dims=(512, 256), dropout_rates=(0.65, 0.55))

    trainer = BinaryMILTrainer(
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

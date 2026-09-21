"""PyTorch Trainer for UCF-Crime Binary Video Anomaly Detection."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ucf_binary_system.dataset import (
    UCFBinaryDataset,
    create_stratified_train_val_split,
    scan_ucf_crime_features,
)
from ucf_binary_system.loss import BinaryMILLoss


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes ROC-AUC for binary classification."""
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    pos_scores = y_score[pos_mask]
    neg_scores = y_score[neg_mask]
    pos_expanded = pos_scores[:, None]
    neg_expanded = neg_scores[None, :]

    concordant = np.sum(pos_expanded > neg_expanded)
    ties = np.sum(pos_expanded == neg_expanded)
    auc = (concordant + 0.5 * ties) / max(1, n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes Average Precision (PR-AUC) for binary classification."""
    if len(y_true) == 0 or np.sum(y_true) == 0:
        return 0.0

    desc_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_indices]
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)
    recalls = tp / max(1, np.sum(y_true))
    precisions = tp / np.maximum(tp + fp, 1e-7)

    ap = 0.0
    for r_thresh in np.linspace(0, 1, 11):
        prec_at_r = precisions[recalls >= r_thresh]
        if len(prec_at_r) > 0:
            ap += np.max(prec_at_r)
    return float(ap / 11.0)


class BinaryMILTrainer:
    """Trainer orchestrator for UCF-Crime binary MIL ranking head.

    Args:
        model: BinaryAnomalyHead PyTorch module.
        features_root: Directory containing ucf_* feature subfolders.
        epochs: Training epochs (default: 25).
        batch_size: DataLoader batch size (default: 32).
        learning_rate: Initial AdamW learning rate (default: 0.001).
        weight_decay: L2 penalty (default: 0.001).
        k_top: Number of top anomalous segments to pool (default: 3).
        device: 'cuda' or 'cpu'.
        save_dir: Checkpoint output directory.
    """

    def __init__(
        self,
        model: nn.Module,
        features_root: str = "extracted_features",
        epochs: int = 25,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 0.001,
        smoothness_weight: float = 0.0001,
        sparsity_weight: float = 0.0001,
        k_top: int = 3,
        val_ratio: float = 0.20,
        early_stop_patience: int = 10,
        device: Optional[str] = None,
        save_dir: str = "ucf_binary_system/checkpoints",
    ) -> None:
        self.model = model
        self.features_root = features_root
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.smoothness_weight = smoothness_weight
        self.sparsity_weight = sparsity_weight
        self.k_top = k_top
        self.val_ratio = val_ratio
        self.early_stop_patience = early_stop_patience
        self.save_dir = save_dir
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.save_dir, exist_ok=True)

        # Scan and split data
        all_samples = scan_ucf_crime_features(self.features_root)
        if not all_samples:
            raise ValueError(f"No UCF-Crime feature files found in {self.features_root}")

        self.train_samples, self.val_samples = create_stratified_train_val_split(
            all_samples, val_ratio=self.val_ratio, seed=42
        )

        print(f"• UCF-Crime Dataset Loaded: {len(all_samples)} total videos")
        print(f"  - Train Set: {len(self.train_samples)} videos")
        print(f"  - Val Set:   {len(self.val_samples)} videos")

    def _evaluate(
        self,
        loader: DataLoader,
        criterion: BinaryMILLoss,
    ) -> Tuple[float, float, float]:
        """Evaluates model on validation loader. Returns (val_loss, val_auc, val_ap)."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds: List[float] = []
        all_targets: List[float] = []

        with torch.no_grad():
            for feats, targets, _ in loader:
                # [B, 32, D]
                feats = feats.to(self.device)
                # [B, 1]
                targets = targets.to(self.device)

                # [B, 32, 1]
                logits = self.model(feats)
                loss_dict = criterion(logits, targets)
                total_loss += loss_dict["total"].item()
                num_batches += 1

                # Probabilities via sigmoid: [B, 32]
                probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                bag_scores = np.max(probs, axis=1)

                all_preds.extend(bag_scores.tolist())
                all_targets.extend(targets.squeeze().cpu().numpy().tolist())

        avg_loss = total_loss / max(1, num_batches)
        y_true = np.array(all_targets, dtype=int)
        y_pred = np.array(all_preds, dtype=float)

        val_auc = compute_roc_auc(y_true, y_pred)
        val_ap = compute_pr_auc(y_true, y_pred)

        return avg_loss, val_auc, val_ap

    def train(self) -> Dict[str, Any]:
        """Runs the complete binary MIL training loop."""
        self.model.to(self.device)
        train_ds = UCFBinaryDataset(self.train_samples)
        val_ds = UCFBinaryDataset(self.val_samples)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, drop_last=False)

        criterion = BinaryMILLoss(
            k=self.k_top,
            smoothness_weight=self.smoothness_weight,
            sparsity_weight=self.sparsity_weight,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        total_steps = self.epochs * max(1, len(train_loader))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

        best_val_auc = 0.0
        best_val_ap = 0.0
        best_val_loss = float("inf")
        best_epoch = 1
        patience = 0

        print("\n" + "=" * 65)
        print("🚀 TRAINING UCF-CRIME BINARY ANOMALY DETECTOR")
        print("=" * 65)

        start_t = time.time()
        for epoch in range(1, self.epochs + 1):
            ep_start = time.time()
            self.model.train()
            epoch_loss = 0.0

            for feats, targets, _ in train_loader:
                # [B, 32, D]
                feats = feats.to(self.device)
                # [B, 1]
                targets = targets.to(self.device)

                optimizer.zero_grad()
                # [B, 32, 1]
                logits = self.model(feats)
                losses = criterion(logits, targets)

                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.75)
                optimizer.step()
                scheduler.step()

                epoch_loss += losses["total"].item()

            train_loss = epoch_loss / max(1, len(train_loader))
            val_loss, val_auc, val_ap = self._evaluate(val_loader, criterion)
            ep_time = time.time() - ep_start

            print(
                f"Epoch {epoch:02d}/{self.epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val ROC-AUC: {val_auc:.4f} | "
                f"Val PR-AUC: {val_ap:.4f} ({ep_time:.1f}s)"
            )

            # Checkpoint on best ROC-AUC
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_ap = val_ap
                best_val_loss = val_loss
                best_epoch = epoch
                patience = 0

                save_path = os.path.join(self.save_dir, "best_binary_model.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "val_auc": best_val_auc,
                        "val_ap": best_val_ap,
                        "val_loss": best_val_loss,
                        "config": {
                            "in_features": getattr(self.model, "in_features", 768),
                            "hidden_dims": getattr(self.model, "hidden_dims", (512, 256)),
                        },
                    },
                    save_path,
                )
            else:
                patience += 1
                if patience >= self.early_stop_patience:
                    print(f"Early stopping triggered at Epoch {epoch}.")
                    break

        total_duration = time.time() - start_t
        print("=" * 65)
        print(f"✓ Training Complete in {total_duration:.1f}s")
        print(f"★ Best Epoch: {best_epoch} | ROC-AUC: {best_val_auc:.4f} | PR-AUC: {best_val_ap:.4f} | Val Loss: {best_val_loss:.4f}")
        print(f"★ Saved Checkpoint: {os.path.join(self.save_dir, 'best_binary_model.pt')}")
        print("=" * 65 + "\n")

        return {
            "best_epoch": best_epoch,
            "best_val_auc": best_val_auc,
            "best_val_ap": best_val_ap,
            "best_val_loss": best_val_loss,
            "duration_sec": total_duration,
        }

"""PyTorch Trainer for Combined UCF-Crime and XD-Violence Binary Video Anomaly Detection."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from combined_binary_system.dataset import (
    CombinedBinaryDataset,
    create_stratified_train_val_split,
    scan_combined_features,
)
from combined_binary_system.loss import BinaryMILLoss


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes ROC-AUC for binary classification without external scikit-learn dependency."""
    pos_mask = y_true == 1
    neg_mask = y_true == 0
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


class CombinedBinaryTrainer:
    """Trainer orchestrator for combined UCF-Crime & XD-Violence binary MIL ranking model.

    Args:
        model: DeepBinaryAnomalyHead PyTorch module.
        features_root: Directory containing ucf_* and xdv_* feature subfolders.
        epochs: Training epochs (default: 25).
        batch_size: DataLoader batch size (default: 32).
        learning_rate: Initial AdamW learning rate (default: 0.001).
        weight_decay: L2 penalty (default: 0.001).
        smoothness_weight: Weight for temporal smoothness regularization (default: 0.0001).
        sparsity_weight: Weight for anomaly segment sparsity (default: 0.0001).
        k_top: Number of top anomalous segments to pool (default: 3).
        val_ratio: Validation partition ratio (default: 0.20).
        early_stop_patience: Max epochs without Val ROC-AUC improvement (default: 8).
        device: 'cuda' or 'cpu'.
        save_dir: Checkpoint output directory.
    """

    def __init__(
        self,
        model: nn.Module,
        features_root: str = "swin3d_t_extracted_features",
        epochs: int = 25,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 0.001,
        smoothness_weight: float = 0.0001,
        sparsity_weight: float = 0.0001,
        k_top: int = 3,
        val_ratio: float = 0.20,
        early_stop_patience: int = 8,
        device: Optional[str] = None,
        save_dir: str = "combined_binary_system/checkpoints",
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
        all_samples = scan_combined_features(self.features_root)
        if not all_samples:
            raise ValueError(f"No feature files found in {self.features_root}")

        self.train_samples, self.val_samples = create_stratified_train_val_split(
            all_samples, val_ratio=self.val_ratio, seed=42
        )

        n_train_norm = sum(1 for _, lbl, _ in self.train_samples if lbl == 0.0)
        n_train_anom = sum(1 for _, lbl, _ in self.train_samples if lbl == 1.0)
        n_val_norm = sum(1 for _, lbl, _ in self.val_samples if lbl == 0.0)
        n_val_anom = sum(1 for _, lbl, _ in self.val_samples if lbl == 1.0)

        print(f"• Combined Dataset Loaded: {len(all_samples)} total videos")
        print(f"  - Train Set: {len(self.train_samples)} videos ({n_train_norm} Normal, {n_train_anom} Anomaly)")
        print(f"  - Val Set:   {len(self.val_samples)} videos ({n_val_norm} Normal, {n_val_anom} Anomaly)")

        # Create datasets and loaders
        train_dataset = CombinedBinaryDataset(self.train_samples)
        val_dataset = CombinedBinaryDataset(self.val_samples)

        use_pin = self.device.type == "cuda"
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=use_pin,
            drop_last=False,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=use_pin,
            drop_last=False,
        )

        # Optimizer, Scheduler & Loss
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=1e-5,
        )
        self.criterion = BinaryMILLoss(
            k=self.k_top,
            smoothness_weight=self.smoothness_weight,
            sparsity_weight=self.sparsity_weight,
        )

    def train_epoch(self) -> float:
        """Runs one training epoch and returns mean total loss."""
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0

        for feats, labels, _ in self.train_loader:
            feats = feats.to(self.device)  # [B, 32, 768]
            labels = labels.to(self.device)  # [B, 1]

            self.optimizer.zero_grad()
            logits = self.model(feats)  # [B, 32, 1]
            loss_dict = self.criterion(logits, labels)
            loss = loss_dict["total"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        self.scheduler.step()
        return epoch_loss / max(1, n_batches)

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float, float]:
        """Evaluates on the validation set.

        Returns:
            Tuple of (mean_val_loss, val_roc_auc, val_pr_auc).
        """
        self.model.eval()
        val_loss = 0.0
        n_batches = 0
        all_labels: List[float] = []
        all_scores: List[float] = []

        for feats, labels, _ in self.val_loader:
            feats = feats.to(self.device)  # [B, 32, 768]
            labels = labels.to(self.device)  # [B, 1]

            logits = self.model(feats)  # [B, 32, 1]
            loss_dict = self.criterion(logits, labels)
            val_loss += loss_dict["total"].item()
            n_batches += 1

            # [B, 32, 1] -> [B, 32]
            probs = torch.sigmoid(logits).squeeze(-1)
            # Bag score: mean of top-k segment anomaly probabilities
            k_val = min(self.k_top, probs.shape[1])
            bag_scores = torch.topk(probs, k=k_val, dim=1).values.mean(dim=1).cpu().numpy()

            all_labels.extend(labels.squeeze(-1).cpu().numpy().tolist())
            all_scores.extend(bag_scores.tolist())

        y_true = np.array(all_labels, dtype=np.int64)
        y_score = np.array(all_scores, dtype=np.float64)

        roc_auc = compute_roc_auc(y_true, y_score)
        pr_auc = compute_pr_auc(y_true, y_score)
        mean_val_loss = val_loss / max(1, n_batches)

        return mean_val_loss, roc_auc, pr_auc

    def train(self) -> Dict[str, Any]:
        """Executes the full training routine with early stopping and checkpointing."""
        best_roc_auc = 0.0
        best_epoch = 0
        epochs_no_improve = 0
        best_checkpoint_path = os.path.join(self.save_dir, "best_combined_binary_model.pt")

        print("\n" + "=" * 65)
        print("🚀 TRAINING COMBINED UCF-CRIME & XD-VIOLENCE BINARY DETECTOR")
        print("=" * 65)

        start_time = time.time()
        for epoch in range(1, self.epochs + 1):
            ep_start = time.time()
            train_loss = self.train_epoch()
            val_loss, val_roc_auc, val_pr_auc = self.evaluate()
            ep_dur = time.time() - ep_start

            print(
                f"Epoch {epoch:02d}/{self.epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val ROC-AUC: {val_roc_auc:.4f} | "
                f"Val PR-AUC: {val_pr_auc:.4f} "
                f"({ep_dur:.1f}s)"
            )

            # Check for best checkpoint
            if val_roc_auc > best_roc_auc:
                best_roc_auc = val_roc_auc
                best_epoch = epoch
                epochs_no_improve = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_roc_auc": val_roc_auc,
                        "val_pr_auc": val_pr_auc,
                        "val_loss": val_loss,
                        "config": {
                            "in_features": getattr(self.model, "in_features", 768),
                            "hidden_dims": getattr(self.model, "hidden_dims", (512, 256, 128)),
                            "dropout_rates": getattr(self.model, "dropout_rates", (0.70, 0.60, 0.50)),
                            "noise_std": getattr(self.model, "noise_std", 0.10),
                        },
                    },
                    best_checkpoint_path,
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.early_stop_patience:
                    print(f"Early stopping triggered at Epoch {epoch}.")
                    break

        total_time = time.time() - start_time
        print("=" * 65)
        print(f"✓ Training Complete in {total_time:.1f}s")
        print(f"★ Best Epoch: {best_epoch} | Best ROC-AUC: {best_roc_auc:.4f}")
        print(f"★ Saved Checkpoint: {best_checkpoint_path}")
        print("=" * 65 + "\n")

        return {
            "best_epoch": best_epoch,
            "best_roc_auc": best_roc_auc,
            "checkpoint_path": best_checkpoint_path,
        }

from __future__ import annotations

import copy
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from data.multiclass_dataset import MultiClassFeatureDataset, create_multiclass_kfold_splits
from loss.multiclass_mil_loss import MultiClassMILLoss
from utils.annotation_parser import XD_CLASSES, CLASS_NAMES


def compute_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes Average Precision (AP) using pure NumPy.

    Args:
        y_true: 1D array of binary ground truth labels (0 or 1).
        y_score: 1D array of predicted confidence scores.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    if np.sum(y_true) == 0:
        return 0.0
    desc_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_indices]
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    recalls = tps / tps[-1]
    precisions = tps / (tps + fps)
    recalls = np.concatenate([[0.0], recalls])
    precisions = np.concatenate([[1.0], precisions])
    return float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]))


class MultiClassMILTrainer:
    """Trainer for Multi-Class Multiple Instance Learning (MIL) Video Anomaly Detection.

    Supports:
    - Stratified K-Fold Cross Validation preserving normal vs anomalous ratio.
    - Top-K Multi-Label BCE loss with temporal smoothness and sparsity penalties.
    - Full TensorBoard logging: step loss, epoch metrics, and per-class AP.
    - CosineAnnealingLR scheduling and AdamW optimization.
    - Gradient clipping and Early Stopping.
    - Best and Last checkpoint persistence per fold.
    """

    def __init__(
        self,
        model: nn.Module,
        features_dir: Optional[str] = None,
        normal_dir: Optional[str] = None,
        anomal_dir: Optional[str] = None,
        epochs: int = 15,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        weight_decay: float = 0.001,
        smoothness_weight: float = 0.0001,
        sparsity_weight: float = 0.0001,
        k_top: int = 1,
        k_fold: int = 5,
        val_ratio: float = 0.2,
        early_stop_patience: int = 8,
        max_grad_norm: float = 0.75,
        device: Optional[str] = None,
        save_dir: str = "checkpoints/multiclass",
        log_dir: str = "runs/multiclass",
        class_list: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.smoothness_weight = smoothness_weight
        self.sparsity_weight = sparsity_weight
        self.k_top = k_top
        self.k_fold = max(1, k_fold)
        self.val_ratio = val_ratio
        self.early_stop_patience = early_stop_patience
        self.max_grad_norm = max_grad_norm
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.class_list = class_list if class_list is not None else XD_CLASSES

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize dataset to retrieve all file paths
        self.full_dataset = MultiClassFeatureDataset(
            features_dir=features_dir,
            normal_dir=normal_dir,
            anomal_dir=anomal_dir,
            class_list=self.class_list,
        )
        self.all_files = self.full_dataset.file_paths

    def _evaluate(
        self,
        loader: DataLoader,
        criterion: MultiClassMILLoss,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Evaluates model on validation loader, calculating losses and per-class AP."""
        self.model.eval()
        total_loss = 0.0
        total_bce = 0.0
        total_smooth = 0.0
        total_sparse = 0.0
        num_batches = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for feats, targets, _ in loader:
                feats = feats.to(self.device)
                targets = targets.to(self.device)

                scores = self.model(feats)  # (B, T, C)
                loss_dict = criterion(scores, targets)

                total_loss += loss_dict["total"].item()
                total_bce += loss_dict["bce"].item()
                total_smooth += loss_dict["smoothness"].item()
                total_sparse += loss_dict["sparsity"].item()
                num_batches += 1

                # Top-K pooling to get video-level probabilities
                topk_logits = torch.topk(scores, k=min(self.k_top, scores.shape[1]), dim=1).values.mean(dim=1)
                probs = torch.sigmoid(topk_logits).cpu().numpy()
                all_preds.append(probs)
                all_targets.append(targets.cpu().numpy())

        num_batches = max(1, num_batches)
        avg_losses = {
            "total": total_loss / num_batches,
            "bce": total_bce / num_batches,
            "smoothness": total_smooth / num_batches,
            "sparsity": total_sparse / num_batches,
        }

        # Calculate per-class AP and mAP
        if all_preds:
            y_pred = np.vstack(all_preds)
            y_true = np.vstack(all_targets)
            per_class_ap: Dict[str, float] = {}
            for idx, c_name in enumerate(self.class_list):
                ap = compute_average_precision(y_true[:, idx], y_pred[:, idx])
                per_class_ap[c_name] = ap
            per_class_ap["mAP"] = float(np.mean(list(per_class_ap.values())))
        else:
            per_class_ap = {c: 0.0 for c in self.class_list}
            per_class_ap["mAP"] = 0.0

        return avg_losses, per_class_ap

    def train(self) -> Dict[str, Any]:
        """Runs K-Fold cross validation training."""
        initial_weights = copy.deepcopy(self.model.state_dict())

        # Generate K-Fold splits
        folds = create_multiclass_kfold_splits(
            self.all_files,
            num_folds=self.k_fold,
            seed=42,
        )

        fold_results: List[Dict[str, Any]] = []
        global_step = 0

        for fold_idx in range(self.k_fold):
            fold_num = fold_idx + 1
            print(f"\n{'='*20} MULTI-CLASS FOLD {fold_num}/{self.k_fold} {'='*20}")

            fold_log_dir = os.path.join(self.log_dir, f"fold_{fold_num}")
            writer = SummaryWriter(log_dir=fold_log_dir)

            # Reset model to initial weights
            self.model.load_state_dict(initial_weights)
            self.model.to(self.device)

            train_files, val_files = folds[fold_idx]
            train_dataset = MultiClassFeatureDataset(files=train_files, class_list=self.class_list)
            val_dataset = MultiClassFeatureDataset(files=val_files, class_list=self.class_list)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
            )

            criterion = MultiClassMILLoss(
                k=self.k_top,
                smoothness_weight=self.smoothness_weight,
                sparsity_weight=self.sparsity_weight,
                from_logits=True,
            )
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.epochs * max(1, len(train_loader)),
                eta_min=1e-5,
            )

            best_val_loss = float("inf")
            best_val_map = 0.0
            patience_counter = 0

            for epoch in range(1, self.epochs + 1):
                self.model.train()
                epoch_loss = 0.0
                epoch_bce = 0.0
                epoch_smooth = 0.0
                epoch_sparse = 0.0

                for batch_idx, (feats, targets, _) in enumerate(train_loader):
                    feats = feats.to(self.device)
                    targets = targets.to(self.device)

                    optimizer.zero_grad()
                    scores = self.model(feats)
                    losses = criterion(scores, targets)

                    losses["total"].backward()
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    optimizer.step()
                    scheduler.step()

                    global_step += 1
                    writer.add_scalar("Step/Loss_Total", losses["total"].item(), global_step)
                    writer.add_scalar("Step/Loss_BCE", losses["bce"].item(), global_step)
                    writer.add_scalar("Step/LR", scheduler.get_last_lr()[0], global_step)

                    epoch_loss += losses["total"].item()
                    epoch_bce += losses["bce"].item()
                    epoch_smooth += losses["smoothness"].item()
                    epoch_sparse += losses["sparsity"].item()

                num_train_batches = max(1, len(train_loader))
                train_avg_loss = epoch_loss / num_train_batches

                # Validation
                val_losses, val_metrics = self._evaluate(val_loader, criterion)

                writer.add_scalar("Epoch/Train_Loss", train_avg_loss, epoch)
                writer.add_scalar("Epoch/Val_Loss", val_losses["total"], epoch)
                writer.add_scalar("Epoch/Val_BCE", val_losses["bce"], epoch)
                writer.add_scalar("Epoch/Val_mAP", val_metrics["mAP"], epoch)
                for c in self.class_list:
                    writer.add_scalar(f"Epoch/Val_AP_{c}", val_metrics.get(c, 0.0), epoch)

                print(
                    f"Fold {fold_num} Epoch {epoch:02d}/{self.epochs:02d} | "
                    f"Train Loss: {train_avg_loss:.4f} | "
                    f"Val Loss: {val_losses['total']:.4f} | "
                    f"Val mAP: {val_metrics['mAP']:.4f}"
                )

                # Early stopping and checkpointing
                if val_losses["total"] < best_val_loss:
                    best_val_loss = val_losses["total"]
                    best_val_map = val_metrics["mAP"]
                    patience_counter = 0

                    best_path = os.path.join(self.save_dir, f"best_loss_fold_{fold_num}.pt")
                    torch.save(
                        {
                            "fold": fold_num,
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "val_loss": best_val_loss,
                            "val_map": best_val_map,
                            "per_class_ap": val_metrics,
                        },
                        best_path,
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stop_patience:
                        print(f"Early stopping triggered for Fold {fold_num} at Epoch {epoch}.")
                        break

            # Save last checkpoint
            last_path = os.path.join(self.save_dir, f"last_fold_{fold_num}.pt")
            torch.save(
                {
                    "fold": fold_num,
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_loss": val_losses["total"],
                    "val_map": val_metrics["mAP"],
                },
                last_path,
            )

            writer.close()
            fold_results.append({
                "fold": fold_num,
                "best_val_loss": best_val_loss,
                "best_val_map": best_val_map,
            })

        mean_loss = float(np.mean([r["best_val_loss"] for r in fold_results]))
        mean_map = float(np.mean([r["best_val_map"] for r in fold_results]))
        print(f"\nTraining Complete across {self.k_fold} folds. Mean Best Val Loss: {mean_loss:.4f}, Mean Best mAP: {mean_map:.4f}")

        return {
            "fold_results": fold_results,
            "mean_best_loss": mean_loss,
            "mean_best_map": mean_map,
        }

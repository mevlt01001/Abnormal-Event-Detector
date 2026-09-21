"""Class-Aware 8-Fold MIL Trainer with TensorBoard and AMOLED Telemetry."""

from __future__ import annotations

import copy
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset import ClassAwareVideoDataset, create_stratified_kfold_splits, load_feature_tensor, scan_feature_files
from data.taxonomy import DEFAULT_MACRO_CLASSES, load_dataset_taxonomy
from loss.mil_loss import ClassAwareMILLoss


def compute_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes Average Precision (AP) for a single class using the PR curve area."""
    if len(y_true) == 0 or np.sum(y_true) == 0:
        return 0.0

    desc_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_indices]

    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)
    recalls = tp / max(1, np.sum(y_true))
    precisions = tp / np.maximum(tp + fp, 1e-7)

    # 11-point interpolated AP
    ap = 0.0
    for r_thresh in np.linspace(0, 1, 11):
        prec_at_r = precisions[recalls >= r_thresh]
        if len(prec_at_r) > 0:
            ap += np.max(prec_at_r)
    return float(ap / 11.0)


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes Receiver Operating Characteristic Area Under Curve (ROC-AUC)."""
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))

    if n_pos == 0 or n_neg == 0:
        return 0.5

    pos_scores = y_score[pos_mask]
    neg_scores = y_score[neg_mask]

    n_pairs = n_pos * n_neg
    pos_expanded = pos_scores[:, None]
    neg_expanded = neg_scores[None, :]

    concordant = np.sum(pos_expanded > neg_expanded)
    ties = np.sum(pos_expanded == neg_expanded)
    auc = (concordant + 0.5 * ties) / max(1, n_pairs)
    return float(np.clip(auc, 0.0, 1.0))


class ClassAwareMILTrainer:
    """8-Fold Stratified Trainer for Class-Aware Video Anomaly Detection.

    Args:
        model: PyTorch model or classification head (e.g. AnomalyHead or AnomalyDetector).
        features_dir: Directory containing consolidated .pt feature files.
        epochs: Number of training epochs per fold (default: 20).
        batch_size: DataLoader mini-batch size (default: 32).
        learning_rate: Initial AdamW learning rate (default: 0.001).
        weight_decay: L2 regularization penalty (default: 0.001).
        k_top: Top-k segments pooled per bag (default: 1).
        k_fold: Number of stratified folds (default: 8).
        device: 'cuda' or 'cpu'.
        save_dir: Checkpoint directory.
        log_dir: TensorBoard log directory.
        state_file: Live JSON telemetry destination for mobile phone monitor.
        class_list: List of macro class names.
    """

    def __init__(
        self,
        model: nn.Module,
        features_dir: str = "data/unified_features",
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 0.001,
        smoothness_weight: float = 0.0001,
        sparsity_weight: float = 0.0001,
        k_top: int = 1,
        k_fold: int = 8,
        early_stop_patience: int = 10,
        device: Optional[str] = None,
        save_dir: str = "checkpoints/class_aware_8fold",
        log_dir: str = "runs/class_aware_8fold",
        state_file: str = "results/training_state.json",
        class_list: Optional[Sequence[str]] = None,
    ) -> None:
        self.model = model
        self.features_dir = features_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.smoothness_weight = smoothness_weight
        self.sparsity_weight = sparsity_weight
        self.k_top = k_top
        self.k_fold = max(1, k_fold)
        self.early_stop_patience = early_stop_patience
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.state_file = state_file
        if class_list is not None:
            self.class_list = list(class_list)
        else:
            tax = load_dataset_taxonomy(self.features_dir)
            self.class_list = tax["anomaly_classes"]
        self.num_classes = len(self.class_list)

        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

        self.all_files = scan_feature_files(self.features_dir)
        if not self.all_files:
            raise ValueError(f"No feature files found in {self.features_dir}")

        # Dynamically discover feature dimension from model or first sample file
        if hasattr(self.model, "in_features"):
            self.in_features = int(self.model.in_features)
        elif hasattr(self.model, "head") and hasattr(self.model.head, "in_features"):
            self.in_features = int(self.model.head.in_features)
        else:
            first_sample = load_feature_tensor(self.all_files[0])
            self.in_features = int(first_sample.shape[-1])

    def _update_state(self, state_data: Dict[str, Any]) -> None:
        """Writes live training status to JSON for AMOLED mobile phone monitor."""
        try:
            temp_file = f"{self.state_file}.tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=2)
            os.replace(temp_file, self.state_file)
        except Exception:
            pass

    def _evaluate(
        self,
        loader: DataLoader,
        criterion: ClassAwareMILLoss,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """Evaluates model on validation fold, returning losses, APs, and ROC-AUC."""
        self.model.eval()
        total_loss, total_bce, total_smooth, total_sparse = 0.0, 0.0, 0.0, 0.0
        num_batches = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for feats, targets, _ in loader:
                # [B, 32, 768]
                feats = feats.to(self.device)
                # [B, num_classes]
                targets = targets.to(self.device)

                # [B, 32, 768] -> [B, 32, num_classes]
                logits = self.model(feats)
                loss_dict = criterion(logits, targets)

                total_loss += loss_dict["total"].item()
                total_bce += loss_dict["bce"].item()
                total_smooth += loss_dict["smoothness"].item()
                total_sparse += loss_dict["sparsity"].item()
                num_batches += 1

                # Top-K pooling per bag
                # [B, 32, num_classes]
                probs = torch.sigmoid(logits)
                k_val = min(self.k_top, probs.shape[1])
                # [B, num_classes]
                bag_scores = torch.topk(probs, k=k_val, dim=1).values.mean(dim=1).cpu().numpy()

                all_preds.extend(bag_scores.tolist())
                all_targets.extend(targets.cpu().numpy().tolist())

        num_batches = max(1, num_batches)
        avg_losses = {
            "total": total_loss / num_batches,
            "bce": total_bce / num_batches,
            "smoothness": total_smooth / num_batches,
            "sparsity": total_sparse / num_batches,
        }

        y_true = np.array(all_targets)  # [N, num_classes]
        y_score = np.array(all_preds)   # [N, num_classes]

        # Calculate per-class AP and overall mAP
        per_class_ap = {}
        for c_idx, c_name in enumerate(self.class_list):
            ap = compute_average_precision(y_true[:, c_idx], y_score[:, c_idx])
            per_class_ap[c_name] = ap

        mean_ap = float(np.mean(list(per_class_ap.values())))

        # Macro Anomaly ROC-AUC: bag is positive if any anomaly class is present
        y_true_binary = (np.sum(y_true, axis=1) > 0.5).astype(int)
        y_score_binary = np.max(y_score, axis=1)
        roc_auc = compute_roc_auc(y_true_binary, y_score_binary)

        metrics = {
            "mAP": mean_ap,
            "ROC_AUC": roc_auc,
        }

        return avg_losses, metrics, per_class_ap

    def train(self) -> Dict[str, Any]:
        """Executes full 8-Fold cross-validation."""
        initial_weights = copy.deepcopy(self.model.state_dict())

        folds = create_stratified_kfold_splits(
            self.all_files,
            num_folds=self.k_fold,
            seed=42,
            class_list=self.class_list,
        )

        fold_results: List[Dict[str, Any]] = []
        global_step = 0
        start_time = time.time()

        summary_writer = SummaryWriter(log_dir=os.path.join(self.log_dir, "summary"))

        for fold_idx in range(self.k_fold):
            fold_num = fold_idx + 1
            print(f"\n{'='*25} CLASS-AWARE FOLD {fold_num}/{self.k_fold} {'='*25}")

            writer = SummaryWriter(log_dir=os.path.join(self.log_dir, f"fold_{fold_num}"))

            self.model.load_state_dict(initial_weights)
            self.model.to(self.device)

            train_files, val_files = folds[fold_idx]
            train_dataset = ClassAwareVideoDataset(train_files, class_list=self.class_list)
            val_dataset = ClassAwareVideoDataset(val_files, class_list=self.class_list)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

            criterion = ClassAwareMILLoss(
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

            best_val_loss = float("inf")
            best_val_map = 0.0
            best_val_auc = 0.0
            best_epoch = 1
            patience = 0

            for epoch in range(1, self.epochs + 1):
                ep_start = time.time()
                self.model.train()
                epoch_loss = 0.0

                for batch_idx, (feats, targets, _) in enumerate(train_loader):
                    # [B, 32, feature_dim]
                    feats = feats.to(self.device)
                    # [B, num_classes]
                    targets = targets.to(self.device)

                    optimizer.zero_grad()
                    # [B, 32, feature_dim] -> [B, 32, num_classes]
                    logits = self.model(feats)
                    losses = criterion(logits, targets)

                    losses["total"].backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.75)
                    optimizer.step()
                    scheduler.step()

                    global_step += 1
                    writer.add_scalar("Step/Loss_Total", losses["total"].item(), global_step)
                    writer.add_scalar("Step/Loss_BCE", losses["bce"].item(), global_step)
                    writer.add_scalar("Step/LR", scheduler.get_last_lr()[0], global_step)
                    epoch_loss += losses["total"].item()

                train_avg_loss = epoch_loss / max(1, len(train_loader))
                val_losses, val_metrics, val_class_ap = self._evaluate(val_loader, criterion)

                writer.add_scalar("Epoch/Train_Loss", train_avg_loss, epoch)
                writer.add_scalar("Epoch/Val_Loss", val_losses["total"], epoch)
                writer.add_scalar("Epoch/Val_mAP", val_metrics["mAP"], epoch)
                writer.add_scalar("Epoch/Val_ROC_AUC", val_metrics["ROC_AUC"], epoch)

                for c_name, ap_val in val_class_ap.items():
                    writer.add_scalar(f"PerClass_AP/{c_name}", ap_val, epoch)

                ep_time = time.time() - ep_start
                total_elapsed = time.time() - start_time
                completed = fold_idx * self.epochs + epoch
                eta = (self.k_fold * self.epochs - completed) * (total_elapsed / max(1, completed))

                print(
                    f"Fold {fold_num}/{self.k_fold} Epoch {epoch:02d}/{self.epochs:02d} | "
                    f"Train Loss: {train_avg_loss:.4f} | "
                    f"Val Loss: {val_losses['total']:.4f} | "
                    f"Val mAP: {val_metrics['mAP']:.4f} | "
                    f"Val AUC: {val_metrics['ROC_AUC']:.4f} ({ep_time:.1f}s)"
                )

                config_dict = {
                    "in_features": self.in_features,
                    "num_classes": self.num_classes,
                    "class_list": self.class_list,
                    "hidden_dims": getattr(self.model, "hidden_dims", (512, 256)),
                }

                # Checkpointing
                if val_metrics["mAP"] > best_val_map:
                    best_val_map = val_metrics["mAP"]
                    best_val_auc = val_metrics["ROC_AUC"]
                    best_epoch = epoch
                    torch.save(
                        {
                            "fold": fold_num,
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "val_loss": val_losses["total"],
                            "val_map": best_val_map,
                            "val_auc": best_val_auc,
                            "class_list": self.class_list,
                            "config": config_dict,
                        },
                        os.path.join(self.save_dir, f"best_map_fold_{fold_num}.pt"),
                    )

                if val_losses["total"] < best_val_loss:
                    best_val_loss = val_losses["total"]
                    patience = 0
                    torch.save(
                        {
                            "fold": fold_num,
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "val_loss": best_val_loss,
                            "val_map": val_metrics["mAP"],
                            "val_auc": val_metrics["ROC_AUC"],
                            "class_list": self.class_list,
                            "config": config_dict,
                        },
                        os.path.join(self.save_dir, f"best_loss_fold_{fold_num}.pt"),
                    )
                else:
                    patience += 1

                # Live status for mobile monitor
                live_state = {
                    "is_training": True,
                    "status": "CLASS_AWARE_TRAINING",
                    "mode": f"Class-Aware MIL ({self.num_classes} Macro Classes)",
                    "current_fold": fold_num,
                    "total_folds": self.k_fold,
                    "current_epoch": epoch,
                    "total_epochs": self.epochs,
                    "train_loss": round(train_avg_loss, 4),
                    "val_loss": round(val_losses["total"], 4),
                    "val_map": round(val_metrics["mAP"], 4),
                    "val_auc": round(val_metrics["ROC_AUC"], 4),
                    "best_val_map": round(best_val_map, 4),
                    "best_val_loss": round(best_val_loss, 4),
                    "best_epoch": best_epoch,
                    "per_class_ap": {k: round(v * 100, 2) for k, v in val_class_ap.items()},
                    "completed_folds": fold_results,
                    "elapsed_sec": round(total_elapsed, 1),
                    "eta_sec": round(eta, 1),
                    "timestamp": time.time(),
                }
                self._update_state(live_state)

                if patience >= self.early_stop_patience:
                    print(f"Early stopping triggered for Fold {fold_num} at Epoch {epoch}.")
                    break

            writer.close()
            fold_results.append({
                "fold": fold_num,
                "best_val_map": round(best_val_map, 4),
                "best_val_auc": round(best_val_auc, 4),
                "best_val_loss": round(best_val_loss, 4),
                "best_epoch": best_epoch,
            })
            summary_writer.add_scalar("Fold_Comparison/Best_Val_mAP", best_val_map, fold_num)
            summary_writer.add_scalar("Fold_Comparison/Best_Val_AUC", best_val_auc, fold_num)

        summary_writer.close()

        # Aggregate Statistics
        mean_map = float(np.mean([r["best_val_map"] for r in fold_results]))
        std_map = float(np.std([r["best_val_map"] for r in fold_results]))
        mean_auc = float(np.mean([r["best_val_auc"] for r in fold_results]))
        std_auc = float(np.std([r["best_val_auc"] for r in fold_results]))
        mean_loss = float(np.mean([r["best_val_loss"] for r in fold_results]))
        std_loss = float(np.std([r["best_val_loss"] for r in fold_results]))

        total_duration = time.time() - start_time
        summary_data = {
            "num_folds": self.k_fold,
            "epochs_per_fold": self.epochs,
            "total_duration_sec": round(total_duration, 1),
            "mean_best_map": round(mean_map, 4),
            "std_best_map": round(std_map, 4),
            "mean_best_auc": round(mean_auc, 4),
            "std_best_auc": round(std_auc, 4),
            "mean_best_loss": round(mean_loss, 4),
            "std_best_loss": round(std_loss, 4),
            "fold_results": fold_results,
            "class_list": self.class_list,
        }

        # Save summary JSON
        summary_path = os.path.join(self.save_dir, "8fold_training_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        results_summary = "results/8fold_training_summary.json"
        with open(results_summary, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        # Mark completed in mobile dashboard
        final_state = {
            "is_training": False,
            "status": "COMPLETED",
            "mode": f"Class-Aware MIL ({self.num_classes} Macro Classes)",
            "current_fold": self.k_fold,
            "total_folds": self.k_fold,
            "current_epoch": self.epochs,
            "total_epochs": self.epochs,
            "mean_best_map": round(mean_map, 4),
            "std_best_map": round(std_map, 4),
            "mean_best_auc": round(mean_auc, 4),
            "std_best_auc": round(std_auc, 4),
            "mean_best_loss": round(mean_loss, 4),
            "std_best_loss": round(std_loss, 4),
            "completed_folds": fold_results,
            "elapsed_sec": round(total_duration, 1),
            "eta_sec": 0.0,
            "timestamp": time.time(),
        }
        self._update_state(final_state)

        print(f"\n{'#'*65}")
        print(f"🎉 8-FOLD CLASS-AWARE TRAINING COMPLETED ({total_duration:.1f}s)")
        print(f"Mean Best Val mAP:  {mean_map:.4f} ± {std_map:.4f}")
        print(f"Mean Best Val AUC:  {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"Mean Best Val Loss: {mean_loss:.4f} ± {std_loss:.4f}")
        print(f"Saved summary to:   {summary_path}")
        print(f"{'#'*65}\n")

        return summary_data

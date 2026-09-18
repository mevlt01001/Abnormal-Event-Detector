from __future__ import annotations

import copy
import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from data.dataset import _load_features
from loss.mil_loss import VideoAnomalyLoss
from model.ranking_head import SegmentRankingHead


class MILTrainer:
    """Trainer for Sultani Multiple Instance Learning (MIL) Video Anomaly Detection.

    Supports:
    - K-Fold Cross Validation (or train/val split if k_fold=1).
    - Step-wise, Epoch-wise, and Fold-wise TensorBoard logging.
    - CosineAnnealingLR scheduling and AdamW optimization.
    - Gradient clipping and Early Stopping.
    - Best and Last checkpoint persistence per fold.
    """

    def __init__(
        self,
        model: nn.Module,
        anomal_dir: str,
        normal_dir: str,
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        weight_decay: float = 0.001,
        smoothness_weight: float = 0.0001,
        sparsity_weight: float = 0.0001,
        k_fold: int = 5,
        val_ratio: float = 0.2,
        early_stop_patience: int = 8,
        max_grad_norm: float = 0.75,
        device: Optional[str] = None,
        save_dir: str = "checkpoints",
        log_dir: str = "runs",
    ) -> None:
        self.model = model
        self.anomal_dir = anomal_dir
        self.normal_dir = normal_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.smoothness_weight = smoothness_weight
        self.sparsity_weight = sparsity_weight
        self.k_fold = max(1, k_fold)
        self.val_ratio = val_ratio
        self.early_stop_patience = early_stop_patience
        self.max_grad_norm = max_grad_norm
        self.save_dir = save_dir
        self.log_dir = log_dir

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self._scan_files()

    def _scan_files(self) -> None:
        if not os.path.isdir(self.anomal_dir):
            raise NotADirectoryError(f"Anomalous directory does not exist: {self.anomal_dir}")
        if not os.path.isdir(self.normal_dir):
            raise NotADirectoryError(f"Normal directory does not exist: {self.normal_dir}")

        self.anomal_files = [
            os.path.join(self.anomal_dir, f)
            for f in os.listdir(self.anomal_dir)
            if f.endswith(".pt")
        ]
        self.normal_files = [
            os.path.join(self.normal_dir, f)
            for f in os.listdir(self.normal_dir)
            if f.endswith(".pt")
        ]

        if not self.anomal_files:
            raise ValueError(f"No .pt files found in {self.anomal_dir}")
        if not self.normal_files:
            raise ValueError(f"No .pt files found in {self.normal_dir}")

    def _chunk_list(self, items: List[str], k: int) -> List[List[str]]:
        """Partitions a list into k roughly equal chunks."""
        n = len(items)
        if k <= 1:
            return [items]
        chunk_size = max(1, n // k)
        chunks = []
        for i in range(k):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < k - 1 else n
            chunks.append(items[start:end])
        return chunks

    def _load_batch(self, file_paths: List[str]) -> torch.Tensor:
        """Loads a batch of feature files and stacks them into (B, N, Dim)."""
        tensors = [_load_features(fp) for fp in file_paths]
        # Check if all shapes match
        if all(t.shape == tensors[0].shape for t in tensors):
            return torch.stack(tensors).to(self.device)
        # Pad along segment dimension if varying
        max_segments = max(t.shape[0] for t in tensors)
        padded = []
        for t in tensors:
            if t.shape[0] < max_segments:
                pad_t = F.pad(t, (0, 0, 0, max_segments - t.shape[0]))
                padded.append(pad_t)
            else:
                padded.append(t)
        return torch.stack(padded).to(self.device)

    def train(self) -> Dict[str, Any]:
        """Runs K-Fold cross validation training with full TensorBoard metrics."""
        initial_weights = copy.deepcopy(self.model.state_dict())
        anomal_files_shuffled = list(self.anomal_files)
        random.shuffle(anomal_files_shuffled)

        anomal_chunks = self._chunk_list(anomal_files_shuffled, self.k_fold)
        fold_best_losses: List[float] = []

        global_step = 0

        for fold in range(self.k_fold):
            fold_num = fold + 1
            print(f"\n{'='*20} FOLD {fold_num}/{self.k_fold} {'='*20}")

            fold_log_dir = os.path.join(self.log_dir, f"fold_{fold_num}")
            writer = SummaryWriter(log_dir=fold_log_dir)

            # Reset model to initial weights
            self.model.load_state_dict(initial_weights)
            self.model.to(self.device)

            # Split train and val files
            if self.k_fold == 1:
                n_val = max(1, int(len(self.anomal_files) * self.val_ratio))
                val_anomal = anomal_files_shuffled[:n_val]
                train_anomal = anomal_files_shuffled[n_val:]
            else:
                val_anomal = anomal_chunks[fold]
                train_anomal = [
                    f for i, chunk in enumerate(anomal_chunks) if i != fold for f in chunk
                ]
                if not train_anomal:
                    train_anomal = list(val_anomal)
                if not val_anomal:
                    val_anomal = list(train_anomal)

            train_normal = list(self.normal_files)

            if len(self.normal_files) < len(val_anomal):
                val_normal = random.choices(self.normal_files, k=len(val_anomal))
            else:
                val_normal = random.sample(self.normal_files, k=len(val_anomal))

            criterion = VideoAnomalyLoss(
                smoothness_weight=self.smoothness_weight,
                sparsity_weight=self.sparsity_weight,
            )
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

            num_batches = max(1, (len(train_anomal) + self.batch_size - 1) // self.batch_size)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.epochs * num_batches,
                eta_min=1e-5,
            )

            best_loss = float("inf")
            patience_counter = 0

            for epoch_idx in range(self.epochs):
                self.model.train()
                epoch_train_loss = 0.0
                epoch_train_hinge = 0.0
                epoch_train_smooth = 0.0
                epoch_train_sparse = 0.0
                epoch_grad_norm = 0.0
                current_lr = scheduler.get_last_lr()[0]
                grad_norm_val = 0.0
                batches_processed = 0

                random.shuffle(train_anomal)
                batch_splits = [
                    train_anomal[i * self.batch_size : (i + 1) * self.batch_size]
                    for i in range(num_batches)
                ]

                for b_idx in range(num_batches):
                    anomal_batch_files = batch_splits[b_idx]
                    if not anomal_batch_files:
                        continue
                    current_b_size = len(anomal_batch_files)
                    normal_batch_files = random.choices(train_normal, k=current_b_size)

                    optimizer.zero_grad()

                    fa_batch = self._load_batch(anomal_batch_files)  # (B, N, Dim)
                    fn_batch = self._load_batch(normal_batch_files)  # (B, N, Dim)

                    y_anom = self.model(fa_batch)
                    y_norm = self.model(fn_batch)

                    loss_dict = criterion(y_anom, y_norm)
                    loss = loss_dict["total"]

                    loss.backward()

                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.max_grad_norm
                    )
                    grad_norm_val = (
                        grad_norm.item()
                        if isinstance(grad_norm, torch.Tensor)
                        else float(grad_norm)
                    )

                    optimizer.step()
                    current_lr = scheduler.get_last_lr()[0]
                    scheduler.step()

                    # Step-wise TensorBoard logging
                    writer.add_scalar("step/train_loss", loss.item(), global_step)
                    writer.add_scalar("step/hinge_loss", loss_dict["hinge"].item(), global_step)
                    writer.add_scalar(
                        "step/smoothness_loss", loss_dict["smoothness"].item(), global_step
                    )
                    writer.add_scalar(
                        "step/sparsity_loss", loss_dict["sparsity"].item(), global_step
                    )
                    writer.add_scalar("step/learning_rate", current_lr, global_step)
                    writer.add_scalar("step/grad_norm", grad_norm_val, global_step)

                    global_step += 1

                    epoch_train_loss += loss.item()
                    epoch_train_hinge += loss_dict["hinge"].item()
                    epoch_train_smooth += loss_dict["smoothness"].item()
                    epoch_train_sparse += loss_dict["sparsity"].item()
                    epoch_grad_norm += grad_norm_val
                    batches_processed += 1

                # Validation phase
                self.model.eval()
                val_loss_total = 0.0
                val_hinge_total = 0.0
                val_smooth_total = 0.0
                val_sparse_total = 0.0
                num_val_items = max(1, len(val_anomal))

                with torch.no_grad():
                    for va_file, vn_file in zip(val_anomal, val_normal):
                        fa = _load_features(va_file).unsqueeze(0).to(self.device)
                        fn = _load_features(vn_file).unsqueeze(0).to(self.device)

                        ya = self.model(fa)
                        yn = self.model(fn)

                        v_loss = criterion(ya, yn)
                        val_loss_total += v_loss["total"].item()
                        val_hinge_total += v_loss["hinge"].item()
                        val_smooth_total += v_loss["smoothness"].item()
                        val_sparse_total += v_loss["sparsity"].item()

                n_batches = max(1, batches_processed)
                avg_train_loss = epoch_train_loss / n_batches
                avg_train_hinge = epoch_train_hinge / n_batches
                avg_train_smooth = epoch_train_smooth / n_batches
                avg_train_sparse = epoch_train_sparse / n_batches
                avg_grad_norm = epoch_grad_norm / n_batches


                avg_val_loss = val_loss_total / num_val_items
                avg_val_hinge = val_hinge_total / num_val_items
                avg_val_smooth = val_smooth_total / num_val_items
                avg_val_sparse = val_sparse_total / num_val_items

                # Epoch-wise TensorBoard logging
                writer.add_scalar("epoch/train_loss", avg_train_loss, epoch_idx + 1)
                writer.add_scalar("epoch/train_hinge", avg_train_hinge, epoch_idx + 1)
                writer.add_scalar("epoch/train_smoothness", avg_train_smooth, epoch_idx + 1)
                writer.add_scalar("epoch/train_sparsity", avg_train_sparse, epoch_idx + 1)
                writer.add_scalar("epoch/val_loss", avg_val_loss, epoch_idx + 1)
                writer.add_scalar("epoch/val_hinge", avg_val_hinge, epoch_idx + 1)
                writer.add_scalar("epoch/val_smoothness", avg_val_smooth, epoch_idx + 1)
                writer.add_scalar("epoch/val_sparsity", avg_val_sparse, epoch_idx + 1)
                writer.add_scalar("epoch/learning_rate", current_lr, epoch_idx + 1)
                writer.add_scalar("epoch/avg_grad_norm", avg_grad_norm, epoch_idx + 1)

                writer.add_scalars(
                    "epoch/comparison",
                    {"train_loss": avg_train_loss, "val_loss": avg_val_loss},
                    epoch_idx + 1,
                )
                writer.flush()

                new_best = avg_val_loss < best_loss
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"Fold {fold_num} | Epoch {epoch_idx + 1:03d}/{self.epochs} - "
                    f"Train Loss: {avg_train_loss:.4f} (Hinge: {avg_train_hinge:.4f}) | "
                    f"Val Loss: {avg_val_loss:.4f} {'*' if new_best else ''}"
                )

                checkpoint = {
                    "fold": fold_num,
                    "epoch": epoch_idx + 1,
                    "val_loss": avg_val_loss,
                    "state_dict": self.model.state_dict(),
                }

                if new_best:
                    best_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(
                        checkpoint,
                        os.path.join(self.save_dir, f"best_loss_fold_{fold_num}.pt"),
                    )
                else:
                    patience_counter += 1

                torch.save(
                    checkpoint,
                    os.path.join(self.save_dir, f"last_fold_{fold_num}.pt"),
                )

                if patience_counter >= self.early_stop_patience:
                    print(
                        f"Early stopping fold {fold_num} at epoch {epoch_idx + 1} "
                        f"(no val improvement for {self.early_stop_patience} epochs)"
                    )
                    break

            # Fold-wise metrics
            writer.add_scalar("fold/best_val_loss", best_loss, fold_num)
            writer.flush()
            writer.close()

            fold_best_losses.append(best_loss)
            print(f"--- Fold {fold_num} Completed. Best Val Loss: {best_loss:.6f}")

        avg_all_folds = sum(fold_best_losses) / len(fold_best_losses)
        print(f"\nAll Folds Completed! Average Best Val Loss: {avg_all_folds:.6f}")

        return {
            "fold_best_losses": fold_best_losses,
            "average_val_loss": avg_all_folds,
        }

"""Modular, DRY PyTorch Trainers for Video Anomaly Detection (BaseTrainer, BinaryTrainer, MultiClassTrainer)."""

from __future__ import annotations

import copy
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from evaluation.metrics import (
    compute_multiclass_auc,
    compute_multiclass_map,
    compute_pr_auc,
    compute_roc_auc,
)
from .loss import BaseMILLoss, BinaryLoss, MultiClassLoss


def _get_dataset_attr(dataset_or_loader: Any, attr: str, default: Any = None) -> Any:
    """Safely extracts attribute even when wrapped in DataLoader or torch.utils.data.Subset."""
    obj = dataset_or_loader
    if isinstance(obj, DataLoader):
        obj = obj.dataset
    val = getattr(obj, attr, None)
    if val is not None:
        return val
    inner = getattr(obj, "dataset", None)
    if inner is not None:
        return getattr(inner, attr, default)
    return default


class BaseTrainer:
    """Foundational Trainer implementing common training loop, logging, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[BaseMILLoss] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epochs: int = 25,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 0.001,
        early_stop_patience: int = 10,
        gradient_clip: float = 1.0,
        device: Optional[Union[str, torch.device]] = None,
        save_dir: str = "checkpoints",
        tensorboard_dir: Optional[str] = None,
        fps: Optional[float] = None,
        clip_size: Optional[int] = None,
        stride: Optional[int] = None,
        overlap: Optional[float] = None,
        num_segments: Optional[int] = None,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.loss_fn = loss_fn if loss_fn is not None else BinaryLoss()
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.early_stop_patience = int(early_stop_patience)
        self.gradient_clip = float(gradient_clip)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Spatio-temporal invariant overrides (if explicitly passed to Trainer)
        self.fps = float(fps) if fps is not None else None
        self.clip_size = int(clip_size) if clip_size is not None else None
        self.stride = int(stride) if stride is not None else None
        self.overlap = float(overlap) if overlap is not None else None
        self.num_segments = int(num_segments) if num_segments is not None else None

        self.tensorboard_dir = tensorboard_dir
        self.writer = None
        if self.tensorboard_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(self.tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

        # Optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        # Scheduler
        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=1e-6
            )

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_bce": [],
            "train_smoothness": [],
            "train_sparsity": [],
            "val_primary_metric": [],
        }
        self.best_metric: float = -float("inf")
        self.best_epoch: int = 0
        self.best_model_weights: Optional[Dict[str, Any]] = None

    def _prepare_loader(self, dataset_or_loader: Union[Dataset, DataLoader], shuffle: bool = True) -> DataLoader:
        """Converts Dataset to DataLoader if needed."""
        if isinstance(dataset_or_loader, DataLoader):
            return dataset_or_loader
        return DataLoader(
            dataset_or_loader,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Runs one full training epoch with gradient clipping."""
        self.model.train()
        epoch_losses: Dict[str, List[float]] = {
            "loss": [],
            "bce": [],
            "smoothness": [],
            "sparsity": [],
        }

        for batch in loader:
            features = batch[0].to(self.device)
            targets = batch[1].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(features)
            loss_dict = self.loss_fn(logits, targets)

            loss = loss_dict["loss"]
            loss.backward()

            if self.gradient_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)

            self.optimizer.step()

            for k in epoch_losses:
                if k in loss_dict:
                    epoch_losses[k].append(float(loss_dict[k].item()))

        return {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_losses.items()}

    def _validate_epoch(self, loader: DataLoader) -> Dict[str, Any]:
        """Abstract validation step to be implemented by BinaryTrainer / MultiClassTrainer."""
        raise NotImplementedError("Subclasses must implement _validate_epoch.")

    def fit(
        self,
        train_dataset: Union[Dataset, DataLoader],
        val_dataset: Optional[Union[Dataset, DataLoader]] = None,
    ) -> Dict[str, Any]:
        """Full training loop with validation, early stopping, and rich checkpointing."""
        # Auto-configure dataset mode if available
        self._configure_dataset_mode(train_dataset)
        if val_dataset is not None:
            self._configure_dataset_mode(val_dataset)

        train_loader = self._prepare_loader(train_dataset, shuffle=True)
        val_loader = self._prepare_loader(val_dataset, shuffle=False) if val_dataset is not None else None

        print(f"🚀 Starting training on {self.device} for {self.epochs} epochs...")
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_metrics = self._train_epoch(train_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_bce"].append(train_metrics["bce"])
            self.history["train_smoothness"].append(train_metrics["smoothness"])
            self.history["train_sparsity"].append(train_metrics["sparsity"])

            val_metrics = None
            primary_metric_val = 0.0
            metric_label = "val_score"

            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader)
                primary_metric_val = float(val_metrics.get("primary_value", 0.0))
                metric_label = val_metrics.get("primary_metric", "val_score")
                self.history["val_primary_metric"].append(primary_metric_val)

            elapsed = time.time() - t0
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Progress log
            log_str = (
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"Loss: {train_metrics['loss']:.4f} (BCE: {train_metrics['bce']:.4f}) | "
                f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s"
            )
            if val_metrics is not None:
                log_str += f" | {metric_label}: {primary_metric_val * 100:.2f}%"
            print(log_str)

            # TensorBoard logging
            if self.writer is not None:
                self.writer.add_scalar("Loss/total", train_metrics["loss"], epoch)
                self.writer.add_scalar("Loss/bce", train_metrics["bce"], epoch)
                self.writer.add_scalar("Loss/smoothness", train_metrics["smoothness"], epoch)
                self.writer.add_scalar("Loss/sparsity", train_metrics["sparsity"], epoch)
                self.writer.add_scalar("Learning_Rate", current_lr, epoch)
                if val_metrics is not None:
                    self.writer.add_scalar(f"Validation/{metric_label}", primary_metric_val, epoch)
                    if "roc_auc" in val_metrics:
                        self.writer.add_scalar("Validation/ROC_AUC", float(val_metrics["roc_auc"]), epoch)
                    if "pr_auc" in val_metrics:
                        self.writer.add_scalar("Validation/PR_AUC", float(val_metrics["pr_auc"]), epoch)
                    if "mAP" in val_metrics:
                        self.writer.add_scalar("Validation/mAP", float(val_metrics["mAP"]), epoch)
                    if "topk_accuracy" in val_metrics:
                        for k_t, acc_v in val_metrics["topk_accuracy"].items():
                            self.writer.add_scalar(f"Validation/Top_{k_t}_Acc", float(acc_v), epoch)

            # Checkpoint & Early stopping
            is_best = False
            if val_metrics is not None:
                if primary_metric_val > self.best_metric:
                    self.best_metric = primary_metric_val
                    self.best_epoch = epoch
                    self.best_model_weights = copy.deepcopy(self.model.state_dict())
                    is_best = True
                    patience_counter = 0
                else:
                    patience_counter += 1

            # Save checkpoint with all losses, architecture, and dataset metadata
            self.save_checkpoint(
                epoch=epoch,
                is_best=is_best,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                dataset=(train_dataset.dataset if isinstance(train_dataset, DataLoader) else train_dataset),
            )

            if val_loader is not None and patience_counter >= self.early_stop_patience:
                print(f"🛑 Early stopping triggered at epoch {epoch} (Best: {metric_label}={self.best_metric * 100:.2f}% at epoch {self.best_epoch})")
                break

        if self.writer is not None:
            self.writer.close()

        print(f"🎉 Training complete! Best model at epoch {self.best_epoch} with {self.best_metric * 100:.2f}%")
        return {
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "history": self.history,
        }

    def _configure_dataset_mode(self, dataset_or_loader: Union[Dataset, DataLoader]) -> None:
        """Hook for subclasses to configure dataset mode before training."""
        pass

    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        train_metrics: Optional[Dict[str, float]] = None,
        val_metrics: Optional[Dict[str, Any]] = None,
        dataset: Optional[Dataset] = None,
    ) -> str:
        """Saves rich checkpoint containing model state, FC layers, hyperparams, losses, and dataset metadata."""
        train_metrics = train_metrics or {}

        # 1. Model architecture configuration
        hidden_dims = getattr(self.model, "hidden_dims", (512, 256))
        dropout_rates = getattr(self.model, "dropout_rates", (0.5, 0.3))
        in_features = getattr(self.model, "in_features", 768)
        num_classes = getattr(self.model, "num_classes", 1)
        backbone_name = getattr(self.model, "backbone_name", None)

        model_config = {
            "in_features": in_features,
            "num_classes": num_classes,
            "hidden_dims": hidden_dims,
            "dropout_rates": dropout_rates,
            "model_name": self.model.__class__.__name__,
            "backbone_name": backbone_name,
        }

        # Pure, prefix-free AnomalyHead state dict
        if hasattr(self.model, "head") and isinstance(self.model.head, nn.Module):
            head_state_dict = self.model.head.state_dict()
        else:
            head_state_dict = self.model.state_dict()

        # 2. Dataset metadata (pure dataset properties & spatio-temporal invariants)
        dataset_metadata = None
        if dataset is not None:
            ds_fps = _get_dataset_attr(dataset, "fps", None)
            ds_clip_size = _get_dataset_attr(dataset, "clip_size", None)
            ds_stride = _get_dataset_attr(dataset, "stride", None)
            ds_overlap = _get_dataset_attr(dataset, "overlap", None)
            ds_num_segments = _get_dataset_attr(dataset, "num_segments", None)

            dataset_metadata = {
                "dataset_name": _get_dataset_attr(dataset, "dataset_name", "Unknown"),
                "split": _get_dataset_attr(dataset, "split", "Unknown"),
                "mode": _get_dataset_attr(dataset, "mode", "Unknown"),
                # dataset_classes: raw dataset class list (e.g. 13 for UCF-Crime), reference only
                "dataset_classes": _get_dataset_attr(dataset, "classes", None),
                # num_classes: model HEAD's output class count (1=binary, 4=combined-mc, 6=xdv-mc, 13=ucf-mc)
                "num_classes": num_classes,
                "feature_dim": in_features,
                "num_segments": self.num_segments if self.num_segments is not None else ds_num_segments,
                "fps": self.fps if self.fps is not None else ds_fps,
                "clip_size": self.clip_size if self.clip_size is not None else ds_clip_size,
                "stride": self.stride if self.stride is not None else ds_stride,
                "overlap": self.overlap if self.overlap is not None else ds_overlap,
            }


        # 3. Evaluation metadata (strictly separated from dataset properties)
        eval_metadata = None
        if val_metrics is not None:
            eval_metadata = {
                "iou_threshold": getattr(self, "iou_threshold", None),
                "metrics": val_metrics,
            }

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "head_state_dict": head_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_metric": self.best_metric,
            # Explicit loss components
            "loss": float(train_metrics.get("loss", 0.0)),
            "bce_loss": float(train_metrics.get("bce", 0.0)),
            "smoothness_loss": float(train_metrics.get("smoothness", 0.0)),
            "sparsity_loss": float(train_metrics.get("sparsity", 0.0)),
            "val_metrics": val_metrics,
            # Model architecture details for AnomalyHead.load_from_checkpoint()
            "config": model_config,
            "hidden_dims": hidden_dims,
            "dropout_rates": dropout_rates,
            "in_features": in_features,
            "num_classes": num_classes,
            # Dataset & Evaluation provenance
            "dataset_metadata": dataset_metadata,
            "eval_metadata": eval_metadata,
            "history": self.history,
        }

        latest_path = os.path.join(self.save_dir, "latest.pt")
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            return best_path

        return latest_path


class BinaryTrainer(BaseTrainer):
    """Trainer specialized for single-score binary video anomaly detection."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[BinaryLoss] = None,
        k_top: int = 3,
        smoothness_weight: float = 0.0001,
        sparsity_weight: float = 0.0001,
        **kwargs: Any,
    ) -> None:
        if loss_fn is None:
            loss_fn = BinaryLoss(
                k=k_top,
                smoothness_weight=smoothness_weight,
                sparsity_weight=sparsity_weight,
            )
        super().__init__(model=model, loss_fn=loss_fn, **kwargs)

    def _configure_dataset_mode(self, dataset_or_loader: Union[Dataset, DataLoader]) -> None:
        """Ensures dataset operates in 'binary' mode, resolving Subsets transparently."""
        ds = dataset_or_loader.dataset if isinstance(dataset_or_loader, DataLoader) else dataset_or_loader
        if hasattr(ds, "set_mode"):
            ds.set_mode("binary")
        elif hasattr(ds, "dataset") and hasattr(ds.dataset, "set_mode"):
            ds.dataset.set_mode("binary")

    @torch.no_grad()
    def _validate_epoch(self, loader: DataLoader) -> Dict[str, Any]:
        """Evaluates binary ROC-AUC and PR-AUC."""
        self.model.eval()
        all_preds = []
        all_targets = []

        for batch in loader:
            features = batch[0].to(self.device)
            targets = batch[1].cpu().reshape(-1)

            logits = self.model(features)
            probs = torch.sigmoid(logits)

            # Bag-level score: max segment probability
            if probs.ndim == 3:
                bag_scores = probs.squeeze(-1).max(dim=1).values
            elif probs.ndim == 2:
                if probs.shape[-1] == 1:
                    bag_scores = probs.squeeze(-1)
                else:
                    bag_scores = probs.max(dim=1).values
            else:
                bag_scores = probs

            all_preds.append(bag_scores.cpu().numpy())
            all_targets.append(targets.numpy())

        y_score = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)

        roc_auc = compute_roc_auc(y_true, y_score)
        pr_auc = compute_pr_auc(y_true, y_score)

        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "primary_metric": "ROC-AUC",
            "primary_value": roc_auc,
        }


class MultiClassTrainer(BaseTrainer):
    """Trainer specialized for multi-class video anomaly detection with explicit IoU-aware mAP."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[MultiClassLoss] = None,
        k_top: int = 1,
        smoothness_weight: float = 0.0001,
        sparsity_weight: float = 0.0001,
        pos_weight: float = 3.0,
        iou_threshold: float = 0.30,
        **kwargs: Any,
    ) -> None:
        model_num_classes = getattr(model, "num_classes", None)
        if model_num_classes is None and hasattr(model, "head"):
            model_num_classes = getattr(model.head, "num_classes", None)
        if model_num_classes == 1:
            raise ValueError(
                "Model has num_classes=1 (single binary anomaly score). "
                "For binary anomaly detection, please use 'BinaryTrainer' instead of 'MultiClassTrainer'."
            )

        if loss_fn is None:
            loss_fn = MultiClassLoss(
                k=k_top,
                smoothness_weight=smoothness_weight,
                sparsity_weight=sparsity_weight,
                pos_weight=pos_weight,
            )
        super().__init__(model=model, loss_fn=loss_fn, **kwargs)
        self.iou_threshold = float(iou_threshold)

    def _configure_dataset_mode(self, dataset_or_loader: Union[Dataset, DataLoader]) -> None:
        """Ensures dataset operates in 'multiclass' mode, preserving exact_match if configured."""
        ds = dataset_or_loader.dataset if isinstance(dataset_or_loader, DataLoader) else dataset_or_loader
        current_mode = getattr(ds, "mode", None)
        if current_mode is None and hasattr(ds, "dataset"):
            current_mode = getattr(ds.dataset, "mode", None)
        if current_mode in ("exact_match", "exact_matching", "exact", "intersection"):
            return
        if current_mode == "binary":
            raise ValueError(
                "Dataset is configured in 'binary' mode. "
                "For binary anomaly detection, please use 'BinaryTrainer' instead of 'MultiClassTrainer'."
            )

        if hasattr(ds, "set_mode"):
            ds.set_mode("multiclass")
        elif hasattr(ds, "dataset") and hasattr(ds.dataset, "set_mode"):
            ds.dataset.set_mode("multiclass")

    @torch.no_grad()
    def _validate_epoch(self, loader: DataLoader) -> Dict[str, Any]:
        """Evaluates mAP at explicit iou_threshold and macro ROC-AUC."""
        self.model.eval()
        all_preds = []
        all_targets = []

        classes = _get_dataset_attr(loader, "classes", None)

        for batch in loader:
            features = batch[0].to(self.device)
            targets = batch[1].cpu()

            logits = self.model(features)
            probs = torch.sigmoid(logits)

            # [B, T, C] -> pool max segment score per class: [B, C]
            if probs.ndim == 3:
                bag_scores = probs.max(dim=1).values
            else:
                bag_scores = probs

            all_preds.append(bag_scores.cpu().numpy())
            all_targets.append(targets.numpy())

        y_score = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)

        map_res = compute_multiclass_map(y_true, y_score, class_names=classes, iou_threshold=self.iou_threshold)
        auc_res = compute_multiclass_auc(y_true, y_score, class_names=classes)

        return {
            "mAP": map_res["mAP"],
            "iou_threshold": self.iou_threshold,
            "metric_name": map_res["metric_name"],
            "macro_auc": auc_res["macro_auc"],
            "per_class_ap": map_res["per_class"],
            "per_class_auc": auc_res["per_class"],
            "primary_metric": map_res["metric_name"],
            "primary_value": map_res["mAP"],
        }

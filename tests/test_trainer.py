"""Unit tests for Trainer, training loop, and rich checkpointing."""

import os
import tempfile
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytest

from model import AnomalyDetector, AnomalyHead
from train.trainer import BinaryTrainer, MultiClassTrainer, _get_dataset_attr


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, mode="multiclass"):
        self.features = torch.randn(10, 32, 768)
        self.mode = mode
        self.fps = 24.0
        self.clip_size = 16
        self.stride = 16
        self.overlap = 0.5
        self.num_segments = 32
        self.classes = ["Violence_Affray", "Accidents_Disasters", "Gun_Violence", "Property_Crimes"]
        self.num_classes = 4
        self.ground_truth_intervals = {"video_0": [(1.0, 5.0)], "video_2": [(2.0, 6.0)]}

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        if self.mode == "binary":
            target = torch.tensor(1.0 if idx % 2 == 0 else 0.0)
        else:
            target = torch.zeros(4)
            if idx % 2 == 0:
                target[idx % 4] = 1.0
        return self.features[idx], target, f"video_{idx}"


def test_multiclass_trainer_and_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = AnomalyHead(in_features=768, num_classes=4)
        ds = DummyDataset(mode="multiclass")

        trainer = MultiClassTrainer(
            model=model,
            epochs=2,
            batch_size=4,
            save_dir=tmpdir,
            device="cpu",
        )

        res = trainer.fit(train_dataset=ds, val_dataset=ds)
        assert len(res["history"]["train_loss"]) == 2

        # Check saved checkpoint
        best_ckpt_path = os.path.join(tmpdir, "best_model.pt")
        assert os.path.isfile(best_ckpt_path)

        # Verify canonical keys: head_state_dict present, backbone_state_dict absent
        raw_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in raw_ckpt
        assert "head_state_dict" in raw_ckpt
        assert "backbone_state_dict" not in raw_ckpt

        # Load with AnomalyDetector
        detector, ckpt = AnomalyDetector.load_from_checkpoint(best_ckpt_path)
        assert detector.num_classes == 4
        assert detector.video_processor.target_fps == 24.0
        assert detector.video_processor.clip_size == 16


def test_binary_trainer_and_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = AnomalyHead(in_features=768, num_classes=1)
        ds = DummyDataset(mode="binary")

        trainer = BinaryTrainer(
            model=model,
            epochs=1,
            batch_size=4,
            save_dir=tmpdir,
            device="cpu",
        )

        res = trainer.fit(train_dataset=ds, val_dataset=ds)
        assert len(res["history"]["train_loss"]) == 1

        best_ckpt_path = os.path.join(tmpdir, "best_model.pt")
        raw_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        assert "head_state_dict" in raw_ckpt
        assert "backbone_state_dict" not in raw_ckpt


def test_anomaly_detector_training_and_evaluator_loading():
    """Verify that AnomalyDetector saved checkpoint can be loaded cleanly by Evaluator via head_state_dict."""
    from evaluation.evaluator import Evaluator

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create AnomalyDetector without heavyweight backbone for unit testing
        detector = AnomalyDetector(backbone_name=None, in_features=768, num_classes=4)
        ds = DummyDataset(mode="multiclass")

        trainer = MultiClassTrainer(
            model=detector,
            epochs=1,
            batch_size=4,
            save_dir=tmpdir,
            device="cpu",
        )
        trainer.fit(train_dataset=ds, val_dataset=ds)

        best_ckpt_path = os.path.join(tmpdir, "best_model.pt")
        raw_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in raw_ckpt
        assert "head_state_dict" in raw_ckpt
        assert "backbone_state_dict" not in raw_ckpt

        # Load with AnomalyHead directly
        head, _ = AnomalyHead.load_from_checkpoint(best_ckpt_path, device="cpu")
        assert head.num_classes == 4

        # Evaluator directly loaded from checkpoint path string
        evaluator = Evaluator(
            model=best_ckpt_path,
            dataset_or_loader=ds,
            device="cpu",
            batch_size=4,
        )
        assert isinstance(evaluator.model, AnomalyHead)
        eval_res = evaluator.evaluate(mode="multiclass", print_summary=False, save_report_dir=None)
        assert "num_samples" in eval_res
        assert eval_res["num_samples"] == len(ds)


def test_multiclass_trainer_binary_guard_raises():
    """Verify MultiClassTrainer rejects num_classes=1 and binary mode datasets."""
    # 1. Reject num_classes=1 model
    model_binary = AnomalyHead(in_features=768, num_classes=1)
    with pytest.raises(ValueError, match="num_classes=1"):
        MultiClassTrainer(model=model_binary)

    # 2. Reject binary dataset in fit()
    model_multi = AnomalyHead(in_features=768, num_classes=4)
    trainer = MultiClassTrainer(model=model_multi, epochs=1, batch_size=2)
    binary_ds = DummyDataset(mode="binary")
    with pytest.raises(ValueError, match="binary"):
        trainer.fit(train_dataset=binary_ds, val_dataset=binary_ds)


def test_binary_trainer_and_evaluator_sweep():
    """Verify BinaryTrainer model can be loaded by Evaluator and run threshold sweep."""
    from evaluation.evaluator import Evaluator

    with tempfile.TemporaryDirectory() as tmpdir:
        model = AnomalyHead(in_features=768, num_classes=1)
        ds = DummyDataset(mode="binary")

        trainer = BinaryTrainer(
            model=model,
            epochs=1,
            batch_size=4,
            save_dir=tmpdir,
            device="cpu",
        )
        trainer.fit(train_dataset=ds, val_dataset=ds)

        best_ckpt = os.path.join(tmpdir, "best_model.pt")
        evaluator = Evaluator(
            model=best_ckpt,
            dataset_or_loader=ds,
            device="cpu",
            batch_size=4,
        )
        eval_res = evaluator.evaluate(
            mode="binary",
            sweep_thresholds=True,
            print_summary=False,
            save_report_dir=None,
        )
        assert "ROC-AUC" in eval_res or "roc_auc" in eval_res or "PR-AUC" in eval_res
        assert "threshold_sweep" in eval_res
        assert "best_f1" in eval_res["threshold_sweep"]

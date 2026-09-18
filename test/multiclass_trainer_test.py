import os
import tempfile
import numpy as np
import pytest
import torch

from model.model import FC_head
from training.multiclass_trainer import MultiClassMILTrainer, compute_average_precision


def test_compute_average_precision():
    # All zeros
    assert compute_average_precision(np.zeros(10), np.random.rand(10)) == 0.0

    # Perfect ranking
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])
    assert np.isclose(compute_average_precision(y_true, y_score), 1.0)

    # Inverted ranking
    y_score_inv = np.array([0.1, 0.2, 0.8, 0.9])
    ap_inv = compute_average_precision(y_true, y_score_inv)
    assert ap_inv < 0.5


def test_multiclass_trainer_mini_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        features_dir = os.path.join(tmpdir, "features")
        norm_dir = os.path.join(features_dir, "normal")
        anom_dir = os.path.join(features_dir, "anomal")
        os.makedirs(norm_dir)
        os.makedirs(anom_dir)

        # Create 4 normal and 6 anomalous mock files
        feat_dim = 64
        for i in range(4):
            torch.save(torch.randn(32, feat_dim), os.path.join(norm_dir, f"video_norm_{i}__label_A.pt"))
        for i in range(6):
            cls = "B1-0-0" if i % 2 == 0 else "B2-G-0"
            torch.save(torch.randn(32, feat_dim), os.path.join(anom_dir, f"video_anom_{i}__label_{cls}.pt"))

        save_dir = os.path.join(tmpdir, "checkpoints")
        log_dir = os.path.join(tmpdir, "runs")

        model = FC_head(in_features=feat_dim, num_classes=6, use_sigmoid=False)

        trainer = MultiClassMILTrainer(
            model=model,
            features_dir=features_dir,
            epochs=2,
            batch_size=2,
            k_top=1,
            k_fold=2,
            save_dir=save_dir,
            log_dir=log_dir,
            device="cpu",
        )

        results = trainer.train()

        assert "fold_results" in results
        assert len(results["fold_results"]) == 2
        assert "mean_best_loss" in results
        assert "mean_best_map" in results

        # Check saved checkpoint files
        assert os.path.isfile(os.path.join(save_dir, "best_loss_fold_1.pt"))
        assert os.path.isfile(os.path.join(save_dir, "best_loss_fold_2.pt"))
        assert os.path.isfile(os.path.join(save_dir, "last_fold_1.pt"))
        assert os.path.isfile(os.path.join(save_dir, "last_fold_2.pt"))

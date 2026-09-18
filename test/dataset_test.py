import os
import pytest
import torch
from torch.utils.data import DataLoader

from data.dataset import PairwiseFeatureDataset, _load_features
from model.ranking_head import SegmentRankingHead
from training.trainer import MILTrainer


@pytest.fixture
def dummy_data_dirs(tmp_path):
    anom_dir = tmp_path / "anomalous"
    norm_dir = tmp_path / "normal"
    anom_dir.mkdir()
    norm_dir.mkdir()

    # Create 6 anomalous dummy .pt files
    for i in range(6):
        feats = torch.randn(32, 64)
        torch.save({"feats": feats, "video_id": f"anom_{i}"}, anom_dir / f"anom_{i}.pt")

    # Create 4 normal dummy .pt files (2 as dict, 2 as raw tensor)
    for i in range(2):
        feats = torch.randn(32, 64)
        torch.save({"feats": feats, "video_id": f"norm_{i}"}, norm_dir / f"norm_{i}.pt")
    for i in range(2, 4):
        feats = torch.randn(32, 64)
        torch.save(feats, norm_dir / f"norm_{i}.pt")

    return str(anom_dir), str(norm_dir)


def test_dataset_loading(dummy_data_dirs):
    anom_dir, norm_dir = dummy_data_dirs
    dataset = PairwiseFeatureDataset(anom_dir, norm_dir)

    assert len(dataset) == 6

    anom_f, norm_f = dataset[0]
    assert anom_f.shape == (32, 64)
    assert norm_f.shape == (32, 64)
    assert isinstance(anom_f, torch.Tensor)
    assert isinstance(norm_f, torch.Tensor)


def test_dataset_dataloader(dummy_data_dirs):
    anom_dir, norm_dir = dummy_data_dirs
    dataset = PairwiseFeatureDataset(anom_dir, norm_dir)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    batch_count = 0
    for a_batch, n_batch in loader:
        assert a_batch.shape == (2, 32, 64)
        assert n_batch.shape == (2, 32, 64)
        batch_count += 1

    assert batch_count == 3


def test_trainer_run_mini(dummy_data_dirs, tmp_path):
    anom_dir, norm_dir = dummy_data_dirs
    save_dir = tmp_path / "checkpoints"
    log_dir = tmp_path / "runs"

    model = SegmentRankingHead(input_dim=64, hidden_dims=[32], dropout_rates=[0.1])

    trainer = MILTrainer(
        model=model,
        anomal_dir=anom_dir,
        normal_dir=norm_dir,
        epochs=2,
        batch_size=2,
        k_fold=2,
        save_dir=str(save_dir),
        log_dir=str(log_dir),
    )

    results = trainer.train()

    assert "fold_best_losses" in results
    assert len(results["fold_best_losses"]) == 2
    assert (save_dir / "best_loss_fold_1.pt").exists()
    assert (save_dir / "last_fold_1.pt").exists()
    assert (save_dir / "best_loss_fold_2.pt").exists()
    assert (save_dir / "last_fold_2.pt").exists()
    assert (log_dir / "fold_1").exists()
    assert (log_dir / "fold_2").exists()


def test_video_extraction_dataset_error_handling():
    from data.video_dataset import VideoExtractionDataset
    dataset = VideoExtractionDataset(["non_existent_video_path.mp4"])
    assert len(dataset) == 1
    item = dataset[0]
    assert item["status"] == "error"


def test_video_extraction_dataset_real_video():
    from data.video_dataset import VideoExtractionDataset, collate_extraction
    video_dir = "/home/n3uron/Videos/XD-Violence-test-videos"
    if os.path.isdir(video_dir):
        files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")][:1]
        if files:
            dataset = VideoExtractionDataset(
                files, num_segments=4, clip_size=16, max_clips_per_segment=1
            )
            loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate_extraction)
            for item in loader:
                assert item["status"] == "ok"
                assert len(item["segments"]) == 4
                assert item["segments"][0].shape[2] == 16

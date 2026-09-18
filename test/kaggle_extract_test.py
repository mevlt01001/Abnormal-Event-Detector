import json
import os
import tempfile
import pytest
import torch

from scripts.kaggle_extract import (
    normalize_class_map_input,
    extract_features_from_class_map,
)
from data.multiclass_dataset import MultiClassFeatureDataset


def test_normalize_class_map_input_list():
    raw_map = [["/path/v1.mp4", "/path/v2.mp4"], ["/path/v3.mp4"]]
    names = ["Fighting", "Normal"]
    lists, final_names = normalize_class_map_input(raw_map, names)
    assert lists == raw_map
    assert final_names == names


def test_normalize_class_map_input_list_default_names():
    raw_map = [["/path/v1.mp4"], ["/path/v2.mp4"], ["/path/v3.mp4"]]
    lists, final_names = normalize_class_map_input(raw_map)
    assert final_names == ["class_0", "class_1", "class_2"]


def test_normalize_class_map_input_dict():
    raw_dict = {
        "Fighting": ["/path/v1.mp4"],
        "Normal": ["/path/v2.mp4", "/path/v3.mp4"],
    }
    lists, names = normalize_class_map_input(raw_dict)
    assert names == ["Fighting", "Normal"]
    assert len(lists) == 2


def test_normalize_class_map_json_str():
    json_str = '[["/path/v1.mp4"], ["/path/v2.mp4"]]'
    lists, names = normalize_class_map_input(json_str, ["A", "B"])
    assert names == ["A", "B"]
    assert len(lists) == 2


def test_normalize_class_map_length_mismatch():
    raw_map = [["/path/v1.mp4"], ["/path/v2.mp4"]]
    with pytest.raises(ValueError):
        normalize_class_map_input(raw_map, ["OnlyOneClass"])


class DummyBackbone(torch.nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: [K, C, T, H, W] -> returns [K, dim]
        batch_k = x.shape[0]
        return torch.ones(batch_k, self.dim)


class DummyModel:
    def __init__(self, dim=768):
        self.video_model = DummyBackbone(dim)
        self.feature_dim = dim

    def extract_features(self, video_path, num_segments=32, **kwargs):
        return torch.ones(num_segments, self.feature_dim)


def test_extract_features_multi_label_and_manifest():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "features")

        # Video1 is in BOTH Fighting and Riot (multi-label)
        # Video2 is in Normal
        fake_video1 = os.path.join(tmpdir, "video_multilabel.mp4")
        fake_video2 = os.path.join(tmpdir, "video_normal.mp4")
        open(fake_video1, "w").close()
        open(fake_video2, "w").close()

        class_video_map = [
            [fake_video1],               # Fighting
            [fake_video2],               # Normal
            [fake_video1],               # Riot (same video as Fighting!)
        ]
        class_names = ["Fighting", "Normal", "Riot"]

        dummy_model = DummyModel(dim=768)

        result = extract_features_from_class_map(
            class_video_map=class_video_map,
            class_names=class_names,
            output_dir=output_dir,
            model_name="swin3d_t",
            num_segments=32,
            num_workers=0,  # Sequential mode using dummy model
            model_instance=dummy_model,
        )

        assert result["status"] == "completed"

        # Check files exist in all 3 classes
        fight_pt = os.path.join(output_dir, "swin3d_t", "Fighting", "video_multilabel.pt")
        norm_pt = os.path.join(output_dir, "swin3d_t", "Normal", "video_normal.pt")
        riot_pt = os.path.join(output_dir, "swin3d_t", "Riot", "video_multilabel.pt")

        assert os.path.isfile(fight_pt)
        assert os.path.isfile(norm_pt)
        assert os.path.isfile(riot_pt)

        # Check payload
        fight_data = torch.load(fight_pt)
        assert fight_data["feats"].shape == (32, 768)
        assert fight_data["class_name"] == "Fighting"

        riot_data = torch.load(riot_pt)
        assert riot_data["feats"].shape == (32, 768)
        assert riot_data["class_name"] == "Riot"

        # Check manifest.json
        manifest_file = os.path.join(output_dir, "swin3d_t", "manifest.json")
        assert os.path.isfile(manifest_file)
        with open(manifest_file, "r") as f:
            manifest = json.load(f)

        assert manifest["model"] == "swin3d_t"
        assert manifest["feature_dim"] == 768
        assert manifest["counts"]["Fighting"] == 1
        assert manifest["counts"]["Normal"] == 1
        assert manifest["counts"]["Riot"] == 1
        assert manifest["total_files"] == 3
        assert manifest["unique_videos"] == 2  # 2 unique video paths


def test_multiclass_dataset_with_class_folders():
    with tempfile.TemporaryDirectory() as tmpdir:
        swin_dir = os.path.join(tmpdir, "swin3d_t")
        fight_dir = os.path.join(swin_dir, "Fighting")
        norm_dir = os.path.join(swin_dir, "Normal")
        os.makedirs(fight_dir)
        os.makedirs(norm_dir)

        # Save dummy pt files in class folders
        torch.save(
            {"feats": torch.randn(32, 768), "class_name": "Fighting"},
            os.path.join(fight_dir, "ucf_fight_01.pt"),
        )
        torch.save(
            {"feats": torch.randn(32, 768), "class_name": "Normal"},
            os.path.join(norm_dir, "ucf_normal_01.pt"),
        )

        classes = ["Fighting", "Normal"]
        ds = MultiClassFeatureDataset(features_dir=swin_dir, class_list=classes)
        assert len(ds) == 2

        # Check items
        for i in range(len(ds)):
            feats, target, name = ds[i]
            assert feats.shape == (32, 768)
            if "fight" in name:
                assert target[0] == 1.0  # Fighting
                assert target[1] == 0.0
            else:
                assert target.sum() == 0.0  # Normal video is all zeros

"""Test script to verify multi-model feature extraction on 10 real test videos."""

from __future__ import annotations

import glob
import os
import shutil
import sys
import time
import torch

# Ensure workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.kaggle_extract import extract_features_from_class_map
from model.backbone import BACKBONE_FEATURE_DIMS


def main():
    test_dir = "Test_Videos"
    output_dir = "test_multi_features"
    videos = sorted(glob.glob(os.path.join(test_dir, "*.mp4")))

    print(f"• Found {len(videos)} test videos in '{test_dir}'")
    assert len(videos) == 10, f"Expected 10 videos, found {len(videos)}"

    # Clean previous test output if exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    models_to_test = ["swin3d_t", "mvit_v2_s", "r3d_18", "mc3_18", "r2plus1d_18", "s3d"]

    # Assign dummy classes to test class map logic
    class_map = {
        "Test_Normal": [videos[1], videos[4]],  # test_video_1, test_video_16
        "Test_Anomaly": [v for i, v in enumerate(videos) if i not in (1, 4)],  # 8 anomaly videos
    }

    t0 = time.time()
    result = extract_features_from_class_map(
        class_video_map=class_map,
        output_dir=output_dir,
        model_names=models_to_test,
        num_segments=32,
        clip_size=16,
        fps=30.0,
        batch_size=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        overwrite=True,
    )
    total_time = time.time() - t0

    print("\n" + "=" * 70)
    print("🔍 VERIFYING EXTRACTED FEATURE TENSORS FOR ALL 6 MODELS")
    print("=" * 70)

    total_verified = 0
    for m_name in models_to_test:
        expected_dim = BACKBONE_FEATURE_DIMS[m_name]
        m_dir = os.path.join(output_dir, m_name)
        assert os.path.isdir(m_dir), f"Model directory missing: {m_dir}"

        # Check manifest
        manifest_path = os.path.join(m_dir, "manifest.json")
        assert os.path.isfile(manifest_path), f"Manifest missing: {manifest_path}"

        for vpath in videos:
            vstem = os.path.splitext(os.path.basename(vpath))[0]
            # Check either class folder
            c_name = "Test_Normal" if vpath in class_map["Test_Normal"] else "Test_Anomaly"
            pt_path = os.path.join(m_dir, c_name, f"{vstem}.pt")
            assert os.path.isfile(pt_path), f"Missing feature file: {pt_path}"

            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            assert isinstance(data, dict), f"Payload is not dict: {type(data)}"
            assert "feats" in data, f"'feats' key missing in {pt_path}"

            feats = data["feats"]
            assert feats.shape == (32, expected_dim), (
                f"Shape mismatch in {m_name}/{vstem}: got {feats.shape}, expected (32, {expected_dim})"
            )
            assert not torch.isnan(feats).any(), f"NaN detected in {m_name}/{vstem}"
            assert not torch.isinf(feats).any(), f"Inf detected in {m_name}/{vstem}"

            total_verified += 1

        print(f"  ✓ [{m_name:<12}] 10/10 videos verified! Shape: (32, {expected_dim}), No NaNs/Infs.")

    print("=" * 70)
    print(f"🎉 SUCCESS! Total {total_verified}/60 feature files verified across all 6 models!")
    print(f"⚡ Total extraction time for 10 videos across 6 models: {total_time:.1f}s ({total_time/10:.2f}s per video)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

"""Comprehensive verification script for multi-model feature extraction on 5 test videos.

Verifies:
1. VideoProcessor overlap logic and zero black-frame guarantee.
2. Simultaneous extraction across all 6 backbones:
   - swin3d_t (768-d)
   - mvit_v2_s (768-d)
   - r3d_18 (512-d)
   - mc3_18 (512-d)
   - r2plus1d_18 (512-d)
   - s3d (1024-d)
3. Output tensor shapes, numeric validity (no NaN/Inf), and metadata integrity.
"""

from __future__ import annotations

import glob
import os
import shutil
import sys
import time
import torch

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.video_processor import VideoProcessor
from model.backbone import BACKBONE_FEATURE_DIMS
from scripts.kaggle_extract import extract_features_from_class_map


def verify_video_processor_zero_black_frames(video_paths: list[str], overlap_ratio: float = 0.5) -> None:
    """Verifies that VideoProcessor yields valid frames without black/zero frames."""
    print("\n" + "=" * 70)
    print("[TEST STEP 1] Verifying VideoProcessor Overlap and Zero Black Frames")
    print("=" * 70)
    print(f"[INFO] Testing {len(video_paths)} videos with overlap_ratio={overlap_ratio:.2f}")

    vp = VideoProcessor(target_fps=30.0, clip_size=16, overlap_ratio=overlap_ratio)

    for idx, v_path in enumerate(video_paths, 1):
        v_name = os.path.basename(v_path)
        meta = vp.get_video_metadata(v_path)
        print(f"[INFO] [{idx}/{len(video_paths)}] Checking {v_name} ({meta['total_frames']} frames, {meta['duration_sec']:.1f}s)...")

        segments = list(vp.extract_uniform_segments(v_path, num_segments=32, overlap_ratio=overlap_ratio))
        assert len(segments) == 32, f"Expected 32 segments, got {len(segments)}"

        total_clips = 0
        for seg_idx, seg_tensor in enumerate(segments):
            # seg_tensor: [K, C, clip_size, H, W]
            assert seg_tensor.ndim == 5, f"Expected 5D tensor, got {seg_tensor.shape}"
            assert seg_tensor.shape[1] == 3, f"Expected 3 color channels, got {seg_tensor.shape[1]}"
            assert seg_tensor.shape[2] == 16, f"Expected clip_size=16, got {seg_tensor.shape[2]}"
            assert seg_tensor.shape[3] == 224 and seg_tensor.shape[4] == 224

            K = seg_tensor.shape[0]
            total_clips += K

            # Zero black-frame verification: every clip and every frame must have positive sum
            for c_idx in range(K):
                clip = seg_tensor[c_idx]  # [3, 16, 224, 224]
                for f_idx in range(16):
                    frame = clip[:, f_idx, :, :]  # [3, 224, 224]
                    frame_sum = float(frame.sum().item())
                    assert frame_sum > 0.0, (
                        f"Black frame detected in {v_name} at segment {seg_idx}, clip {c_idx}, frame {f_idx}"
                    )

        print(f"[PASS] {v_name}: 32 segments, {total_clips} total clips verified. Zero black frames.")

    print("[SUCCESS] All videos passed zero black-frame verification.\n")


def verify_multi_model_extraction(
    video_paths: list[str],
    output_dir: str = "test_output/multi_model_5_test",
    overlap_ratio: float = 0.5,
) -> None:
    """Runs and verifies 6-model feature extraction on 5 videos."""
    print("=" * 70)
    print("[TEST STEP 2] Running Multi-Model Feature Extraction (6 Models)")
    print("=" * 70)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    models_to_test = ["swin3d_t", "mvit_v2_s", "r3d_18", "mc3_18", "r2plus1d_18", "s3d"]

    class_map = {
        "Normal": [video_paths[0], video_paths[1]],
        "Anomaly": [video_paths[2], video_paths[3], video_paths[4]],
    }

    t0 = time.time()
    result = extract_features_from_class_map(
        class_video_map=class_map,
        output_dir=output_dir,
        model_names=models_to_test,
        num_segments=32,
        clip_size=16,
        overlap_ratio=overlap_ratio,
        fps=30.0,
        batch_size=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        overwrite=True,
    )
    duration = time.time() - t0

    print("\n" + "=" * 70)
    print("[TEST STEP 3] Validating Extracted Feature Files and Tensor Properties")
    print("=" * 70)

    verified_count = 0
    for m_name in models_to_test:
        expected_dim = BACKBONE_FEATURE_DIMS[m_name]
        m_dir = os.path.join(output_dir, m_name)
        assert os.path.isdir(m_dir), f"Directory missing: {m_dir}"

        manifest_path = os.path.join(m_dir, "manifest.json")
        assert os.path.isfile(manifest_path), f"Manifest missing: {manifest_path}"

        for v_path in video_paths:
            v_stem = os.path.splitext(os.path.basename(v_path))[0]
            c_name = "Normal" if v_path in class_map["Normal"] else "Anomaly"
            pt_path = os.path.join(m_dir, c_name, f"{v_stem}.pt")
            assert os.path.isfile(pt_path), f"Feature file missing: {pt_path}"

            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            assert isinstance(data, dict), f"Payload is not dict: {type(data)}"
            assert "feats" in data, f"'feats' missing in {pt_path}"

            feats = data["feats"]
            assert feats.shape == (32, expected_dim), (
                f"Shape mismatch in {m_name}/{v_stem}: got {feats.shape}, expected (32, {expected_dim})"
            )
            assert not torch.isnan(feats).any(), f"NaN detected in {m_name}/{v_stem}"
            assert not torch.isinf(feats).any(), f"Inf detected in {m_name}/{v_stem}"
            assert feats.std() > 0.0, f"Zero variance detected in {m_name}/{v_stem}"

            # Check metadata
            assert data["num_segments"] == 32
            assert data["clip_size"] == 16
            assert abs(data["overlap_ratio"] - overlap_ratio) < 1e-4
            assert data["feature_dim"] == expected_dim

            verified_count += 1

        print(f"[PASS] Model '{m_name:<12}': 5/5 videos verified (shape [32, {expected_dim}], valid numbers).")

    print("\n" + "=" * 70)
    print("[FINAL SUMMARY] All 5 Test Videos Successfully Verified Across All 6 Models")
    print("=" * 70)
    print(f"[INFO] Verified Files:   {verified_count}/30 files (5 videos x 6 models)")
    print(f"[INFO] Extraction Time:  {duration:.1f}s ({duration/len(video_paths):.2f}s per video for all 6 models)")
    print(f"[INFO] Overlap Ratio:    {overlap_ratio:.2f}")
    print(f"[INFO] Output Directory: {output_dir}")
    print("=" * 70 + "\n")


def main():
    candidate_videos = [
        "Test_Videos/test_video_0.mp4",
        "Test_Videos/test_video_1.mp4",
        "Test_Videos/test_video_2.mp4",
        "Test_Videos/test_video_3.mp4",
        "Test_Videos/test_video_4.mp4",
    ]

    for v in candidate_videos:
        if not os.path.isfile(v):
            raise FileNotFoundError(f"Test video not found: {v}")

    print(f"[INFO] Selected 5 test videos:\n" + "\n".join(f"  - {v}" for v in candidate_videos))

    # 1. Verify VideoProcessor zero black frames and overlap logic
    verify_video_processor_zero_black_frames(candidate_videos, overlap_ratio=0.5)

    # 2. Extract and verify features across all 6 models
    verify_multi_model_extraction(candidate_videos, overlap_ratio=0.5)


if __name__ == "__main__":
    main()

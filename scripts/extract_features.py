#!/usr/bin/env python3
"""Feature extraction script for video anomaly detection datasets.

Extracts 32-segment (or custom) feature representations from video files
using 3D ConvNet backbones (e.g. R3D-18) and saves them as .pt tensors.
"""

from __future__ import annotations

import argparse
import os
import sys
import random
import time
from typing import List

# Add repository root to python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from tqdm import tqdm

from model.model import Model
from utils.annotation_parser import strip_video_ext



def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract video features for Sultani MIL Anomaly Detection"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="/home/n3uron/Videos/XD-Violence-test-videos",
        help="Directory containing video files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/extracted_features",
        help="Directory where .pt feature files will be saved",
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=25,
        help="Number of random videos to extract features from (-1 for all)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="r3d_18",
        help="Backbone model name (r3d_18, mc3_18, s3d, etc.)",
    )
    parser.add_argument(
        "--num_segments",
        type=int,
        default=32,
        help="Number of temporal segments per video",
    )
    parser.add_argument(
        "--clip_size",
        type=int,
        default=16,
        help="Number of frames per clip",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target frame rate for resampling",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for 3D model inference",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for video selection",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run extraction on (cuda or cpu)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of background DataLoader workers for parallel video decoding (0 for sequential)",
    )
    parser.add_argument(
        "--max_clips_per_segment",
        type=int,
        default=2,
        help="Max 16-frame clips sampled per segment to prevent slowdown on long videos (None for all)",
    )
    parser.add_argument(
        "--auto_classify",
        action="store_true",
        default=True,
        help="Automatically split XD-Violence into normal/ and anomal/ subfolders based on _label_A",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted feature files",
    )
    return parser.parse_args()


def get_video_files(video_dir: str) -> List[str]:
    valid_exts = {".mp4", ".avi", ".mkv", ".mov"}
    video_files = [
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ]
    return sorted(video_files)


def main():
    args = parse_args()
    print(f"=== Video Feature Extraction ===")
    print(f"Video Directory: {args.video_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Model Backbone:  {args.model_name}")
    print(f"Device:          {args.device}")
    print(f"Num Segments:    {args.num_segments}")
    print(f"Clip Size:       {args.clip_size}")
    print(f"Target FPS:      {args.fps}")

    if not os.path.isdir(args.video_dir):
        raise NotADirectoryError(f"Video directory not found: {args.video_dir}")

    all_videos = get_video_files(args.video_dir)
    print(f"Total videos found in directory: {len(all_videos)}")

    if not all_videos:
        print("No videos found to process!")
        return

    # Select random videos
    if args.num_videos > 0 and args.num_videos < len(all_videos):
        random.seed(args.seed)
        selected_videos = random.sample(all_videos, args.num_videos)
    else:
        selected_videos = all_videos

    print(f"Selected {len(selected_videos)} videos for feature extraction.")

    # Initialize model
    print(f"Loading {args.model_name} model...")
    model = Model(base_model_name=args.model_name).to(args.device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.auto_classify:
        os.makedirs(os.path.join(args.output_dir, "normal"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "anomal"), exist_ok=True)

    from data.video_dataset import VideoExtractionDataset, collate_extraction

    success_count = 0
    fail_count = 0
    start_time = time.time()

    # Pre-filter already extracted videos
    videos_to_process = []
    for v_path in selected_videos:
        v_name = strip_video_ext(os.path.basename(v_path))
        subfolder = "normal" if ("_label_A" in v_name and args.auto_classify) else ("anomal" if args.auto_classify else "")
        out_file = os.path.join(args.output_dir, subfolder, f"{v_name}.pt")
        if args.overwrite or not os.path.exists(out_file):
            videos_to_process.append(v_path)
        else:
            success_count += 1

    print(f"Already extracted: {success_count} | To extract: {len(videos_to_process)}")

    if not videos_to_process:
        print("All selected videos are already extracted!")
        return

    if args.num_workers > 0:
        # High-performance asynchronous parallel pipeline
        dataset = VideoExtractionDataset(
            video_paths=videos_to_process,
            num_segments=args.num_segments,
            clip_size=args.clip_size,
            fps=args.fps,
            max_clips_per_segment=args.max_clips_per_segment,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            prefetch_factor=2,
            collate_fn=collate_extraction,
        )

        progress = tqdm(loader, desc="Parallel Extraction", position=0, leave=True)

        for item in progress:
            if item["status"] != "ok":
                print(f"\n[ERROR] Failed to decode {item.get('video_name')}: {item.get('error')}")
                fail_count += 1
                continue

            v_name = item["video_name"]
            v_path = item["video_path"]
            progress.set_postfix({"current": v_name[:25]})

            subfolder = "normal" if ("_label_A" in v_name and args.auto_classify) else ("anomal" if args.auto_classify else "")
            out_file = os.path.join(args.output_dir, subfolder, f"{v_name}.pt")

            try:
                segment_tensors = item["segments"]
                segment_features = []

                sub_pbar = tqdm(
                    segment_tensors,
                    desc=f"  ↳ [{v_name[:24]}]",
                    position=1,
                    leave=False,
                    total=len(segment_tensors),
                )

                with torch.no_grad():
                    for seg_tensor in sub_pbar:
                        seg_tensor = seg_tensor.to(args.device)
                        if seg_tensor.dtype == torch.uint8 or seg_tensor.max() > 1.0:
                            seg_tensor = seg_tensor.float() / 255.0

                        if seg_tensor.shape[0] <= args.batch_size:
                            feats = model.video_model(seg_tensor)
                        else:
                            chunks = torch.split(seg_tensor, args.batch_size, dim=0)
                            feats = torch.cat([model.video_model(c) for c in chunks], dim=0)

                        seg_feature = feats.mean(dim=0, keepdim=True)
                        segment_features.append(seg_feature.cpu())

                sub_pbar.close()

                all_features = torch.cat(segment_features, dim=0).float()

                data = {
                    "feats": all_features,
                    "video_name": v_name,
                    "video_path": v_path,
                    "model": args.model_name,
                    "num_segments": args.num_segments,
                    "clip_size": args.clip_size,
                    "fps": args.fps,
                    "feature_dim": all_features.shape[-1],
                }
                torch.save(data, out_file)
                success_count += 1
            except Exception as e:
                print(f"\n[ERROR] GPU inference failed on {v_name}: {e}")
                fail_count += 1
    else:
        # Sequential fallback
        progress = tqdm(videos_to_process, desc="Sequential Extraction", position=0, leave=True)
        for v_path in progress:
            v_name = strip_video_ext(os.path.basename(v_path))
            progress.set_postfix({"current": v_name[:25]})
            subfolder = "normal" if ("_label_A" in v_name and args.auto_classify) else ("anomal" if args.auto_classify else "")
            out_file = os.path.join(args.output_dir, subfolder, f"{v_name}.pt")

            try:
                feats = model.extract_features(
                    video_path=v_path,
                    num_segments=args.num_segments,
                    clip_size=args.clip_size,
                    fps=args.fps,
                    batch_size=args.batch_size,
                    show_progress=True,
                )
                data = {
                    "feats": feats.cpu().float(),
                    "video_name": v_name,
                    "video_path": v_path,
                    "model": args.model_name,
                    "num_segments": args.num_segments,
                    "clip_size": args.clip_size,
                    "fps": args.fps,
                    "feature_dim": feats.shape[-1],
                }
                torch.save(data, out_file)
                success_count += 1
            except Exception as e:
                print(f"\n[ERROR] Failed on {v_name}: {e}")
                fail_count += 1

    total_time = time.time() - start_time
    print(f"\n=== Extraction Summary ===")
    print(f"Total Processed: {len(selected_videos)}")
    print(f"Successfully Extracted: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total Time: {total_time:.1f}s ({total_time / max(1, len(videos_to_process)):.2f}s per pending video)")
    print(f"Saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()

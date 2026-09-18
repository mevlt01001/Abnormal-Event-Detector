from __future__ import annotations

import gc
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu

from utils.video_utils import process_clip_tensor


class VideoExtractionDataset(Dataset):
    """PyTorch Dataset for parallel asynchronous video decoding during feature extraction.

    Used with DataLoader(num_workers > 0, prefetch_factor > 0) to eliminate CPU-GPU
    synchronization bottlenecks. Background worker processes decode multiple videos in parallel,
    feeding ready-to-infer tensors directly to the GPU.
    """

    def __init__(
        self,
        video_paths: List[str],
        num_segments: int = 32,
        clip_size: int = 16,
        fps: int = 30,
        max_clips_per_segment: Optional[int] = 2,
        resize_dim: Tuple[int, int] = (224, 224),
        crop_dim: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.video_paths = list(video_paths)
        self.num_segments = num_segments
        self.clip_size = clip_size
        self.fps = fps
        self.max_clips_per_segment = max_clips_per_segment
        self.resize_dim = resize_dim
        self.crop_dim = crop_dim

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_path = self.video_paths[idx]
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        if not os.path.exists(video_path):
            return {
                "status": "error",
                "error": f"File not found: {video_path}",
                "video_path": video_path,
                "video_name": video_name,
            }

        try:
            vr = VideoReader(
                video_path,
                ctx=cpu(0),
                width=self.resize_dim[1],
                height=self.resize_dim[0],
                num_threads=2,
            )
            total_vr_frames = len(vr)
            if total_vr_frames == 0:
                return {
                    "status": "error",
                    "error": f"Video has 0 frames: {video_path}",
                    "video_path": video_path,
                    "video_name": video_name,
                }

            avg_fps = vr.get_avg_fps()
            step = avg_fps / self.fps if (avg_fps > 0 and self.fps > 0) else 1.0

            frame_indices = np.arange(0, total_vr_frames, step).astype(int)
            total_frames = len(frame_indices)

            total_possible_clips = total_frames // self.clip_size

            if total_possible_clips >= self.num_segments:
                clip_starts = [i * self.clip_size for i in range(total_possible_clips)]
                segment_clip_indices = np.array_split(clip_starts, self.num_segments)
            else:
                starts = np.linspace(
                    0, max(0, total_frames - self.clip_size), self.num_segments, dtype=int
                )
                segment_clip_indices = [[s] for s in starts]

            segment_tensors: List[torch.Tensor] = []

            for clip_start_list in segment_clip_indices:
                # Optionally cap clips per segment for extreme performance
                if (
                    self.max_clips_per_segment is not None
                    and len(clip_start_list) > self.max_clips_per_segment
                ):
                    indices_sub = np.linspace(
                        0, len(clip_start_list) - 1, self.max_clips_per_segment, dtype=int
                    )
                    clip_start_list = [clip_start_list[i] for i in indices_sub]

                seg_clips = []
                for start_f in clip_start_list:
                    sub_indices = frame_indices[start_f : start_f + self.clip_size]

                    # Edge-replication padding for ultra-short clips
                    if len(sub_indices) < self.clip_size:
                        pad_len = self.clip_size - len(sub_indices)
                        sub_indices = np.pad(sub_indices, (0, pad_len), mode="edge")

                    frames = vr.get_batch(sub_indices.tolist()).asnumpy()

                    if self.crop_dim:
                        h, w = frames.shape[1:3]
                        crop_h, crop_w = self.crop_dim
                        sy = max(0, (h - crop_h) // 2)
                        sx = max(0, (w - crop_w) // 2)
                        frames = frames[:, sy : sy + crop_h, sx : sx + crop_w, :]

                    clip_tensor = process_clip_tensor(frames, clip_size=self.clip_size)
                    seg_clips.append(clip_tensor)

                segment_tensor = torch.cat(seg_clips, dim=0)
                segment_tensors.append(segment_tensor)

            del vr
            gc.collect()

            return {
                "status": "ok",
                "video_path": video_path,
                "video_name": video_name,
                "segments": segment_tensors,
                "num_segments": self.num_segments,
                "clip_size": self.clip_size,
                "fps": self.fps,
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "video_path": video_path,
                "video_name": video_name,
            }


def collate_extraction(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for DataLoader batching without concatenating segment lists."""
    return batch[0]

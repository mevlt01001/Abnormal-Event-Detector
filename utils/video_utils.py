from __future__ import annotations

import gc
import os
from typing import Generator, Tuple, Optional
import numpy as np
import torch
from decord import VideoReader, cpu


def get_video_length(video_path: str) -> float:
    """Calculates video duration in seconds using Decord.

    Args:
        video_path: Path to the video file.

    Returns:
        Duration of the video in seconds (float).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    if fps <= 0:
        return 0.0
    return float(total_frames / fps)


def process_clip_tensor(frames_array: np.ndarray, clip_size: int = 16) -> torch.Tensor:
    """Converts numpy array of frames [T, H, W, C] to torch tensor [1, 3, T, H, W].

    Pads temporally if T < clip_size.
    """
    tensor = torch.from_numpy(frames_array)  # [T, H, W, C]
    tensor = tensor.unsqueeze(0)  # [1, T, H, W, C]
    tensor = tensor.permute(0, 4, 1, 2, 3).contiguous()  # [1, C, T, H, W]

    T = tensor.shape[2]
    if T < clip_size:
        pad_size = clip_size - T
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, pad_size))
    elif T > clip_size:
        tensor = tensor[:, :, :clip_size, :, :]

    return tensor


def fetch_video_segments(
    video_path: str,
    num_segments: int = 32,
    fps: int = 30,
    clip_size: int = 16,
    resize_dim: Tuple[int, int] = (224, 224),
    crop_dim: Optional[Tuple[int, int]] = None,
) -> Generator[torch.Tensor, None, None]:
    """Extracts temporal segments from a video as clip tensors for Sultani MIL feature extraction.

    Implements a clip-first partitioning strategy to eliminate black-frame zero padding:
    - Resamples the video to the target frame rate (default: 30 FPS).
    - If the video contains at least `num_segments` non-overlapping full clips (e.g. 900 frames -> 56 clips),
      it partitions all consecutive clips across `num_segments` segments.
    - If the video is shorter than `num_segments * clip_size` (e.g. < 512 frames), it computes dynamic
      overlapping clip offsets (sliding window) so that every segment receives a full, real 16-frame clip
      without any black zero-padding.
    - For edge cases where a video has fewer than `clip_size` total frames, it uses edge-replication
      padding (repeating the final frame) instead of black frames to preserve gradient stability.

    Args:
        video_path: Path to the input video file.
        num_segments: Number of temporal segments to divide the video into (default: 32).
        fps: Target frame rate for temporal resampling (default: 30).
        clip_size: Number of frames per clip expected by the 3D backbone (default: 16).
        resize_dim: (height, width) resolution to decode frames with Decord (default: (224, 224)).
        crop_dim: Optional (height, width) spatial center-crop dimensions.

    Yields:
        torch.Tensor of shape (K, 3, clip_size, H, W) for each segment, where K is the number
        of 16-frame clips assigned to that segment (K >= 1).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    vr = VideoReader(video_path, ctx=cpu(0), width=resize_dim[1], height=resize_dim[0], num_threads=4)
    total_vr_frames = len(vr)
    if total_vr_frames == 0:
        raise ValueError(f"Video {video_path} contains 0 frames.")

    avg_fps = vr.get_avg_fps()
    step = avg_fps / fps if (avg_fps > 0 and fps > 0) else 1.0

    try:
        # 1. Resample original frame indices to match target FPS
        frame_indices = np.arange(0, total_vr_frames, step).astype(int)
        total_frames = len(frame_indices)

        # 2. Determine clip start indices and distribution across segments
        total_possible_clips = total_frames // clip_size

        if total_possible_clips >= num_segments:
            # Long video: Extract consecutive full clips without padding
            # e.g., 900 frames -> 56 clips; distribute 56 clips across 32 segments
            clip_starts = [i * clip_size for i in range(total_possible_clips)]
            # Partition clip indices into num_segments lists (e.g. [[0, 1], [2, 3], ...])
            segment_clip_indices = np.array_split(clip_starts, num_segments)
        else:
            # Short video (< num_segments * clip_size): Sample overlapping clips via dynamic stride
            # Ensures every segment receives exactly 1 complete 16-frame real clip with zero black-padding
            starts = np.linspace(0, max(0, total_frames - clip_size), num_segments, dtype=int)
            segment_clip_indices = [[s] for s in starts]

        # 3. Read clips via Decord batch decoding and yield per segment
        for clip_start_list in segment_clip_indices:
            seg_clips = []
            for start_f in clip_start_list:
                sub_indices = frame_indices[start_f : start_f + clip_size]

                # For extremely short videos (< clip_size frames), apply edge replication instead of black padding
                if len(sub_indices) < clip_size:
                    pad_len = clip_size - len(sub_indices)
                    sub_indices = np.pad(sub_indices, (0, pad_len), mode="edge")

                frames = vr.get_batch(sub_indices.tolist()).asnumpy()

                if crop_dim:
                    h, w = frames.shape[1:3]
                    crop_h, crop_w = crop_dim
                    sy = max(0, (h - crop_h) // 2)
                    sx = max(0, (w - crop_w) // 2)
                    frames = frames[:, sy : sy + crop_h, sx : sx + crop_w, :]

                clip_tensor = process_clip_tensor(frames, clip_size=clip_size)
                seg_clips.append(clip_tensor)

            # Yield all clips belonging to this segment: [K, 3, clip_size, H, W]
            yield torch.cat(seg_clips, dim=0)
    finally:
        del vr
        gc.collect()



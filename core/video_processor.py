"""Unified video processing engine powered by Decord.

Single source of truth for reading, temporal segmenting, sliding clip generation,
and clipping of video files without code duplication.
"""

from __future__ import annotations

import gc
import os
from typing import Any, Dict, Generator, List, Optional, Tuple
import numpy as np
import torch
from decord import VideoReader, cpu


class VideoProcessor:
    """Decord-based video frame processor and temporal clip sampler.

    Args:
        target_fps: Target frame rate for temporal resampling (default: 20.0).
            Avoids hardcoded sampling and allows user-defined temporal resolution.
        clip_size: Number of consecutive frames per spatio-temporal clip (default: 16).
        stride: Temporal stride (in sampled frames) between sliding clips (default: 16).
        width: Output frame width (default: 224).
        height: Output frame height (default: 224).
    """

    def __init__(
        self,
        target_fps: float = 20.0,
        clip_size: int = 16,
        stride: int = 16,
        width: int = 224,
        height: int = 224,
    ) -> None:
        self.target_fps = float(target_fps)
        self.clip_size = int(clip_size)
        self.stride = int(stride)
        self.width = int(width)
        self.height = int(height)

    def get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Reads basic video metadata without loading full frame tensors.

        Args:
            video_path: Path to video file (.mp4, .avi, etc.).

        Returns:
            Dictionary containing duration_sec, original_fps, total_frames, width, height.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file does not exist: {video_path}")

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        orig_fps = vr.get_avg_fps()
        duration_sec = total_frames / max(1.0, orig_fps)
        h, w, _ = vr[0].shape

        del vr
        gc.collect()

        return {
            "video_path": video_path,
            "duration_sec": float(duration_sec),
            "original_fps": float(orig_fps),
            "total_frames": int(total_frames),
            "height": int(h),
            "width": int(w),
        }

    def extract_uniform_segments(
        self,
        video_path: str,
        num_segments: int = 32,
    ) -> Generator[torch.Tensor, None, None]:
        """Uniformly divides video into `num_segments` temporal segments (Sultani CVPR 2018 MIL).

        For each segment, extracts 16-frame clips, converts uint8 -> float32 [0, 1],
        and yields segment tensors formatted for 3D CNN / Transformer backbones.

        Args:
            video_path: Absolute or relative path to the video file.
            num_segments: Number of temporal bags/segments (default: 32).

        Yields:
            seg_tensor: Spatio-temporal clip tensor of shape [K, C, clip_size, H, W]
                where K is the number of 16-frame clips sampled within this segment,
                C=3 (RGB), clip_size=16 frames, H=224, W=224.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        vr = VideoReader(video_path, ctx=cpu(0), width=self.width, height=self.height)
        total_vr_frames = len(vr)
        orig_fps = vr.get_avg_fps()

        # Dynamic step based on target_fps (not hardcoded)
        step = (orig_fps / self.target_fps) if (orig_fps > 0 and self.target_fps > 0) else 1.0
        sampled_indices = torch.arange(0, total_vr_frames, step=step).long()
        total_sampled = len(sampled_indices)

        if total_sampled < self.clip_size:
            # Replicate indices if video is shorter than clip_size
            sampled_indices = sampled_indices.repeat(int(np.ceil(self.clip_size / max(1, total_sampled))))[:self.clip_size]
            total_sampled = self.clip_size

        segment_boundaries = np.linspace(0, total_sampled, num_segments + 1, dtype=int)

        try:
            for s_idx in range(num_segments):
                seg_start = segment_boundaries[s_idx]
                seg_end = segment_boundaries[s_idx + 1]
                seg_indices = sampled_indices[seg_start:seg_end]

                if len(seg_indices) < self.clip_size:
                    if len(seg_indices) == 0:
                        seg_indices = sampled_indices[-self.clip_size:]
                    else:
                        pad = sampled_indices[max(0, seg_end - self.clip_size):seg_end]
                        seg_indices = pad if len(pad) == self.clip_size else sampled_indices[:self.clip_size]

                # Sample 16-frame clips inside this segment
                clips_in_seg = []
                for c_start in range(0, max(1, len(seg_indices) - self.clip_size + 1), self.stride):
                    c_end = c_start + self.clip_size
                    c_frame_ids = seg_indices[c_start:c_end]
                    if len(c_frame_ids) == self.clip_size:
                        # [clip_size, H, W, C] numpy array from Decord
                        frames = vr.get_batch(c_frame_ids.tolist()).asnumpy()
                        # [clip_size, H, W, C] -> [clip_size, H, W, C] torch uint8
                        clip_t = torch.from_numpy(frames)
                        # [clip_size, H, W, C] -> [C, clip_size, H, W] float32 in [0, 1]
                        clip_t = clip_t.permute(3, 0, 1, 2).float() / 255.0
                        clips_in_seg.append(clip_t)

                if not clips_in_seg:
                    c_frame_ids = seg_indices[:self.clip_size]
                    frames = vr.get_batch(c_frame_ids.tolist()).asnumpy()
                    clip_t = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
                    clips_in_seg.append(clip_t)

                # [K, C, clip_size, H, W]
                segment_tensor = torch.stack(clips_in_seg, dim=0)
                yield segment_tensor

        finally:
            del vr
            gc.collect()

    def extract_sliding_clips(
        self,
        video_path: str,
        batch_size: int = 8,
    ) -> Tuple[Generator[torch.Tensor, None, None], int, float]:
        """Generates continuous sliding clips for fine-grained second-by-second inference.

        Args:
            video_path: Path to video file.
            batch_size: Batch size of sliding clips yielded per step.

        Returns:
            Tuple containing:
                clip_generator: Yields tensors of shape [B, C, clip_size, H, W]
                num_clips: Total number of temporal clips
                duration_sec: Video length in seconds
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        vr = VideoReader(video_path, ctx=cpu(0), width=self.width, height=self.height)
        total_vr_frames = len(vr)
        orig_fps = vr.get_avg_fps()
        duration_sec = total_vr_frames / max(1.0, orig_fps)

        step = (orig_fps / self.target_fps) if (orig_fps > 0 and self.target_fps > 0) else 1.0
        frame_indices = torch.arange(0, total_vr_frames, step=step).long()
        total_sampled = len(frame_indices)

        if total_sampled < self.clip_size:
            frame_indices = frame_indices.repeat(int(np.ceil(self.clip_size / max(1, total_sampled))))[:self.clip_size]
            total_sampled = self.clip_size

        clip_starts = list(range(0, total_sampled - self.clip_size + 1, self.stride))
        if not clip_starts:
            clip_starts = [0]
        total_clips = len(clip_starts)

        def _generator(reader: VideoReader):
            try:
                batch_clips = []
                for start_idx in clip_starts:
                    end_idx = start_idx + self.clip_size
                    ids = frame_indices[start_idx:end_idx]
                    # [clip_size, H, W, C]
                    raw_frames = reader.get_batch(ids.tolist()).asnumpy()
                    # [clip_size, H, W, C] -> [C, clip_size, H, W] in [0, 1]
                    clip_tensor = torch.from_numpy(raw_frames).permute(3, 0, 1, 2).float() / 255.0
                    batch_clips.append(clip_tensor)

                    if len(batch_clips) == batch_size:
                        # [B, C, clip_size, H, W]
                        yield torch.stack(batch_clips, dim=0)
                        batch_clips = []

                if batch_clips:
                    # [Remainder, C, clip_size, H, W]
                    yield torch.stack(batch_clips, dim=0)
            finally:
                del reader
                gc.collect()

        return _generator(vr), total_clips, duration_sec

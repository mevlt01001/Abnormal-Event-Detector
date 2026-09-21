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
        stride: Optional[int] = None,
        overlap_ratio: float = 0.0,
        width: int = 224,
        height: int = 224,
    ) -> None:
        self.target_fps = float(target_fps)
        self.clip_size = int(clip_size)
        self.overlap_ratio = max(0.0, min(0.95, float(overlap_ratio)))
        self.width = int(width)
        self.height = int(height)

        if stride is not None:
            self.stride = max(1, int(stride))
        elif self.overlap_ratio > 0.0:
            self.stride = max(1, int(round(self.clip_size * (1.0 - self.overlap_ratio))))
        else:
            self.stride = int(self.clip_size)

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
        overlap_ratio: Optional[float] = None,
        stride: Optional[int] = None,
    ) -> Generator[torch.Tensor, None, None]:
        """Uniformly divides video into temporal segments and samples spatiotemporal clips.

        Zero Black-Frame Guarantee:
        - Clips always contain authentic video frames. No black (zero) padding is used.
        - For short segments, real adjacent frames are sampled around the temporal center.
        - When video length < clip_size, frames are temporally interpolated from available real frames.

        Args:
            video_path: Path to the video file.
            num_segments: Number of temporal segments (default: 32).
            overlap_ratio: Optional overlap ratio between adjacent clips (0.0 to 0.95).
            stride: Optional explicit step between consecutive clips.

        Yields:
            segment_tensor: Tensor of shape [K, C, clip_size, H, W] in range [0.0, 1.0].
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Determine effective stride based on overlap
        if stride is not None:
            eff_stride = max(1, int(stride))
        elif overlap_ratio is not None:
            ov = max(0.0, min(0.95, float(overlap_ratio)))
            eff_stride = max(1, int(round(self.clip_size * (1.0 - ov))))
        else:
            eff_stride = self.stride

        vr = VideoReader(video_path, ctx=cpu(0), width=self.width, height=self.height)
        total_vr_frames = len(vr)
        orig_fps = vr.get_avg_fps()

        # Resample frame indices based on target_fps
        step = (orig_fps / self.target_fps) if (orig_fps > 0 and self.target_fps > 0) else 1.0
        sampled_indices = torch.arange(0, total_vr_frames, step=step).long()
        total_sampled = len(sampled_indices)

        # Handle extremely short videos by interpolating authentic video frames
        if total_sampled < self.clip_size:
            idx = torch.linspace(0, max(0, total_vr_frames - 1), self.clip_size).round().long()
            sampled_indices = idx
            total_sampled = len(sampled_indices)

        segment_boundaries = np.linspace(0, total_sampled, num_segments + 1, dtype=int)

        try:
            for s_idx in range(num_segments):
                seg_start = int(segment_boundaries[s_idx])
                seg_end = int(segment_boundaries[s_idx + 1])
                seg_indices = sampled_indices[seg_start:seg_end]
                seg_len = len(seg_indices)

                clips_in_seg: List[torch.Tensor] = []

                if seg_len >= self.clip_size:
                    # Slide clips with effective stride
                    for c_start in range(0, seg_len - self.clip_size + 1, eff_stride):
                        c_frame_ids = seg_indices[c_start : c_start + self.clip_size]
                        frames = vr.get_batch(c_frame_ids.tolist()).asnumpy()
                        clip_t = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
                        clips_in_seg.append(clip_t)

                    # If the tail was missed due to stride, anchor a final clip at the end of the segment
                    tail_start = seg_len - self.clip_size
                    if (
                        tail_start > 0
                        and (len(clips_in_seg) == 0 or (seg_len - self.clip_size) % eff_stride != 0)
                    ):
                        c_frame_ids = seg_indices[tail_start:]
                        frames = vr.get_batch(c_frame_ids.tolist()).asnumpy()
                        clip_t = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
                        clips_in_seg.append(clip_t)

                else:
                    # Segment is shorter than clip_size: expand window around center from sampled_indices
                    center = (seg_start + seg_end) // 2
                    win_start = max(0, center - self.clip_size // 2)
                    win_end = win_start + self.clip_size
                    if win_end > total_sampled:
                        win_end = total_sampled
                        win_start = max(0, win_end - self.clip_size)

                    c_frame_ids = sampled_indices[win_start:win_end]

                    # Fallback for video shorter than clip_size: authentic linear frame interpolation
                    if len(c_frame_ids) < self.clip_size:
                        rep_idx = torch.linspace(0, max(0, total_vr_frames - 1), self.clip_size).round().long()
                        c_frame_ids = rep_idx

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

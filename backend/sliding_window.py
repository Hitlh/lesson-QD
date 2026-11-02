"""Temporal bookkeeping for overlapping inference on a shared two-second grid."""
from __future__ import annotations

import math
from collections.abc import Sequence


def clip_count(duration: float, clip_duration: float = 2.0) -> int:
    if not math.isfinite(duration) or duration <= 0 or clip_duration <= 0:
        raise ValueError("Video and clip durations must be finite and positive")
    # Containers can outlast their final video frame by a few milliseconds.
    nearest = round(duration / clip_duration)
    if nearest > 0 and abs(duration - nearest * clip_duration) <= 0.05:
        return nearest
    return max(1, math.ceil(duration / clip_duration))


def build_windows(num_clips: int, window_clips: int = 75, overlap_clips: int = 15) -> list[tuple[int, int]]:
    if num_clips <= 0 or not 1 <= window_clips <= 75 or not 0 <= overlap_clips < window_clips:
        raise ValueError("Require positive clip count, window <= 75 and 0 <= overlap < window")
    windows = []
    start = 0
    while True:
        end = min(start + window_clips, num_clips)
        windows.append((start, end))
        if end == num_clips:
            return windows
        start += window_clips - overlap_clips


def globalize_candidates(
    candidates: Sequence[Sequence[float]], start_clip: int, end_clip: int,
    duration: float, clip_duration: float = 2.0,
) -> list[list[float]]:
    offset = start_clip * clip_duration
    valid_duration = min((end_clip - start_clip) * clip_duration, duration - offset)
    # Clip BEFORE adding the offset: out-of-window predictions must not spill
    # into a neighboring window's time range.
    return [[
        offset + max(0.0, min(float(row[0]), valid_duration)),
        offset + max(0.0, min(float(row[1]), valid_duration)),
        float(row[2]),
    ] for row in candidates]


def merge_saliency(num_clips: int, windows: Sequence[tuple[int, Sequence[float]]]) -> list[float]:
    totals = [0.0] * num_clips
    counts = [0] * num_clips
    for start, scores in windows:
        if start < 0 or start + len(scores) > num_clips:
            raise ValueError("Saliency window outside video")
        for index, score in enumerate(scores, start):
            if not math.isfinite(score):
                raise ValueError("Non-finite saliency score")
            totals[index] += score
            counts[index] += 1
    if any(count == 0 for count in counts):
        raise ValueError("Some video clips were not covered by inference")
    return [round(total / count, 4) for total, count in zip(totals, counts)]

"""Query-local temporal suppression, independent of model/GPU dependencies.

Inputs must use seconds on the same video timeline. For sliding windows, first
convert every prediction to global coordinates, then pass all candidates here.
"""
from __future__ import annotations

import math
from typing import Literal, Sequence

NMSMethod = Literal["none", "hard", "soft_linear"]
NMS_METHODS = ("none", "hard", "soft_linear")


def temporal_iou(a: Sequence[float], b: Sequence[float]) -> float:
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - intersection
    return intersection / union if union > 0 else 0.0


def postprocess_moments(
    candidates: Sequence[Sequence[float]],
    duration: float,
    *,
    method: NMSMethod = "hard",
    iou_threshold: float = 0.5,
    min_score: float = 0.0,
    top_k: int = 3,
) -> tuple[list[dict[str, float]], dict[str, str | float | int]]:
    """Clean all candidates, suppress overlaps, then select Top-K.

    Suppression uses strict IoU > threshold. Ties retain input order. Linear
    Soft-NMS reselects the highest score after each decay and removes scores
    <= min_score. Other modes preserve original scores. Inputs are not mutated.
    """
    if method not in NMS_METHODS:
        raise ValueError(f"Unsupported NMS method: {method}")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    if not math.isfinite(iou_threshold) or not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")
    if not math.isfinite(min_score) or min_score < 0:
        raise ValueError("min_score must be finite and nonnegative")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    pool = []
    for row in candidates:
        if len(row) != 3:
            continue
        try:
            start, end, score = map(float, row)
        except (TypeError, ValueError, OverflowError):
            continue
        if not all(math.isfinite(value) for value in (start, end, score)):
            continue
        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))
        # Never return an empty interval, including after API rounding.
        if round(end, 4) <= round(start, 4):
            continue
        pool.append({"start": start, "end": end, "score": score, "raw_score": score})

    valid_count = len(pool)
    pool.sort(key=lambda moment: moment["score"], reverse=True)
    kept = []
    if method == "none":
        kept = pool
    else:
        while pool:
            best = pool.pop(0)
            if method == "soft_linear" and best["score"] <= min_score:
                break
            kept.append(best)
            remaining = []
            for moment in pool:
                overlap = temporal_iou(
                    (best["start"], best["end"]),
                    (moment["start"], moment["end"]),
                )
                if overlap > iou_threshold:
                    if method == "hard":
                        continue
                    moment["score"] *= 1.0 - overlap
                remaining.append(moment)
            pool = sorted(remaining, key=lambda moment: moment["score"], reverse=True)

    moments = [
        {key: round(value, 4) for key, value in moment.items()}
        for moment in kept[:top_k]
    ]
    info = {
        "method": method,
        "iou_threshold": iou_threshold,
        "min_score": min_score,
        "candidates_before": len(candidates),
        "candidates_valid": valid_count,
        "candidates_after": len(kept),
        "returned_count": len(moments),
    }
    return moments, info

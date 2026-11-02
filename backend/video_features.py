"""Bounded video decoding and query-independent visual features.

Short videos retain the original extraction path. Long videos decode aligned,
non-overlapping blocks; only the much smaller features survive between blocks.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from .sliding_window import clip_count

logger = logging.getLogger("uvicorn.error")


def _decode_block(path: Path, start: float, duration: float, info: dict, *, fps: float, frames: int, size: int = 224):
    import ffmpeg
    import numpy as np
    import torch

    height, width = info["height"], info["width"]
    if height >= width:
        scaled_height, scaled_width = int(height * size / width), size
    else:
        scaled_height, scaled_width = size, int(width * size / height)
    stream = (
        ffmpeg.input(str(path), ss=start, t=duration)
        # At EOF, clone the final frame so a partial two-second clip is kept.
        # vframes bounds the output to the exact global clip grid.
        .filter("tpad", stop_mode="clone", stop_duration=2)
        .filter("fps", fps=fps)
        .filter("scale", scaled_width, scaled_height)
        .crop((scaled_width - size) // 2, (scaled_height - size) // 2, size, size)
    )
    try:
        raw, _ = stream.output("pipe:", format="rawvideo", pix_fmt="rgb24", vframes=frames).run(
            capture_stdout=True, quiet=True,
        )
    except ffmpeg.Error as exc:
        raise RuntimeError(f"无法解码视频 {start:g} 秒处的特征块") from exc
    array = np.frombuffer(raw, dtype=np.uint8)
    expected = frames * size * size * 3
    if array.size != expected:
        raise RuntimeError(f"视频 {start:g} 秒处解码不完整，无法对齐两秒特征")
    return torch.from_numpy(array.reshape(frames, size, size, 3).copy())


def _normalize_and_join(clip_features, slowfast_features=None):
    import torch
    import torch.nn.functional as functional

    clip_features = functional.normalize(clip_features, dim=-1, eps=1e-5)
    if slowfast_features is not None:
        slowfast_features = functional.normalize(slowfast_features, dim=-1, eps=1e-5)
        common = min(len(clip_features), len(slowfast_features))
        if common == 0:
            raise RuntimeError("SlowFast 或 CLIP 未能生成视频特征")
        features = torch.cat([slowfast_features[:common], clip_features[:common]], dim=1)
    else:
        features = clip_features
    if len(features) == 0:
        raise RuntimeError("无法生成视频特征")
    # Match the FP32 promotion previously performed when appending TEF.
    return features.detach().float().cpu()


def extract_video_features(
    predictor: Any, path: Path, duration: float, *,
    slowfast_components: tuple[Any, Any] | None = None,
    slowfast_batch_size: int = 16, chunk_clips: int = 30,
):
    import torch

    extractor = predictor.feature_extractor
    if duration <= 150.05:
        # Preserve the existing preprocessing, sampling and modality trim so
        # switching on long-video support does not alter short-video results.
        clip_features = extractor.encode_video(str(path))
        slowfast_features = None
        if slowfast_components is not None:
            loader, encoder = slowfast_components
            frames = loader(str(path))
            if frames is None:
                raise RuntimeError("SlowFast 无法读取该视频")
            slowfast_features = encoder(frames, bsz=slowfast_batch_size)
        features = _normalize_and_join(clip_features, slowfast_features)
        if len(features) > 75:
            # Unusual container/sampling lengths use the aligned block path.
            del features, clip_features, slowfast_features
        else:
            return features

    info = extractor.video_loader._get_video_info(str(path))
    info["fps"] = max(1, info["fps"])
    total_clips = clip_count(duration)
    blocks = []
    with torch.no_grad():
        for start in range(0, total_clips, chunk_clips):
            count = min(chunk_clips, total_clips - start)
            seconds = min(count * 2.0, duration - start * 2.0)
            logger.info("Extracting %s feature block %d/%d (%g-%gs)", path.name,
                        start // chunk_clips + 1, math.ceil(total_clips / chunk_clips), start * 2, start * 2 + seconds)
            clip_frames = _decode_block(path, start * 2.0, seconds, info, fps=0.5, frames=count)
            clip_frames = extractor.video_preprocessor(clip_frames.permute(0, 3, 1, 2).float())
            clip_features = []
            for offset in range(0, count, 60):
                clip_features.append(extractor.clip_extractor.encode_image(clip_frames[offset:offset + 60].to(predictor.device)))
            clip_features = torch.cat(clip_features)
            del clip_frames
            slowfast_features = None
            if slowfast_components is not None:
                loader, encoder = slowfast_components
                frames = _decode_block(path, start * 2.0, seconds, info, fps=30, frames=count * 60)
                frames = loader._preprocess(frames, dict(info, duration=count * 2.0))
                slowfast_features = encoder(frames, bsz=slowfast_batch_size)
                del frames
                if len(slowfast_features) != count:
                    raise RuntimeError("SlowFast 特征未对齐全局两秒网格")
            if len(clip_features) != count:
                raise RuntimeError("CLIP 特征未对齐全局两秒网格")
            blocks.append(_normalize_and_join(clip_features, slowfast_features))
            del clip_features, slowfast_features
    features = torch.cat(blocks)
    if len(features) != total_clips:
        raise RuntimeError("视频尾部特征缺失")
    return features

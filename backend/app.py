from __future__ import annotations

import os
import logging
import math
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .config import PROJECT_ROOT, configured_path
from .temporal_postprocessing import NMSMethod, postprocess_moments
from .sliding_window import build_windows, globalize_candidates, merge_saliency
from .video_features import extract_video_features


class PostprocessingOptions(BaseModel):
    method: NMSMethod = "hard"
    iou_threshold: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    min_score: float = Field(default=0.0, ge=0, allow_inf_nan=False)


POSTPROCESSING = PostprocessingOptions(
    method=os.getenv("QD_DETR_NMS_METHOD", "hard"),
    iou_threshold=float(os.getenv("QD_DETR_NMS_THRESHOLD", "0.5")),
    min_score=float(os.getenv("QD_DETR_NMS_MIN_SCORE", "0")),
)


DEFAULT_QD_DETR_ROOT = PROJECT_ROOT / "QD-DETR"
QD_DETR_ROOT = configured_path("QD_DETR_ROOT", "QD-DETR")
CHECKPOINT_PATH = configured_path("QD_DETR_CHECKPOINT", "models/qv_model_best.ckpt")
STORAGE_DIR = configured_path("VIDEO_STORAGE_DIR", "storage")
FRONTEND_DIST = configured_path("FRONTEND_DIST", "frontend/dist")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "256")) * 1024 * 1024
MAX_VIDEO_DURATION = float(os.getenv("MAX_VIDEO_DURATION", "600"))
WINDOW_CLIPS = int(os.getenv("QD_DETR_WINDOW_CLIPS", "75"))
WINDOW_OVERLAP_CLIPS = int(os.getenv("QD_DETR_WINDOW_OVERLAP_CLIPS", "15"))
FEATURE_CHUNK_CLIPS = int(os.getenv("FEATURE_CHUNK_CLIPS", "30"))
CLIP_DURATION = 2.0
logger = logging.getLogger("uvicorn.error")
# Fail early instead of silently producing gaps or exceeding model capacity.
build_windows(1, WINDOW_CLIPS, WINDOW_OVERLAP_CLIPS)
if not 1 <= FEATURE_CHUNK_CLIPS <= 75:
    raise RuntimeError("FEATURE_CHUNK_CLIPS must be between 1 and 75")
if not math.isfinite(MAX_VIDEO_DURATION) or MAX_VIDEO_DURATION <= 0:
    raise RuntimeError("MAX_VIDEO_DURATION must be finite and positive")
MODEL_DEVICE = os.getenv("QD_DETR_DEVICE", "cuda")
FEATURE_MODE = os.getenv("QD_DETR_FEATURE_MODE", "slowfast_clip").strip().lower()
CLIP_MODEL_PATH = str(configured_path("QD_DETR_CLIP_MODEL", "models/ViT-B-32.pt"))
LIGHTHOUSE_ROOT = configured_path("LIGHTHOUSE_ROOT", "lighthouse")
SLOWFAST_CHECKPOINT = configured_path("SLOWFAST_CHECKPOINT", "models/SLOWFAST_8x8_R50.pkl")
SLOWFAST_BATCH_SIZE = max(1, int(os.getenv("SLOWFAST_BATCH_SIZE", "16")))
FEATURE_CACHE_SIZE = max(1, int(os.getenv("FEATURE_CACHE_SIZE", "32")))
_ranker_path = os.getenv("QD_DETR_WINDOW_RANKER", "").strip()
WINDOW_RANKER_PATH = configured_path("QD_DETR_WINDOW_RANKER", _ranker_path) if _ranker_path else None
WINDOW_RANKER_STRENGTH = (
    float(os.environ["QD_DETR_WINDOW_RANKER_STRENGTH"])
    if os.getenv("QD_DETR_WINDOW_RANKER_STRENGTH", "").strip() else None
)
if WINDOW_RANKER_STRENGTH is not None and (
    not math.isfinite(WINDOW_RANKER_STRENGTH) or not 0 <= WINDOW_RANKER_STRENGTH <= 8
):
    raise RuntimeError("Window ranker strength must be between 0 and 8")
ALLOWED_SUFFIXES = {".mp4", ".webm", ".mov"}
FEATURE_MODES = {"clip", "slowfast_clip"}

if FEATURE_MODE not in FEATURE_MODES:
    raise RuntimeError(
        f"Unsupported QD_DETR_FEATURE_MODE={FEATURE_MODE!r}; "
        f"choose one of {sorted(FEATURE_MODES)}"
    )

# When the service is launched with an environment's absolute Python path
# instead of activating that environment, its ffmpeg binaries may not be on
# PATH. Add the interpreter's bin directory so ffmpeg-python and ffprobe agree.
PYTHON_BIN_DIR = str(Path(sys.executable).resolve().parent)
if PYTHON_BIN_DIR not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = PYTHON_BIN_DIR + os.pathsep + os.environ.get("PATH", "")

# The official checkpoint is trusted and contains argparse.Namespace metadata.
# PyTorch 2.6+ otherwise changes torch.load to weights-only mode.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

STORAGE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="FrameSeek QD-DETR API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    video_id: str = Field(min_length=1, max_length=64)
    query: str = Field(min_length=1, max_length=160)
    top_k: int = Field(default=3, ge=1, le=5)
    nms_method: NMSMethod = POSTPROCESSING.method
    nms_threshold: float = Field(default=POSTPROCESSING.iou_threshold, ge=0, le=1, allow_inf_nan=False)
    nms_min_score: float = Field(default=POSTPROCESSING.min_score, ge=0, allow_inf_nan=False)


class VideoRecord(BaseModel):
    video_id: str
    filename: str
    path: Path
    duration: float


_videos: dict[str, VideoRecord] = {}
_predictor: Any | None = None
_window_ranker: Any | None = None
_slowfast_loader: Any | None = None
_slowfast_extractor: Any | None = None
_video_feature_cache: OrderedDict[str, Any] = OrderedDict()
_load_lock = threading.Lock()
_ranker_lock = threading.Lock()
_inference_lock = threading.Lock()


def _probe_duration(path: Path) -> float:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        duration = float(completed.stdout.strip())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="无法读取视频，请确认 FFmpeg 已安装且文件有效") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise HTTPException(status_code=400, detail="视频时长无效")
    return duration


def _get_predictor() -> Any:
    global _predictor
    if _predictor is not None:
        return _predictor

    with _load_lock:
        if _predictor is not None:
            return _predictor
        if not QD_DETR_ROOT.is_dir():
            raise RuntimeError(f"QD-DETR repository not found: {QD_DETR_ROOT}")
        if not CHECKPOINT_PATH.is_file():
            raise RuntimeError(f"QD-DETR checkpoint not found: {CHECKPOINT_PATH}")
        if str(QD_DETR_ROOT) not in sys.path:
            sys.path.insert(0, str(QD_DETR_ROOT))

        from run_on_video.run import QDDETRPredictor

        predictor = QDDETRPredictor(
            ckpt_path=str(CHECKPOINT_PATH),
            clip_model_name_or_path=CLIP_MODEL_PATH,
            device=MODEL_DEVICE,
        )
        # The upstream demo omits eval(), so enforce deterministic inference
        # even when this backend points at an unmodified official checkout.
        predictor.model.eval()
        _predictor = predictor
        return _predictor


def _get_window_ranker(predictor: Any) -> Any | None:
    global _window_ranker
    if WINDOW_RANKER_PATH is None:
        return None
    with _ranker_lock:
        if _window_ranker is None:
            from .window_reranker import WindowRanker
            _window_ranker = WindowRanker(
                WINDOW_RANKER_PATH, CHECKPOINT_PATH, predictor.device,
                strength=WINDOW_RANKER_STRENGTH,
            )
        return _window_ranker


def _get_slowfast_components() -> tuple[Any, Any]:
    global _slowfast_loader, _slowfast_extractor
    if _slowfast_loader is not None and _slowfast_extractor is not None:
        return _slowfast_loader, _slowfast_extractor

    if not LIGHTHOUSE_ROOT.is_dir():
        raise RuntimeError(f"Lighthouse repository not found: {LIGHTHOUSE_ROOT}")
    if not SLOWFAST_CHECKPOINT.is_file():
        raise RuntimeError(f"SlowFast checkpoint not found: {SLOWFAST_CHECKPOINT}")
    if str(LIGHTHOUSE_ROOT) not in sys.path:
        sys.path.insert(0, str(LIGHTHOUSE_ROOT))

    try:
        from lighthouse.feature_extractor.vision_encoders.slowfast import SlowFast
        from lighthouse.frame_loaders.slowfast_loader import SlowFastLoader
    except ImportError as exc:
        raise RuntimeError(
            "SlowFast dependencies are incomplete; install requirements.txt "
            "inside the QD-DETR environment"
        ) from exc

    _slowfast_loader = SlowFastLoader(
        clip_len=2.0,
        framerate=30,
        size=224,
        device=MODEL_DEVICE,
        centercrop=True,
    )
    _slowfast_extractor = SlowFast(
        device=MODEL_DEVICE,
        model_path=str(SLOWFAST_CHECKPOINT),
    )
    return _slowfast_loader, _slowfast_extractor


def _get_video_features(predictor: Any, record: VideoRecord) -> tuple[Any, bool]:
    cached = _video_feature_cache.get(record.video_id)
    if cached is not None:
        _video_feature_cache.move_to_end(record.video_id)
        return cached, True

    # Cache only query-independent appearance, never whole-video TEF. Each
    # inference window constructs its own local time encoding below.
    cached_features = extract_video_features(
        predictor, record.path, record.duration,
        slowfast_components=_get_slowfast_components() if FEATURE_MODE == "slowfast_clip" else None,
        slowfast_batch_size=SLOWFAST_BATCH_SIZE,
        chunk_clips=FEATURE_CHUNK_CLIPS,
    )
    expected_dimension = predictor.model.input_vid_proj[0].net[1].in_features
    if cached_features.shape[-1] + 2 != expected_dimension:
        raise RuntimeError(
            "checkpoint 需要 "
            f"{expected_dimension} 维视频输入，但 {FEATURE_MODE} 特征链路加上时间编码后生成 "
            f"{cached_features.shape[-1] + 2} 维；请检查 feature mode 与 checkpoint 是否匹配"
        )

    _video_feature_cache[record.video_id] = cached_features
    _video_feature_cache.move_to_end(record.video_id)
    while len(_video_feature_cache) > FEATURE_CACHE_SIZE:
        _video_feature_cache.popitem(last=False)
    return cached_features, False


def _predict_with_cached_features(
    predictor: Any,
    record: VideoRecord,
    query: str,
) -> tuple[dict[str, Any], bool]:
    import torch
    import torch.nn.functional as functional

    from qd_detr.span_utils import span_cxw_to_xx
    from utils.tensor_utils import pad_sequences_1d

    cached_video_features, cache_hit = _get_video_features(predictor, record)
    frame_count = len(cached_video_features)
    windows = build_windows(frame_count, WINDOW_CLIPS, WINDOW_OVERLAP_CLIPS)
    ranked_windows = []
    saliency_windows = []
    window_details = []
    ranker = _get_window_ranker(predictor) if len(windows) > 1 else None
    if ranker is not None and ranker.strength == 0:
        ranker = None
    if ranker is not None:
        from .window_reranker import CaptureWindow, combine_scores

    with torch.no_grad():
        query_features = predictor.feature_extractor.encode_text([query])
        query_features, query_mask = pad_sequences_1d(
            query_features,
            dtype=torch.float32,
            device=predictor.device,
            fixed_length=None,
        )
        query_features = functional.normalize(query_features, dim=-1, eps=1e-5)

        for window_id, (start, end) in enumerate(windows):
            count = end - start
            visual = cached_video_features[start:end].to(predictor.device)
            temporal_start = torch.arange(count, dtype=torch.float32, device=predictor.device) / count
            temporal = torch.stack([temporal_start, temporal_start + 1.0 / count], dim=1)
            video_features = torch.cat([visual, temporal], dim=1).unsqueeze(0)
            video_mask = torch.ones(1, count, device=predictor.device)
            with CaptureWindow(predictor.model) if ranker is not None else nullcontext() as capture:
                outputs = predictor.model(
                    src_vid=video_features, src_vid_mask=video_mask,
                    src_txt=query_features, src_txt_mask=query_mask,
                )
                relevance = float(ranker.score(capture.features(outputs, count))[0]) if ranker is not None else None
            probabilities = functional.softmax(outputs["pred_logits"], -1)[0, :, 0].cpu()
            spans = span_cxw_to_xx(outputs["pred_spans"][0].cpu()) * (count * CLIP_DURATION)
            local_candidates = torch.cat([spans, probabilities[:, None]], dim=1).tolist()
            if ranker is not None:
                scores = combine_scores([r[2] for r in local_candidates], relevance, ranker.strength)
                local_candidates = [[row[0], row[1], score] for row, score in zip(local_candidates, scores)]
            ranked_windows.extend(globalize_candidates(local_candidates, start, end, record.duration, CLIP_DURATION))
            saliency = outputs["saliency_scores"][0, :count].half().cpu().tolist()
            if len(saliency) != count:
                raise RuntimeError("模型高光输出未覆盖完整窗口")
            saliency_windows.append((start, saliency))
            window_details.append({
                "window_id": window_id,
                "start": start * CLIP_DURATION,
                "end": round(min(end * CLIP_DURATION, record.duration), 4),
                "candidate_count": len(local_candidates),
                **({"relevance_logit": round(relevance, 6)} if relevance is not None else {}),
            })
            logger.info("Video %s query inference window %d/%d complete", record.video_id, window_id + 1, len(windows))
        ranked_windows.sort(key=lambda row: row[2], reverse=True)
        saliency_scores = merge_saliency(frame_count, saliency_windows)

    return {
        "query": query,
        "vid": str(record.path),
        "pred_relevant_windows": ranked_windows,
        "pred_saliency_scores": saliency_scores,
        "window_ranking": {
            "enabled": ranker is not None,
            "method": "window_head" if ranker is not None else "none",
            **({"strength": ranker.strength} if ranker is not None else {}),
        },
        "windowing": {
            "enabled": len(windows) > 1,
            "window_count": len(windows),
            "window_seconds": WINDOW_CLIPS * CLIP_DURATION,
            "overlap_seconds": WINDOW_OVERLAP_CLIPS * CLIP_DURATION,
            "feature_clip_count": frame_count,
            "score_aggregation": "window_reranked" if ranker is not None else "raw_model_score",
            "saliency_aggregation": "mean",
            "windows": window_details,
        },
    }, cache_hit


def _run_model(record: VideoRecord, request: SearchRequest) -> dict[str, Any]:
    started_at = time.perf_counter()
    with _inference_lock:
        predictor = _get_predictor()
        prediction, cache_hit = _predict_with_cached_features(
            predictor,
            record,
            request.query.strip(),
        )

    moments, postprocessing = postprocess_moments(
        prediction["pred_relevant_windows"],
        record.duration,
        method=request.nms_method,
        iou_threshold=request.nms_threshold,
        min_score=request.nms_min_score,
        top_k=request.top_k,
    )

    return {
        "video_id": record.video_id,
        "query": request.query.strip(),
        "duration": round(record.duration, 4),
        "moments": moments,
        "postprocessing": postprocessing,
        "saliency_scores": prediction["pred_saliency_scores"],
        "clip_duration": CLIP_DURATION,
        **({"windowing": prediction["windowing"]} if "windowing" in prediction else {}),
        **({"window_ranking": prediction["window_ranking"]} if "window_ranking" in prediction else {}),
        "model_version": (
            "qd-detr-slowfast-clip-main" if FEATURE_MODE == "slowfast_clip"
            else "qd-detr-clip-official"
        ),
        "video_feature_cache_hit": cache_hit,
        "inference_time_ms": round((time.perf_counter() - started_at) * 1000, 1),
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    repository_ready = QD_DETR_ROOT.is_dir() and CHECKPOINT_PATH.is_file()
    slowfast_ready = (
        FEATURE_MODE != "slowfast_clip"
        or (LIGHTHOUSE_ROOT.is_dir() and SLOWFAST_CHECKPOINT.is_file())
    )
    ffmpeg_ready = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
    ranker_ready = WINDOW_RANKER_PATH is None or WINDOW_RANKER_PATH.is_file()
    return {
        "status": "ready" if repository_ready and slowfast_ready and ffmpeg_ready and ranker_ready else "degraded",
        "model_loaded": _predictor is not None,
        "device": MODEL_DEVICE,
        "checkpoint": str(CHECKPOINT_PATH),
        "feature_mode": FEATURE_MODE,
        "model_label": "SlowFast + CLIP" if FEATURE_MODE == "slowfast_clip" else "CLIP",
        "slowfast_ready": slowfast_ready,
        "slowfast_batch_size": SLOWFAST_BATCH_SIZE,
        "feature_cache_entries": len(_video_feature_cache),
        "feature_cache_capacity": FEATURE_CACHE_SIZE,
        "window_ranking": {
            "configured": WINDOW_RANKER_PATH is not None,
            "loaded": _window_ranker is not None,
            "ready": ranker_ready,
        },
        "postprocessing": POSTPROCESSING.model_dump(),
        "max_video_duration": MAX_VIDEO_DURATION,
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "windowing": {
            "window_seconds": WINDOW_CLIPS * CLIP_DURATION,
            "overlap_seconds": WINDOW_OVERLAP_CLIPS * CLIP_DURATION,
            "feature_chunk_seconds": FEATURE_CHUNK_CLIPS * CLIP_DURATION,
        },
    }


@app.post("/api/videos", status_code=status.HTTP_201_CREATED)
async def upload_video(
    file: UploadFile = File(...),
    duration: float | None = Form(default=None),
) -> dict[str, Any]:
    del duration  # The backend trusts ffprobe rather than browser metadata.
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="仅支持 MP4、WebM 和 MOV 视频")

    video_id = uuid.uuid4().hex[:12]
    path = STORAGE_DIR / f"{video_id}{suffix}"
    written = 0
    try:
        with path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"视频不能超过 {MAX_UPLOAD_BYTES // 1024 // 1024} MB",
                    )
                output.write(chunk)

        actual_duration = await run_in_threadpool(_probe_duration, path)
        if actual_duration > MAX_VIDEO_DURATION + 0.05:
            raise HTTPException(
                status_code=422,
                detail=f"当前服务最多处理 {MAX_VIDEO_DURATION:g} 秒视频，请截短或调整服务时长配置",
            )

        record = VideoRecord(
            video_id=video_id,
            filename=Path(file.filename or path.name).name,
            path=path,
            duration=actual_duration,
        )
        _videos[video_id] = record
        return {
            "video_id": video_id,
            "filename": record.filename,
            "duration": round(actual_duration, 4),
            "status": "ready",
            "stream_url": f"/api/videos/{video_id}/stream",
            "inference_mode": f"qd-detr-{FEATURE_MODE.replace('_', '-')}",
        }
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


@app.get("/api/videos/{video_id}")
def get_video(video_id: str) -> dict[str, Any]:
    record = _videos.get(video_id)
    if record is None:
        raise HTTPException(status_code=404, detail="视频不存在或服务已重启，请重新上传")
    return {
        "video_id": video_id,
        "duration": round(record.duration, 4),
        "status": "ready",
        "progress": 100,
        "stage": "ready",
        "stream_url": f"/api/videos/{video_id}/stream",
        "error": None,
    }


@app.get("/api/videos/{video_id}/stream")
def stream_video(video_id: str) -> FileResponse:
    record = _videos.get(video_id)
    if record is None or not record.path.is_file():
        raise HTTPException(status_code=404, detail="视频不存在")
    media_types = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime"}
    return FileResponse(record.path, media_type=media_types.get(record.path.suffix.lower()))


@app.post("/api/search")
async def search(request: SearchRequest) -> dict[str, Any]:
    record = _videos.get(request.video_id)
    if record is None:
        raise HTTPException(status_code=404, detail="视频不存在或服务已重启，请重新上传")
    try:
        return await run_in_threadpool(_run_model, record, request)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"模型推理失败：{exc}") from exc


if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

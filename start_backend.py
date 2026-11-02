"""Portable FrameSeek entry point. Environment overrides remain configurable."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent


def configure(args):
    from backend.config import configured_path
    defaults = {
        'QD_DETR_ROOT': 'QD-DETR',
        'QD_DETR_CHECKPOINT': 'models/tacos_model_best.ckpt' if args.profile == 'tacos' else 'models/qv_model_best.ckpt',
        'QD_DETR_FEATURE_MODE': 'slowfast_clip',
        'QD_DETR_CLIP_MODEL': 'models/ViT-B-32.pt',
        'LIGHTHOUSE_ROOT': 'lighthouse',
        'SLOWFAST_CHECKPOINT': 'models/SLOWFAST_8x8_R50.pkl',
        'QD_DETR_WINDOW_RANKER': 'models/window_ranker.pt' if args.profile == 'tacos' else '',
        'QD_DETR_DEVICE': 'auto',
        'VIDEO_STORAGE_DIR': 'storage',
        'FRONTEND_DIST': 'frontend/dist',
        'CORS_ORIGINS': 'http://localhost:5173,http://127.0.0.1:5173',
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    if args.device is not None:
        os.environ['QD_DETR_DEVICE'] = args.device
    if args.slowfast_batch_size is not None:
        os.environ['SLOWFAST_BATCH_SIZE'] = str(args.slowfast_batch_size)
    if args.cors_origin:
        os.environ['CORS_ORIGINS'] = ','.join(args.cors_origin)
    for name in ('QD_DETR_ROOT', 'QD_DETR_CHECKPOINT', 'QD_DETR_CLIP_MODEL',
                 'LIGHTHOUSE_ROOT', 'SLOWFAST_CHECKPOINT', 'VIDEO_STORAGE_DIR',
                 'FRONTEND_DIST', 'QD_DETR_WINDOW_RANKER'):
        if os.environ[name].strip():
            os.environ[name] = str(configured_path(name, defaults[name]))
    binary_dir = str(Path(sys.executable).resolve().parent)
    os.environ['PATH'] = binary_dir + os.pathsep + os.environ.get('PATH', '')


def main():
    # Load only this checkout's .env, never a caller's unrelated settings.
    try:
        from dotenv import load_dotenv
    except ImportError:
        raise SystemExit('Install requirements.txt first (python-dotenv is missing).')
    load_dotenv(ROOT / '.env', override=False)
    parser = argparse.ArgumentParser(description='FrameSeek model/backend service')
    parser.add_argument('--profile', choices=['qv', 'tacos'], default=os.getenv('FRAMESEEK_PROFILE', 'qv'))
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--host', default=os.getenv('FRAMESEEK_HOST', '127.0.0.1'))
    parser.add_argument('--port', type=int, default=int(os.getenv('FRAMESEEK_PORT', '8000')))
    parser.add_argument('--cors-origin', action='append')
    parser.add_argument('--slowfast-batch-size', type=int)
    parser.add_argument('--check', action='store_true', help='Validate files, imports and FFmpeg without loading weights')
    parser.add_argument('--warmup', action='store_true', help='Also load the selected model components')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535')
    if args.slowfast_batch_size is not None and args.slowfast_batch_size < 1:
        parser.error('SlowFast batch size must be positive')
    configure(args)
    modules = ['torch', 'torchvision', 'fastapi', 'uvicorn', 'multipart', 'numpy',
               'scipy', 'pandas', 'PIL', 'ffmpeg', 'ftfy', 'regex', 'tqdm',
               'einops', 'fvcore', 'iopath', 'yaml']
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        parser.error('Missing dependencies: ' + ', '.join(missing) + '. See README.md.')
    import torch
    device = os.environ['QD_DETR_DEVICE']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device not in {'cpu', 'cuda'}:
        parser.error('QD_DETR_DEVICE must be auto, cuda or cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; install matching PyTorch/driver or use --device cpu')
    os.environ['QD_DETR_DEVICE'] = device
    required = ['QD_DETR_CHECKPOINT', 'QD_DETR_CLIP_MODEL']
    if os.environ['QD_DETR_FEATURE_MODE'] == 'slowfast_clip':
        required.append('SLOWFAST_CHECKPOINT')
    if os.environ['QD_DETR_WINDOW_RANKER'].strip():
        required.append('QD_DETR_WINDOW_RANKER')
    missing_files = [f'{k}: {os.environ[k]}' for k in required if not Path(os.environ[k]).is_file()]
    if missing_files:
        parser.error('Missing model assets (see models/README.md):\n' + '\n'.join(missing_files))
    vocab = Path(os.environ['QD_DETR_ROOT']) / 'run_on_video/clip/bpe_simple_vocab_16e6.txt.gz'
    if not vocab.is_file():
        parser.error(f'Missing CLIP tokenizer vocabulary: {vocab}')
    for binary in ('ffmpeg', 'ffprobe'):
        if shutil.which(binary) is None:
            parser.error(f'{binary} not found on PATH; see README.md')
    from backend import app as api
    print(f'Profile={args.profile}, device={device}, NMS={api.POSTPROCESSING.model_dump()}, '
          f'window={api.WINDOW_CLIPS * 2}s, overlap={api.WINDOW_OVERLAP_CLIPS * 2}s, '
          f'max={api.MAX_VIDEO_DURATION}s', flush=True)
    if args.warmup:
        predictor = api._get_predictor()
        if api.FEATURE_MODE == 'slowfast_clip':
            api._get_slowfast_components()
        if api.WINDOW_RANKER_PATH:
            api._get_window_ranker(predictor)
        print('Selected model components loaded.', flush=True)
    if args.check:
        print('Environment check passed. Use scripts/smoke_test.py for real video inference.', flush=True)
        return
    import uvicorn
    # Keep one worker: uploaded-video registry and feature cache are process-local.
    uvicorn.run(api.app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

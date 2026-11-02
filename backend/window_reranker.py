"""Optional checkpoint-bound window relevance head; leaves QD-DETR frozen."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn

FEATURE_VERSION = "qd-global-mean-max-v1"
FEATURE_DIM = 773  # 3 * hidden_dim(256) + five window/candidate statistics


class WindowHead(nn.Module):
    def __init__(self, dimension=FEATURE_DIM, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(),
                                 nn.Dropout(.2), nn.Linear(hidden, 1))

    def forward(self, features):
        return self.net(features).squeeze(-1)


class CaptureWindow:
    """Capture only the positive transformer pass, removing hooks on errors."""
    def __init__(self, model):
        self.model = model
        self.memory = None
        self.hook = None

    def __enter__(self):
        def capture(_module, _inputs, output):
            if self.memory is None:
                self.memory = (output[2].detach(), output[3].detach())
        self.hook = self.model.transformer.register_forward_hook(capture)
        return self

    def features(self, outputs, clip_count):
        if self.memory is None:
            raise RuntimeError("Window ranker did not receive transformer representations")
        local, global_ = self.memory
        if local.shape[1] != clip_count or global_.shape[-1] != 256:
            raise ValueError("Window ranker requires unpadded QD-DETR windows with hidden dimension 256")
        logits = (outputs['pred_logits'][..., 0] - outputs['pred_logits'][..., 1]).detach().float()
        best = logits.argmax(-1)
        widths = outputs['pred_spans'].detach()[torch.arange(len(best), device=best.device), best, 1].float()
        stats = torch.stack([logits.max(-1).values, logits.mean(-1), logits.std(-1, unbiased=False),
                             widths, torch.full_like(widths, clip_count/75.)], dim=-1)
        features = torch.cat([global_.float(), local.float().mean(1), local.float().amax(1), stats], dim=-1)
        if features.shape[-1] != FEATURE_DIM or not torch.isfinite(features).all():
            raise ValueError("Invalid window-ranker representation")
        return features

    def __exit__(self, *_):
        if self.hook is not None:
            self.hook.remove()
        self.memory = None


def combine_scores(scores, window_logit, strength):
    """Ranking score, not a calibrated probability. Zero is an exact bypass."""
    if not math.isfinite(strength) or strength < 0 or not math.isfinite(window_logit):
        raise ValueError("Window ranker strengths and logits must be finite")
    if strength == 0:
        return list(scores)
    result = []
    for score in scores:
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Candidate scores must lie in [0, 1]")
        p = min(1-1e-7, max(1e-7, float(score)))
        value = (math.log(p/(1-p)) + strength*window_logit)/(1+strength)
        probability = 1/(1+math.exp(-value)) if value >= 0 else math.exp(value)/(1+math.exp(value))
        result.append(probability)
    return result


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


class WindowRanker:
    def __init__(self, path, base_checkpoint, device, strength=None):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint['feature_version'] != FEATURE_VERSION or checkpoint['feature_dim'] != FEATURE_DIM:
            raise ValueError("Unsupported window-ranker feature format")
        if checkpoint['base_checkpoint_sha256'] != file_sha256(base_checkpoint):
            raise ValueError("窗口评分器与当前 QD-DETR 权重不匹配，请使用配套权重")
        self.strength = float(checkpoint['strength'] if strength is None else strength)
        if not math.isfinite(self.strength) or not 0 <= self.strength <= 8:
            raise ValueError("Window ranker strength must be between 0 and 8")
        self.head = WindowHead(hidden=checkpoint['hidden']).to(device)
        self.head.load_state_dict(checkpoint['state_dict'], strict=True)
        self.head.eval()
        self.mean = checkpoint['mean'].to(device)
        self.scale = checkpoint['scale'].to(device)
        if self.mean.shape != (FEATURE_DIM,) or self.scale.shape != (FEATURE_DIM,):
            raise ValueError("Invalid window-ranker normalization shape")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("Invalid window-ranker normalization")
        if not all(torch.isfinite(p).all() for p in self.head.parameters()):
            raise ValueError("Non-finite window-ranker parameters")
        self.base_checkpoint_sha256 = checkpoint['base_checkpoint_sha256']
        self.path = str(Path(path).resolve())

    @torch.no_grad()
    def score(self, features):
        value = self.head((features-self.mean)/self.scale)
        if not torch.isfinite(value).all():
            raise RuntimeError("Window ranker produced non-finite scores")
        return value

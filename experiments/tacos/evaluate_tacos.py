"""Zero-shot TACoS retrieval evaluation using the application's inference path.

Uses public precomputed CLIP/SlowFast/text features, not raw-video decoding.
No fitting, threshold selection, or training is performed on either split.
"""
from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tiou(a, b):
    overlap = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = a[1] - a[0] + b[1] - b[0] - overlap
    return overlap / union if union > 0 else 0.0


def best_iou(predictions, ground_truth):
    return max((tiou(p, g) for p in predictions for g in ground_truth), default=0.0)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_model(path):
    from qd_detr.model import QDDETR, build_position_encoding, build_transformer
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    args = checkpoint['opt']
    pos, txt_pos = build_position_encoding(args)
    model = QDDETR(
        build_transformer(args), pos, txt_pos,
        txt_dim=args.t_feat_dim, vid_dim=args.v_feat_dim, num_queries=args.num_queries,
        input_dropout=args.input_dropout, aux_loss=args.aux_loss,
        contrastive_align_loss=args.contrastive_align_loss,
        contrastive_hdim=args.contrastive_hdim, span_loss_type=args.span_loss_type,
        use_txt_pos=args.use_txt_pos, n_input_proj=args.n_input_proj,
    )
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.cuda().eval()


class FirstWindowRecorder:
    def __init__(self, model):
        self.model = model
        self.first = None

    def __call__(self, **inputs):
        output = self.model(**inputs)
        if self.first is None:
            from qd_detr.span_utils import span_cxw_to_xx
            count = inputs['src_vid'].shape[1]
            spans = span_cxw_to_xx(output['pred_spans'][0].cpu()) * (count * 2.0)
            scores = F.softmax(output['pred_logits'], -1)[0, :, 0].cpu()
            self.first = (count, torch.cat([spans, scores[:, None]], dim=1).tolist())
        return output


def load_visual(root, vid, duration):
    parts, lengths = [], []
    for modality, dimension in [('slowfast', 2304), ('clip', 512)]:
        with np.load(root / modality / f'{vid}.npz', allow_pickle=False) as archive:
            data = archive['features']
        if data.ndim != 2 or data.shape[1] != dimension or not np.isfinite(data).all():
            raise ValueError(f'Unexpected {modality} features for {vid}: {data.shape}')
        lengths.append(len(data))
        parts.append(F.normalize(torch.from_numpy(data.astype(np.float32)), dim=-1, eps=1e-5))
    length = min(lengths)
    # The official feature release can have one extra final clip per modality.
    # Follow its common-length trim, without rescaling the full video to 150s.
    if abs(length * 2.0 - duration) > 2.05:
        raise ValueError(f'Two-second feature grid does not match annotation: {vid}, {length}, {duration}')
    visual = torch.cat([part[:length] for part in parts], dim=1)
    return visual, {'lengths_slowfast_clip': lengths, 'common_length': length,
                    'coverage_seconds': length * 2.0, 'duration': duration,
                    'uncovered_tail_seconds': max(0.0, duration - length * 2.0)}


def summarize(records, mode):
    if not records:
        return {'query_count': 0, 'video_count': 0}
    ret = {'query_count': len(records), 'video_count': len({r['vid'] for r in records})}
    for k in (1, 5):
        for threshold in (.3, .5, .7):
            ret[f'R{k}@{threshold}'] = round(100 * statistics.mean(r[f'{mode}_iou{k}'] >= threshold for r in records), 3)
    ret['top1_mean_iou'] = round(100 * statistics.mean(r[f'{mode}_iou1'] for r in records), 3)
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qd-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['test', 'val'], required=True)
    parser.add_argument('--min-duration', type=float, default=150)
    parser.add_argument('--max-duration', type=float, default=600)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--exclude-predictions', type=Path)
    options = parser.parse_args()
    assert tiou([0, 10], [0, 10]) == 1
    assert tiou([0, 10], [10, 20]) == 0
    assert abs(tiou([0, 10], [5, 15]) - 1 / 3) < 1e-12
    sys.path.insert(0, str(options.qd_root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    os.environ['QD_DETR_ROOT'] = str(options.qd_root)
    os.environ['QD_DETR_CHECKPOINT'] = str(options.checkpoint)
    os.environ['QD_DETR_FEATURE_MODE'] = 'slowfast_clip'
    os.environ['QD_DETR_NMS_METHOD'] = 'hard'
    os.environ['QD_DETR_NMS_THRESHOLD'] = '0.5'
    os.environ['QD_DETR_NMS_MIN_SCORE'] = '0'
    os.environ['QD_DETR_WINDOW_CLIPS'] = '75'
    os.environ['QD_DETR_WINDOW_OVERLAP_CLIPS'] = '15'
    from backend import app
    from backend.sliding_window import globalize_candidates
    from backend.temporal_postprocessing import postprocess_moments
    torch.set_num_threads(4)
    torch.manual_seed(2018)
    np.random.seed(2018)
    output = options.output
    output.mkdir(parents=True, exist_ok=True)
    source_rows = read_jsonl(options.annotations)
    rows = [r for r in source_rows if options.min_duration < r['duration'] <= options.max_duration]
    if options.exclude_predictions:
        excluded = {r['qid'] for r in read_jsonl(options.exclude_predictions)}
        rows = [r for r in rows if r['qid'] not in excluded]
    if options.limit:
        rows = rows[:options.limit]
    assert len({r['qid'] for r in rows}) == len(rows)
    by_video = defaultdict(list)
    for row in rows:
        if not row['relevant_windows'] or any(not 0 <= a < b <= row['duration'] + .05 for a, b in row['relevant_windows']):
            raise ValueError(f'Invalid annotation {row["qid"]}')
        by_video[row['vid']].append(row)
        for name in [options.features/'clip_text'/f'{row["qid"]}.npz', options.features/'clip'/f'{row["vid"]}.npz', options.features/'slowfast'/f'{row["vid"]}.npz']:
            if not name.is_file():
                raise FileNotFoundError(name)
    manifest = {
        'dataset': 'TACoS', 'split': options.split,
        'evaluation_type': 'cross-dataset zero-shot, precomputed-feature evaluation',
        'duration_filter': {'exclusive_min': options.min_duration, 'inclusive_max': options.max_duration},
        'source_query_count': len(source_rows), 'selected_query_count': len(rows),
        'selected_video_count': len(by_video), 'checkpoint': str(options.checkpoint),
        'checkpoint_sha256': sha256(options.checkpoint),
        'annotations_sha256': sha256(options.annotations),
        'feature_source': 'Lighthouse TACoS public feature release, Google Drive id 1rYzme9JNAk3niH1K81wgT13pOMn005jb',
        'feature_normalization': 'per-modality L2; [SlowFast2304, CLIP512]; local two-dimensional TEF',
        'window_seconds': 150, 'overlap_seconds': 30, 'nms': {'method': 'hard', 'threshold': .5},
        'score_aggregation': 'raw model foreground softmax score; no cross-window calibration',
        'source_hashes': {name: sha256(PROJECT_ROOT/'backend'/name) for name in ['app.py','sliding_window.py','temporal_postprocessing.py','video_features.py']},
        'python': sys.version, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
        'training_or_tuning': 'none',
        'excluded_predictions': str(options.exclude_predictions) if options.exclude_predictions else None,
    }
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print('START', json.dumps(manifest), flush=True)
    recorder = FirstWindowRecorder(load_model(options.checkpoint))
    text_lookup = SimpleNamespace(current=None)
    predictor = SimpleNamespace(device='cuda', clip_len=2, model=recorder,
                                feature_extractor=SimpleNamespace(encode_text=lambda _: [text_lookup.current]))
    visual_lookup = SimpleNamespace(current=None)
    app._get_video_features = lambda *_: (visual_lookup.current, True)
    records, coverage = [], {}
    started = time.monotonic()
    with (output/'predictions.jsonl').open('w') as stream:
        for video_index, (vid, queries) in enumerate(by_video.items(), 1):
            duration = queries[0]['duration']
            visual_lookup.current, coverage[vid] = load_visual(options.features, vid, duration)
            record = app.VideoRecord(video_id=vid, filename=f'{vid}.mp4', path=Path(f'{vid}.mp4'), duration=duration)
            for query in queries:
                with np.load(options.features/'clip_text'/f'{query["qid"]}.npz', allow_pickle=False) as archive:
                    text = archive['last_hidden_state']
                if text.ndim != 2 or text.shape[1] != 512 or not 0 < len(text) <= 77 or not np.isfinite(text).all():
                    raise ValueError(f'Unexpected text features {query["qid"]}: {text.shape}')
                text_lookup.current = torch.from_numpy(text.astype(np.float32))
                recorder.first = None
                prediction, _ = app._predict_with_cached_features(predictor, record, query['query'])
                raw = prediction['pred_relevant_windows']
                sliding, info = postprocess_moments(raw, duration, method='hard', iou_threshold=.5, top_k=len(raw))
                count, first = recorder.first
                first = globalize_candidates(first, 0, count, duration)
                prefix, _ = postprocess_moments(first, duration, method='hard', iou_threshold=.5, top_k=len(first))
                sliding_rows = [[m['start'], m['end'], m['score']] for m in sliding]
                prefix_rows = [[m['start'], m['end'], m['score']] for m in prefix]
                gt = query['relevant_windows']
                values = {}
                for mode, preds in [('sliding', sliding_rows), ('prefix150', prefix_rows)]:
                    for k in (1, 5):
                        values[f'{mode}_iou{k}'] = best_iou(preds[:k], gt)
                raw_oracle = best_iou(raw, gt)
                nms_oracle = best_iou(sliding_rows, gt)
                if values['sliding_iou1'] >= .5:
                    category = 'hit_top1'
                elif values['sliding_iou5'] >= .5:
                    category = 'hit_rank2_to_5'
                elif nms_oracle >= .5:
                    category = 'correct_candidate_outside_top5'
                elif raw_oracle >= .5:
                    category = 'correct_candidate_removed_by_nms'
                else:
                    category = 'no_correct_candidate_iou05'
                row = {
                    'qid': query['qid'], 'vid': vid, 'query': query['query'], 'duration': duration,
                    'ground_truth': gt, 'late_target': all(a >= 150 for a, _ in gt),
                    'crosses_window_end': any(a < w['end'] < b for a, b in gt for w in prediction['windowing']['windows'][:-1]),
                    'target_duration': max(b-a for a,b in gt), 'window_count': prediction['windowing']['window_count'],
                    **values, 'raw_oracle_iou': raw_oracle, 'nms_oracle_iou': nms_oracle,
                    'error_category_iou05': category, 'sliding_top1': sliding_rows[0] if sliding_rows else [],
                    'prefix150_top1': prefix_rows[0] if prefix_rows else [],
                }
                records.append(row)
                stream.write(json.dumps(dict(row, sliding=sliding_rows, prefix150=prefix_rows, raw_candidates=raw), ensure_ascii=False)+'\n')
            stream.flush()
            print('PROGRESS', video_index, '/', len(by_video), vid, 'queries',len(records), '/', len(rows), 'seconds',round(time.monotonic()-started,1), flush=True)
    groups = {
        'all_selected': records,
        'target_starts_after_150': [r for r in records if r['late_target']],
        'target_starts_before_150': [r for r in records if not r['late_target']],
        'target_crosses_window_end': [r for r in records if r['crosses_window_end']],
        'target_shorter_than_5_seconds': [r for r in records if r['target_duration'] < 5],
        'video_up_to_300_seconds': [r for r in records if r['duration'] <= 300],
        'video_over_300_seconds': [r for r in records if r['duration'] > 300],
    }
    metrics = {
        'groups': {name: {mode:summarize(group,mode) for mode in ['prefix150','sliding']} for name,group in groups.items()},
        'candidate_diagnostics': {
            'raw_oracle_R_at_05': round(100*statistics.mean(r['raw_oracle_iou']>=.5 for r in records),3),
            'after_nms_oracle_R_at_05': round(100*statistics.mean(r['nms_oracle_iou']>=.5 for r in records),3),
            'error_counts': dict(Counter(r['error_category_iou05'] for r in records)),
        },
        'median_target_seconds': statistics.median(r['target_duration'] for r in records),
        'inference_seconds': round(time.monotonic()-started,3),
        'limitations': [
            'Frozen QVHighlights-trained checkpoint; no TACoS training or parameter tuning.',
            'Duration-filtered subset, not the full TACoS official test benchmark.',
            'Public precomputed features; raw video decoding, upload and UI are not evaluated here.',
            'Prefix150 is a diagnostic first-window-only comparison, not a model trained on TACoS.',
            'Common-length modality trim can omit a fraction of the final two-second interval; see coverage.json.',
        ],
    }
    (output/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
    (output/'coverage.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2))
    with (output/'per_query.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    per_video = {vid:{mode:summarize([r for r in records if r['vid']==vid],mode) for mode in ['prefix150','sliding']} for vid in by_video}
    (output/'per_video.json').write_text(json.dumps(per_video,ensure_ascii=False,indent=2))
    print('FINISHED',json.dumps(metrics,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()

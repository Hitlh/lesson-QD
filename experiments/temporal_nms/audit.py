"""Read-only audit of downloaded validation predictions; no model inference."""
import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'QD-DETR'))
from standalone_eval.eval import compute_mr_ap, compute_mr_r1
from utils.temporal_nms import temporal_nms


def load(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def iou(a, b):
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - intersection
    return intersection / union if union > 0 else 0.0


def nms(rows, threshold):
    kept = []
    for row in sorted(rows, key=lambda x: x[2], reverse=True):
        if all(iou(row, old) <= threshold for old in kept):
            kept.append(list(row))
    return kept


def soft_nms(rows, threshold=0.5):
    pool = [list(x) for x in rows]
    kept = []
    while pool:
        pool.sort(key=lambda x: x[2], reverse=True)
        best = pool.pop(0)
        if best[2] <= 0:
            break
        kept.append(best)
        for row in pool:
            overlap = iou(best, row)
            if overlap > threshold:
                row[2] *= 1 - overlap
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-metrics', type=Path, help='Optional saved official MR metrics for parity checking')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    predictions = load(args.predictions)
    ground_truth = load(args.annotations)
    gt = {x['qid']: x for x in ground_truth}
    assert len(predictions) == len(gt) == len({x['qid'] for x in predictions})
    assert {x['qid'] for x in predictions} == set(gt)
    invalid = [(x['qid'], row) for x in predictions for row in x['pred_relevant_windows']
               if not all(math.isfinite(v) for v in row) or row[1] <= row[0]]
    assert not invalid, invalid[:5]
    assert all(x['pred_relevant_windows'] == sorted(x['pred_relevant_windows'], key=lambda r: r[2], reverse=True) for x in predictions)
    total = len(predictions)
    report = {
        'predictions_sha256': hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        'limitations': [
            'Offline best-checkpoint validation predictions, not a fresh raw-video backend run.',
            'Saved validation timestamps have official postprocessing including two-second rounding.',
            'Temporal overlap does not by itself prove semantic redundancy.',
            'Threshold comparisons on validation data are exploratory, not held-out test improvements.',
        ],
        'query_count': total,
        'video_count': len({x['vid'] for x in predictions}),
        'candidates_per_query': sorted({len(x['pred_relevant_windows']) for x in predictions}),
        'definition': 'high overlap means temporal IoU strictly greater than threshold; queries, not videos, are the denominator',
        'overlap_statistics': [],
    }
    for k in [2, 3, 5, 10]:
        for threshold in [0.5, 0.7, 0.9]:
            flagged = exact = all_pairwise = high_score = 0
            removed = []
            for x in predictions:
                rows = x['pred_relevant_windows'][:k]
                pairs = [(a, b) for i, a in enumerate(rows) for b in rows[i+1:]]
                flagged += any(iou(a, b) > threshold for a, b in pairs)
                high_score += any(iou(a, b) > threshold and min(a[2], b[2]) >= 0.8 for a, b in pairs)
                exact += any(a[:2] == b[:2] for a, b in pairs)
                all_pairwise += bool(pairs) and all(iou(a, b) > threshold for a, b in pairs)
                kept = nms(rows, threshold)
                assert kept == temporal_nms(rows, threshold, max_after_nms=10)
                removed.append(len(rows) - len(kept))
            report['overlap_statistics'].append({
                'top_k': k, 'iou_threshold': threshold,
                'queries_with_overlap': flagged,
                'queries_with_overlap_pct': round(100 * flagged / total, 3),
                'queries_with_high_score_overlap': high_score,
                'queries_with_high_score_overlap_pct': round(100 * high_score / total, 3),
                'queries_with_exact_duplicate': exact,
                'queries_all_candidates_pairwise_overlap': all_pairwise,
                'average_candidates_after_nms_within_top_k': round(k - statistics.mean(removed), 4),
            })
    per_query = []
    for x in predictions:
        rows = x['pred_relevant_windows']
        top3 = rows[:3]
        pairs = [(a,b) for i,a in enumerate(top3) for b in top3[i+1:]]
        per_query.append({
            'qid': x['qid'], 'vid': x['vid'], 'query': x['query'],
            'top1_top2_iou': round(iou(rows[0],rows[1]), 6),
            'top3_max_pair_iou': round(max(iou(a,b) for a,b in pairs), 6),
            'top3_min_pair_iou': round(min(iou(a,b) for a,b in pairs), 6),
            'top3_kept_at_05': len(nms(top3, 0.5)),
            'top3': json.dumps(top3),
            'top3_after_global_hard05': json.dumps(nms(rows,0.5)[:3]),
            'ground_truth': json.dumps(gt[x['qid']]['relevant_windows']),
        })
    with (args.output / 'per_query_overlap.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_query[0]))
        writer.writeheader()
        writer.writerows(per_query)
    report['examples'] = sorted(per_query, key=lambda x: (x['top3_min_pair_iou'],x['top3_max_pair_iou']), reverse=True)[:5]
    modes = {'none': lambda r:r}
    modes.update({f'hard_{t}': (lambda r,t=t:nms(r,t)) for t in [0.3,0.5,0.7,0.9]})
    modes['soft_linear_0.5'] = soft_nms
    results = []
    for mode, fn in modes.items():
        modified = [dict(x, pred_relevant_windows=fn(x['pred_relevant_windows'])) for x in predictions]
        ap = compute_mr_ap(modified,ground_truth,num_workers=1)
        r1 = compute_mr_r1(modified,ground_truth)
        result = {'method':mode,'mAP':ap['average'],'mAP@0.5':ap['0.5'],'mAP@0.75':ap['0.75'],
                  'R1@0.5':r1['0.5'],'R1@0.7':r1['0.7'],
                  'average_retained_candidates':round(statistics.mean(len(x['pred_relevant_windows']) for x in modified),4)}
        for k in [3,5]:
            for threshold in [0.5,0.7]:
                hits = sum(any(iou(p,g) >= threshold for p in x['pred_relevant_windows'][:k]
                               for g in gt[x['qid']]['relevant_windows']) for x in modified)
                result[f'queries_any_gt_hit_R{k}@{threshold}_pct'] = round(100*hits/total,3)
        if mode == 'none' and args.baseline_metrics:
            stored = json.loads(args.baseline_metrics.read_text())['brief']
            for key in ['mAP','mAP@0.5','mAP@0.75','R1@0.5','R1@0.7']:
                assert result[key] == stored['MR-full-'+key], (key,result[key],stored)
        results.append(result)
        print(json.dumps(result),flush=True)
    report['metric_comparison'] = results
    with (args.output/'metric_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    (args.output/'audit_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('SUMMARY',json.dumps({k:v for k,v in report.items() if k not in ['examples','metric_comparison']},ensure_ascii=False))
    print('EXAMPLES',json.dumps(report['examples'],ensure_ascii=False))


if __name__ == '__main__':
    main()

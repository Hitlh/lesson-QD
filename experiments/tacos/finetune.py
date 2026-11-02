"""Small, bounded TACoS adaptation using the native checkpoint architecture.

One positive and, where available, one disjoint negative sliding window per
query per epoch. Validation is full-video retrieval, never GT-selected windows.
"""
from __future__ import annotations

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from evaluate_tacos import load_visual, read_jsonl, sha256, best_iou
from backend.sliding_window import build_windows, globalize_candidates
from backend.temporal_postprocessing import postprocess_moments


class Features:
    def __init__(self, rows, root):
        self.visual, self.text, self.coverage = {}, {}, {}
        for row in rows:
            vid, qid = row['vid'], row['qid']
            assert len(row['relevant_windows']) == 1
            a, b = row['relevant_windows'][0]
            assert 0 <= a < b <= row['duration'] + .05
            if vid not in self.visual:
                self.visual[vid], self.coverage[vid] = load_visual(root, vid, row['duration'])
            if qid not in self.text:
                with np.load(root/'clip_text'/f'{qid}.npz', allow_pickle=False) as archive:
                    text = archive['last_hidden_state'].astype(np.float32)
                assert text.ndim == 2 and text.shape[1] == 512 and 0 < len(text) <= 32
                assert np.isfinite(text).all()
                self.text[qid] = F.normalize(torch.from_numpy(text), dim=-1, eps=1e-5)

    def sample(self, row, window, positive):
        start, end = window
        visual = self.visual[row['vid']][start:end]
        count = len(visual)
        t = torch.arange(count, dtype=torch.float32) / count
        video = torch.cat([visual, torch.stack([t, t + 1/count], dim=1)], dim=-1)
        a, b = row['relevant_windows'][0]
        # The model and application scale normalized spans by count * 2.
        if positive:
            a, b = max(a, start*2)-start*2, min(b, end*2)-start*2
            assert 0 <= a < b <= count*2 + 1e-5
            span = torch.tensor([[(a+b)/(2*count*2), (b-a)/(count*2)]], dtype=torch.float32)
        else:
            assert b <= start*2 or a >= end*2
            span = torch.empty((0, 2), dtype=torch.float32)
        return {'video': video, 'text': self.text[row['qid']], 'span': span,
                'qid': row['qid'], 'start': start, 'end': end, 'positive': positive}


class WindowPairs(Dataset):
    def __init__(self, rows, features):
        self.rows, self.features, self.pools = rows, features, []
        for row in rows:
            a, b = row['relevant_windows'][0]
            windows = build_windows(len(features.visual[row['vid']]))
            coverage = [max(0, min(b, e*2)-max(a, s*2))/(b-a) for s, e in windows]
            # Retain at least half of the target, or the best available window
            # for events longer than two windows. Never mark partial overlaps negative.
            positive = [w for w, fraction in zip(windows, coverage) if fraction >= .5]
            if not positive:
                maximum = max(coverage)
                assert maximum > 0, f'Uncovered training target: {row["qid"]}'
                positive = [w for w, fraction in zip(windows, coverage) if fraction == maximum]
            negative = [w for w, fraction in zip(windows, coverage) if fraction == 0]
            self.pools.append((positive, negative))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        positive, negative = self.pools[index]
        pair = [self.features.sample(row, random.choice(positive), True)]
        if negative:
            pair.append(self.features.sample(row, random.choice(negative), False))
        return pair


def collate_pairs(pairs):
    return [sample for pair in pairs for sample in pair]


def epoch_window_batches(dataset, maximum):
    """Sample once per query, then batch globally by exact window length."""
    groups = {}
    for index, (positive, negative) in enumerate(dataset.pools):
        choices = [(random.choice(positive), True)]
        if negative:
            choices.append((random.choice(negative), False))
        for window, label in choices:
            groups.setdefault(window[1]-window[0], []).append((index, window, label))
    batches = []
    for group in groups.values():
        random.shuffle(group)
        batches.extend(group[i:i+maximum] for i in range(0, len(group), maximum))
    random.shuffle(batches)
    return batches


def inputs(samples, device='cuda'):
    batch, vlen, tlen = len(samples), max(len(s['video']) for s in samples), max(len(s['text']) for s in samples)
    video = torch.zeros(batch, vlen, 2818)
    text = torch.zeros(batch, tlen, 512)
    vmask, tmask = torch.zeros(batch, vlen), torch.zeros(batch, tlen)
    for i, sample in enumerate(samples):
        n, m = len(sample['video']), len(sample['text'])
        video[i, :n], text[i, :m] = sample['video'], sample['text']
        vmask[i, :n], tmask[i, :m] = 1, 1
    return {k: value.to(device) for k, value in dict(src_vid=video, src_txt=text, src_vid_mask=vmask, src_txt_mask=tmask).items()}


def forward_by_length(model, samples):
    # The upstream T2V attention's repeated 3-D padding mask is sensitive to
    # mixed video lengths. Avoid video padding in both training and evaluation;
    # the deployed single-query path also has no padded video tokens.
    groups = {}
    for sample in samples:
        groups.setdefault(len(sample['video']), []).append(sample)
    ordered, outputs = [], []
    for group in groups.values():
        ordered.extend(group)
        outputs.append(model(**inputs(group)))
    merged = {key: torch.cat([o[key] for o in outputs], dim=0) for key in ('pred_logits', 'pred_spans')}
    if 'aux_outputs' in outputs[0]:
        merged['aux_outputs'] = [
            {key: torch.cat([o['aux_outputs'][i][key] for o in outputs], dim=0)
             for key in ('pred_logits', 'pred_spans')}
            for i in range(len(outputs[0]['aux_outputs']))]
    return merged, ordered


def summarize(rows):
    n = len(rows)
    result = {'queries': n, 'videos': len({r['vid'] for r in rows})}
    for k in (1, 5):
        for threshold in (.3, .5, .7):
            hits = sum(r[f'iou{k}'] >= threshold for r in rows)
            result[f'R{k}@{threshold}'] = round(100*hits/n, 3)
            result[f'R{k}@{threshold}_hits'] = hits
    return result


@torch.no_grad()
def evaluate(model, rows, features, destination, batch_size=32):
    from qd_detr.span_utils import span_cxw_to_xx
    model.eval()
    queues, captures = {}, {r['qid']: [] for r in rows}
    def flush(length):
        pending = queues[length]
        if not pending:
            return
        output = model(**inputs([s for _, s in pending]))
        spans = span_cxw_to_xx(output['pred_spans'].cpu())
        scores = output['pred_logits'].softmax(-1)[:, :, 0].cpu()
        for i, (row, sample) in enumerate(pending):
            count = sample['end']-sample['start']
            local = torch.cat([spans[i] * (count*2), scores[i, :, None]], dim=-1).tolist()
            captures[row['qid']].append((sample['start'], globalize_candidates(local, sample['start'], sample['end'], row['duration'])))
        pending.clear()
    for row in rows:
        for window in build_windows(len(features.visual[row['vid']])):
            # Evaluation input construction uses no annotations.
            start, end = window
            visual = features.visual[row['vid']][start:end]
            count = len(visual)
            t = torch.arange(count, dtype=torch.float32)/count
            sample = {'video': torch.cat([visual, torch.stack([t,t+1/count],-1)],-1),
                      'text': features.text[row['qid']], 'start': start, 'end': end}
            pending = queues.setdefault(count, [])
            pending.append((row, sample))
            if len(pending) == batch_size:
                flush(count)
    for length in queues:
        flush(length)
    results = []
    with destination.open('w') as stream:
        for row in rows:
            raw = [candidate for _, candidates in sorted(captures[row['qid']], key=lambda item:item[0]) for candidate in candidates]
            raw.sort(key=lambda p:p[2], reverse=True)
            kept, _ = postprocess_moments(raw, row['duration'], method='hard', iou_threshold=.5, top_k=5)
            predictions = [[m['start'],m['end'],m['score']] for m in kept]
            result = {k:row[k] for k in ('qid','vid','query','duration')}
            result.update(ground_truth=row['relevant_windows'], predictions=predictions, raw_candidates=raw,
                          iou1=best_iou(predictions[:1],row['relevant_windows']),
                          iou5=best_iou(predictions[:5],row['relevant_windows']))
            results.append(result)
            stream.write(json.dumps(result,ensure_ascii=False)+'\n')
    groups = {'full_validation':results,
              'up_to_600_seconds':[r for r in results if r['duration']<=600],
              'long_150_to_600':[r for r in results if 150<r['duration']<=600]}
    return {name:summarize(group) for name,group in groups.items()}


def parity_check(path, reference):
    """Check batched evaluation against the prior actual single-query backend."""
    baseline={r['qid']:r for r in read_jsonl(reference)}
    current=read_jsonl(path)
    assert len(current)==len(baseline)==4436
    max_span, max_score = 0., 0.
    for row in current:
        prior=baseline[row['qid']]
        # Compare in slot-independent score order; near ties may swap slots.
        old, new=prior['raw_candidates'], row['raw_candidates']
        assert len(old)==len(new)
        for p in new:
            q=min(old,key=lambda q:(max(abs(q[i]-p[i]) for i in range(2)), abs(q[2]-p[2])))
            max_span=max(max_span,max(abs(p[i]-q[i]) for i in range(2)))
            max_score=max(max_score,abs(p[2]-q[2]))
    counts={}
    for k in (1,5):
        for threshold in (.3,.5,.7):
            old=sum(r[f'sliding_iou{k}']>=threshold for r in baseline.values())
            new=sum(r[f'iou{k}']>=threshold for r in current)
            counts[f'R{k}@{threshold}']={'single_backend':old,'batched':new}
            assert old==new, f'Batched evaluation changes metric: {counts}'
    assert max_span<.03 and max_score<.001, (max_span,max_score)
    return {'queries':len(current),'max_candidate_boundary_difference_seconds':max_span,
            'max_candidate_score_difference':max_score,'hit_counts':counts}


def save_checkpoint(path, model, opt, epoch, metrics, optimizer=None, scheduler=None):
    checkpoint={'model':model.state_dict(),'opt':opt,'epoch':epoch,
                'finetune_validation_metrics':metrics,'training_dataset':'TACoS train'}
    if optimizer is not None:
        checkpoint.update(optimizer=optimizer.state_dict(),lr_scheduler=scheduler.state_dict(),
                          rng_python=random.getstate(),rng_numpy=np.random.get_state(),
                          rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all())
    temporary=path.with_suffix('.tmp')
    torch.save(checkpoint,temporary)
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--qd-root',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--train',type=Path,required=True)
    parser.add_argument('--validation',type=Path,required=True)
    parser.add_argument('--baseline-predictions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--patience',type=int,default=3)
    parser.add_argument('--query-batch-size',type=int,default=16)
    parser.add_argument('--lr',type=float,default=2e-5)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    sys.path.insert(0,str(args.qd_root))
    from qd_detr.model import build_model
    random.seed(2018); np.random.seed(2018); torch.manual_seed(2018); torch.cuda.manual_seed_all(2018)
    torch.set_num_threads(4)
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    train_rows, val_rows=read_jsonl(args.train),read_jsonl(args.validation)
    assert len(train_rows)==9790 and len(val_rows)==4436
    assert len({r['qid'] for r in train_rows})==9790
    assert not ({r['vid'] for r in train_rows}&{r['vid'] for r in val_rows})
    print('PREFLIGHT_LOADING_FEATURES',flush=True)
    train_features=Features(train_rows,args.features)
    val_features=Features(val_rows,args.features)
    dataset=WindowPairs(train_rows,train_features)
    base=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    opt=copy.deepcopy(base['opt'])
    opt.device=torch.device('cuda');opt.dset_name='tacos';opt.results_dir=str(output)
    opt.train_path=str(args.train);opt.eval_path=str(args.validation);opt.lr=args.lr
    opt.n_epoch=args.epochs;opt.lw_saliency=0.
    model,criterion=build_model(opt)
    model.load_state_dict(base['model'],strict=True);model.cuda()
    criterion.losses=['spans','labels']
    # TACoS has temporal bounds, but no human saliency ratings. Preserve the
    # saliency head without inventing human-rating labels; not auto-deployed.
    for name,parameter in model.named_parameters():
        if name.startswith(('saliency_proj1.','saliency_proj2.')):
            parameter.requires_grad_(False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=args.lr*.1)
    config={**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'seed':2018,'initial_checkpoint_sha256':sha256(args.checkpoint),
            'script_sha256':sha256(Path(__file__)),
            'train_annotation_sha256':sha256(args.train),'val_annotation_sha256':sha256(args.validation),
            'train_queries':9790,'train_videos':len(train_features.visual),
            'validation_queries':4436,'validation_videos':len(val_features.visual),
            'queries_with_disjoint_negative_window':sum(bool(n) for p,n in dataset.pools),
            'window_seconds':150,'overlap_seconds':30,'nms_threshold':.5,
            'positive_selection':'Random among >=50% target coverage windows; otherwise maximal coverage. Intersections clipped to local window; original 2s scale.',
            'negative_selection':'One randomly selected zero-overlap window from same video, if available, per query and epoch.',
            'losses':'Native Hungarian L1/GIoU/classification, including auxiliary decoder losses; no saliency-rating loss.',
            'batching':'Group by exact video-window clip count; no video padding, to match single-query deployment and avoid upstream mixed-length attention-mask effects.',
            'window_batch_size':args.query_batch_size*2,
            'selection':'Max full-validation R1@0.5; tie-break R5@0.5; keep earliest exact tie. Test data not read.',
            'backend_hashes':{p.name:sha256(p) for p in (PROJECT_ROOT/'backend').glob('*.py')},
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__}
    (output/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    print('CONFIG',json.dumps(config),flush=True)
    loader=DataLoader(dataset,batch_size=args.query_batch_size,shuffle=True,num_workers=0,collate_fn=collate_pairs)
    def step(samples):
        output_, ordered=forward_by_length(model,samples)
        if any(s['positive'] for s in samples):
            targets={'span_labels':[{'spans':s['span'].cuda()} for s in ordered]}
            losses=criterion(output_,targets)
        else:
            # Empty-target batches have only the native no-object CE loss;
            # do not average an empty regression tensor (which yields NaN).
            losses={}
            for suffix,out in [('',output_)]+[(f'_{i}',out) for i,out in enumerate(output_.get('aux_outputs',[]))]:
                logits=out['pred_logits']
                labels=torch.full(logits.shape[:2],criterion.background_label,device=logits.device,dtype=torch.long)
                losses['loss_label'+suffix]=F.cross_entropy(logits.transpose(1,2),labels,criterion.empty_weight,reduction='none').mean()
        loss=sum(v*criterion.weight_dict[k] for k,v in losses.items() if k in criterion.weight_dict)
        assert torch.isfinite(loss), {k:float(v.detach()) if torch.is_tensor(v) else v for k,v in losses.items()}
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True)
        optimizer.step()
        return float(loss.detach()),float(norm),len(samples),sum(s['positive'] for s in samples)
    if args.smoke:
        model.train()
        samples=next(iter(loader))
        before=model.class_embed.weight.detach().clone()
        for _ in range(3):
            print('SMOKE_STEP',step(samples),flush=True)
        negatives=[s for s in samples if not s['positive']]
        assert negatives
        print('SMOKE_ALL_NEGATIVE',step(negatives),flush=True)
        assert not torch.equal(before,model.class_embed.weight)
        save_checkpoint(output/'smoke.ckpt',model,opt,0,{},optimizer,scheduler)
        restored=torch.load(output/'smoke.ckpt',map_location='cpu',weights_only=False)
        model.load_state_dict(restored['model'],strict=True)
        print('SMOKE_OK',flush=True)
        return
    baseline=evaluate(model,val_rows,val_features,output/'validation_epoch_00.jsonl')
    parity=parity_check(output/'validation_epoch_00.jsonl',args.baseline_predictions)
    (output/'baseline_parity.json').write_text(json.dumps(parity,indent=2))
    print('BASELINE_PARITY',json.dumps(parity),flush=True)
    def score(metrics):
        m=metrics['full_validation']
        return m['R1@0.5_hits'],m['R5@0.5_hits']
    best_score,best_epoch,bad_epochs=score(baseline),0,0
    save_checkpoint(output/'model_best.ckpt',model,opt,0,baseline)
    history=[{'epoch':0,'validation':baseline,'selected':True}]
    (output/'history.json').write_text(json.dumps(history,indent=2))
    for epoch in range(1,args.epochs+1):
        started=time.monotonic();model.train();total_loss=0.;samples_seen=0;positive_seen=0
        lr=optimizer.param_groups[0]['lr']
        batches=epoch_window_batches(dataset,args.query_batch_size*2)
        for index,references in enumerate(batches,1):
            samples=[train_features.sample(train_rows[i],window,positive) for i,window,positive in references]
            assert len({len(s['video']) for s in samples})==1
            loss,norm,count,positive=step(samples)
            total_loss+=loss;samples_seen+=count;positive_seen+=positive
            if index%100==0 or index==len(batches):
                print('TRAIN',json.dumps({'epoch':epoch,'batch':index,'batches':len(batches),'mean_loss':total_loss/index,'gradient_norm':norm,'seconds':round(time.monotonic()-started,1)}),flush=True)
        scheduler.step()
        validation=evaluate(model,val_rows,val_features,output/f'validation_epoch_{epoch:02d}.jsonl')
        selected=score(validation)>best_score
        record={'epoch':epoch,'learning_rate':lr,'mean_loss':total_loss/len(batches),
                'samples':samples_seen,'positive_samples':positive_seen,'negative_samples':samples_seen-positive_seen,
                'validation':validation,'selected':selected,'seconds':round(time.monotonic()-started,2)}
        if selected:
            best_score,best_epoch,bad_epochs=score(validation),epoch,0
            save_checkpoint(output/'model_best.ckpt',model,opt,epoch,validation)
        else:
            bad_epochs+=1
        save_checkpoint(output/'model_latest.ckpt',model,opt,epoch,validation,optimizer,scheduler)
        history.append(record);(output/'history.json').write_text(json.dumps(history,indent=2))
        print('EPOCH_END',json.dumps(record),flush=True)
        if bad_epochs>=args.patience and epoch>=3:
            print('EARLY_STOP',epoch,flush=True);break
    result={'best_epoch':best_epoch,'best_validation_score_hits':best_score,
            'checkpoint':str(output/'model_best.ckpt'),'checkpoint_sha256':sha256(output/'model_best.ckpt'),
            'completed_epochs':history[-1]['epoch'],'baseline_validation':baseline,
            'best_validation':next(r['validation'] for r in history if r['epoch']==best_epoch),
            'test_evaluation':'not performed by training script','deployed':False}
    (output/'training_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('FINISHED',json.dumps(result),flush=True)


if __name__=='__main__':
    main()

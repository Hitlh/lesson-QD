"""Train a small head with same-video hard negatives, selecting on validation."""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from backend.window_reranker import WindowHead,FEATURE_DIM,FEATURE_VERSION,file_sha256
from ranking_metrics import Query,read,evaluate,diagnose

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'outputs/window_ranker')
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--extracted-dir',type=Path,default=PROJECT_ROOT/'outputs/window_features')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    random.seed(2018);np.random.seed(2018);torch.manual_seed(2018);torch.cuda.manual_seed_all(2018);torch.set_num_threads(4)
    train_rows=read(args.extracted_dir/'train/windows.jsonl');val_rows=read(args.extracted_dir/'val/windows.jsonl')
    assert len(train_rows)==9790 and len(val_rows)==4436
    assert not ({r['vid'] for r in train_rows}&{r['vid'] for r in val_rows})
    train_manifest=json.loads((args.extracted_dir/'train/manifest.json').read_text())
    val_manifest=json.loads((args.extracted_dir/'val/manifest.json').read_text())
    assert train_manifest['base_checkpoint_sha256']==val_manifest['base_checkpoint_sha256']
    x=torch.from_numpy(np.load(args.extracted_dir/'train/features.npy')).cuda()
    vx=torch.from_numpy(np.load(args.extracted_dir/'val/features.npy')).cuda()
    multi_indices=[w['feature_index'] for row in train_rows if len(row['windows'])>1 for w in row['windows']]
    mean=x[multi_indices].mean(0);scale=x[multi_indices].std(0,unbiased=False).clamp_min(.01)
    x=(x-mean)/scale;vx=(vx-mean)/scale
    queries=[Query(r) for r in val_rows];baseline=evaluate(queries)
    reference={r['qid']:r for r in read(args.baseline)}
    # Accept both the batched fine-tune evaluator and the single-query backend evaluator.
    for row in reference.values():
        for k in (1,5):
            if f'iou{k}' not in row:
                row[f'iou{k}'] = row[f'sliding_iou{k}']
    _,base_rows=evaluate(queries,return_rows=True)
    for row in base_rows:
        for k in (1,5):assert abs(row[f'iou{k}']-reference[row['qid']][f'iou{k}'])<.001
    for k in (1,5):
        for threshold in (.3,.5,.7):
            assert baseline['all'][f'R{k}@{threshold}_hits']==sum(r[f'iou{k}']>=threshold for r in reference.values())
    diagnostics=diagnose(queries)
    (args.output/'baseline_diagnostics.json').write_text(json.dumps({'baseline':baseline,'diagnostics':diagnostics},indent=2))
    print('BASELINE_DIAGNOSTICS',json.dumps(diagnostics),flush=True)
    pools=[]
    for row in train_rows:
        a,b=row['relevant_windows'][0]
        coverage=[max(0,min(b,w['end'])-max(a,w['start']))/(b-a) for w in row['windows']]
        maximum=max(coverage)
        positives=[w['feature_index'] for w,c in zip(row['windows'],coverage) if c>=.5 or c==maximum]
        negatives=[w['feature_index'] for w,c in zip(row['windows'],coverage) if c==0]
        # Initial hard negative is the disjoint window with highest base foreground score.
        def base_score(w):return max(p[2] for p in w['candidates'])
        hard=max((w for w,c in zip(row['windows'],coverage) if c==0),key=base_score)['feature_index'] if negatives else None
        pools.append((positives,negatives,hard))
    eligible=[i for i,(p,n,h) in enumerate(pools) if n]
    late=[train_rows[i]['relevant_windows'][0][0]>=150 for i in range(len(train_rows))]
    late_count=sum(late[i] for i in eligible)
    group_weights={True:len(eligible)/(2*late_count),False:len(eligible)/(2*(len(eligible)-late_count))}
    config={'epochs':20,'seed':2018,'lr':.001,'weight_decay':.001,'hidden':64,
            'strength_grid':[0,.125,.25,.5,1,2,4],'evaluate_epochs':[2,4,8,12,16,20],
            'feature_version':FEATURE_VERSION,'feature_dim':FEATURE_DIM,
            'base_checkpoint_sha256':train_manifest['base_checkpoint_sha256'],
            'train_queries':len(train_rows),'queries_with_hard_negative':sum(bool(n) for p,n,h in pools),
            'training_queries_used':len(eligible),'late_training_queries_used':late_count,
            'query_weighting':'Equal total loss weight for early and late target groups; only same-video positive/negative pairs are used. No global offset enters features.',
            'loss':'Per-query balanced BCE + 0.5 * pairwise softplus(margin 0.5); no backbone updates.',
            'negative_mining':'Highest-base-score zero-overlap window plus current-head hardest zero-overlap window, if distinct.',
            'selection':'Require non-decreasing validation full R1/R5 and late R1/R5; maximize mean(full R1,late R1), tie-break corresponding R5; lower strength and earlier epoch win exact ties. Zero-strength fallback included.',
            'train_script_sha256':file_sha256(Path(__file__)),'module_sha256':file_sha256(PROJECT_ROOT/'backend/window_reranker.py'),
            'test_read':False}
    (args.output/'training_config.json').write_text(json.dumps(config,indent=2))
    head=WindowHead(hidden=64).cuda();optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.001)
    def objective(m):
        return (m['all']['R1@0.5_hits']/m['all']['queries']+m['late']['R1@0.5_hits']/m['late']['queries'],
                m['all']['R5@0.5_hits']/m['all']['queries']+m['late']['R5@0.5_hits']/m['late']['queries'])
    def feasible(m):
        return all(m[g][key]>=baseline[g][key] for g,key in [('all','R1@0.5_hits'),('late','R1@0.5_hits'),('all','R5@0.5_hits'),('late','R5@0.5_hits')])
    best={'epoch':0,'strength':0.,'metrics':baseline};best_state=copy.deepcopy(head.state_dict());history=[];started=time.monotonic()
    for epoch in range(1,21):
        head.eval()
        with torch.no_grad():current=head(x).cpu().numpy()
        order=eligible[:];random.shuffle(order);loss_total=0.;steps=0
        head.train()
        for start in range(0,len(order),128):
            batch=order[start:start+128];indices=[];labels=[];query_weights=[];pairs=[]
            for qi in batch:
                positives,negatives,hard=pools[qi];positive=random.choice(positives)
                weight=group_weights[late[qi]]
                position=len(indices);indices.append(positive);labels.append(1.);query_weights.append(weight)
                if negatives:
                    chosen=list(dict.fromkeys([hard,max(negatives,key=lambda i:current[i])]))
                    for negative in chosen:
                        pairs.append((position,len(indices),weight/len(chosen)));indices.append(negative);labels.append(0.);query_weights.append(weight/len(chosen))
            logits=head(x[indices]);labels=torch.tensor(labels,device='cuda');weights=torch.tensor(query_weights,device='cuda')
            bce=(F.binary_cross_entropy_with_logits(logits,labels,reduction='none')*weights).sum()/len(batch)
            if pairs:
                pos,neg,pair_weights=zip(*pairs)
                ranking=(F.softplus(.5+logits[list(neg)]-logits[list(pos)])*torch.tensor(pair_weights,device='cuda')).sum()/len(batch)
            else:ranking=logits.sum()*0
            loss=bce+.5*ranking;assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);optimizer.step()
            loss_total+=float(loss.detach());steps+=1
        if epoch in config['evaluate_epochs']:
            head.eval()
            with torch.no_grad():logits=head(vx).cpu().numpy()
            attempts=[]
            for strength in config['strength_grid'][1:]:
                m=evaluate(queries,logits,strength);accepted=feasible(m)
                attempt={'strength':strength,'metrics':m,'feasible':accepted};attempts.append(attempt)
                if accepted and objective(m)>objective(best['metrics']):
                    best={'epoch':epoch,'strength':strength,'metrics':m};best_state=copy.deepcopy(head.state_dict())
            record={'epoch':epoch,'loss':loss_total/steps,'attempts':attempts,'best':best,'seconds':round(time.monotonic()-started,2)}
            history.append(record);(args.output/'history.json').write_text(json.dumps(history,indent=2))
            print('EVALUATION',json.dumps({'epoch':epoch,'best_epoch':best['epoch'],'strength':best['strength'],
                                         'full_R1':best['metrics']['all']['R1@0.5'],'late_R1':best['metrics']['late']['R1@0.5'],
                                         'full_R5':best['metrics']['all']['R5@0.5'],'seconds':round(time.monotonic()-started,1)}),flush=True)
    checkpoint={'feature_version':FEATURE_VERSION,'feature_dim':FEATURE_DIM,'hidden':64,
                'base_checkpoint_sha256':config['base_checkpoint_sha256'],'strength':best['strength'],
                'state_dict':{k:v.cpu() for k,v in best_state.items()},'mean':mean.cpu(),'scale':scale.cpu(),
                'selected_epoch':best['epoch']}
    torch.save(checkpoint,args.output/'window_ranker.pt')
    head.load_state_dict(best_state);head.eval()
    with torch.no_grad():logits=head(vx).cpu().numpy()
    final,values=evaluate(queries,logits,best['strength'],return_rows=True)
    assert final==best['metrics']
    with (args.output/'validation_predictions.jsonl').open('w') as f:
        for r in values:f.write(json.dumps(r)+'\n')
    np.save(args.output/'validation_window_logits.npy',logits)
    result={'baseline':baseline,'selected':best,'ranker_sha256':file_sha256(args.output/'window_ranker.pt'),
            'config':config,'backbone_unchanged':True,'test_read':False}
    (args.output/'selection.json').write_text(json.dumps(result,indent=2))
    print('FINISHED',json.dumps({'selected_epoch':best['epoch'],'strength':best['strength'],'metrics':final}),flush=True)


if __name__=='__main__':main()

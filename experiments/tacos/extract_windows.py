"""Frozen, unpadded window representations and original candidate predictions."""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    for name in ('qd-root','checkpoint','features','annotations','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--split',choices=['train','val','test'],required=True)
    args=p.parse_args();sys.path.insert(0,str(args.qd_root));torch.set_num_threads(4)
    from evaluate_tacos import load_model,read_jsonl,sha256
    from finetune import Features,inputs
    from backend.sliding_window import build_windows,globalize_candidates
    from backend.window_reranker import CaptureWindow,FEATURE_VERSION
    from qd_detr.span_utils import span_cxw_to_xx
    rows=read_jsonl(args.annotations)
    assert len({r['qid'] for r in rows})==len(rows)
    features=Features(rows,args.features);model=load_model(args.checkpoint)
    queues={};records={r['qid']:{**r,'windows':[]} for r in rows};vectors=[];started=time.monotonic()
    def flush(count):
        pending=queues[count]
        if not pending:return
        with torch.no_grad(),CaptureWindow(model) as capture:
            outputs=model(**inputs([s for r,s in pending]))
            embeddings=capture.features(outputs,count).cpu().numpy()
            spans=span_cxw_to_xx(outputs['pred_spans'].cpu())*(count*2)
            scores=outputs['pred_logits'].softmax(-1)[:,:,0].cpu()
            for i,(row,sample) in enumerate(pending):
                candidates=torch.cat([spans[i],scores[i,:,None]],-1).tolist()
                records[row['qid']]['windows'].append({
                    'clip_start':sample['start'],'clip_end':sample['end'],
                    'start':sample['start']*2.,'end':min(sample['end']*2.,row['duration']),
                    'feature_index':len(vectors),
                    'candidates':globalize_candidates(candidates,sample['start'],sample['end'],row['duration']),
                })
                vectors.append(embeddings[i])
        pending.clear()
    for index,row in enumerate(rows,1):
        for start,end in build_windows(len(features.visual[row['vid']])):
            count=end-start;visual=features.visual[row['vid']][start:end]
            t=torch.arange(count,dtype=torch.float32)/count
            sample={'video':torch.cat([visual,torch.stack([t,t+1/count],-1)],-1),
                    'text':features.text[row['qid']],'start':start,'end':end}
            pending=queues.setdefault(count,[]);pending.append((row,sample))
            if len(pending)==32:flush(count)
        if index%1000==0:print('PROGRESS',args.split,index,'/',len(rows),round(time.monotonic()-started,1),flush=True)
    for count in queues:flush(count)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'windows.jsonl').open('w') as f:
        for row in rows:
            record=records[row['qid']];record['windows'].sort(key=lambda w:w['clip_start'])
            f.write(json.dumps(record,ensure_ascii=False)+'\n')
    np.save(args.output/'features.npy',np.stack(vectors))
    manifest={'split':args.split,'queries':len(rows),'videos':len(features.visual),'windows':len(vectors),
              'base_checkpoint_sha256':sha256(args.checkpoint),'annotation_sha256':sha256(args.annotations),
              'feature_version':FEATURE_VERSION,'extraction_script_sha256':sha256(Path(__file__)),
              'ranker_module_sha256':sha256(PROJECT_ROOT/'backend/window_reranker.py'),
              'seconds':round(time.monotonic()-started,2),'model_frozen':True,
              'inputs':'Frozen checkpoint representations; no global window position and no ground-truth-dependent inputs.'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('FINISHED',json.dumps(manifest),flush=True)


if __name__=='__main__':main()

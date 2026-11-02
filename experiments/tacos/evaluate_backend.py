"""Exercise the real backend with a frozen checkpoint and cached input features."""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    for name in ('qd-root','checkpoint','features','annotations','output','pack'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--http',action='store_true')
    p.add_argument('--ranker',type=Path,required=True)
    args=p.parse_args();sys.path.insert(0,str(args.qd_root));torch.set_num_threads(4)
    os.environ.update(QD_DETR_ROOT=str(args.qd_root),QD_DETR_CHECKPOINT=str(args.checkpoint),
                      QD_DETR_FEATURE_MODE='slowfast_clip',QD_DETR_WINDOW_RANKER=str(args.ranker),
                      QD_DETR_NMS_METHOD='hard',QD_DETR_NMS_THRESHOLD='0.5',QD_DETR_NMS_MIN_SCORE='0')
    os.environ.pop('QD_DETR_WINDOW_RANKER_STRENGTH',None)
    from backend import app
    from backend.window_reranker import WindowRanker,file_sha256
    from evaluate_tacos import load_model,read_jsonl,best_iou
    from finetune import Features
    from ranking_metrics import Query,evaluate
    from fastapi.testclient import TestClient
    rows=read_jsonl(args.annotations);features=Features(rows,args.features)
    packed=read_jsonl(args.pack/'windows.jsonl');queries=[Query(r) for r in packed]
    manifest=json.loads((args.pack/'manifest.json').read_text())
    assert manifest['base_checkpoint_sha256']==file_sha256(args.checkpoint)
    ranker=WindowRanker(args.ranker,args.checkpoint,'cuda')
    with torch.no_grad():logits=ranker.score(torch.from_numpy(np.load(args.pack/'features.npy')).cuda()).cpu().numpy()
    cached_metrics,cached_values=evaluate(queries,logits,ranker.strength,return_rows=True)
    cached={r['qid']:r for r in cached_values}
    predictor=SimpleNamespace(device='cuda',model=load_model(args.checkpoint),
                              feature_extractor=SimpleNamespace(encode_text=lambda _:[current['text']]))
    current={};app._predictor=predictor
    app._get_video_features=lambda *_:(current['visual'],True)
    args.output.mkdir(parents=True,exist_ok=True);started=time.monotonic();records=[];max_difference=0.;max_score_difference=0.
    client=TestClient(app.app)
    with (args.output/'predictions.jsonl').open('w') as f:
        for index,row in enumerate(rows,1):
            current['visual']=features.visual[row['vid']]
            with np.load(args.features/'clip_text'/f'{row["qid"]}.npz',allow_pickle=False) as archive:
                current['text']=torch.from_numpy(archive['last_hidden_state'].astype(np.float32))
            record=app.VideoRecord(video_id=row['vid'],filename=row['vid']+'.mp4',path=Path(row['vid']+'.mp4'),duration=row['duration'])
            if args.http:
                if len(row['query'])>160:continue  # HTTP length guard belongs to the product; do not change it for a benchmark.
                app._videos[row['vid']]=record
                response=client.post('/api/search',json={'video_id':row['vid'],'query':row['query'],'top_k':5})
                assert response.status_code==200,response.text
                response=response.json()
            else:
                request=app.SearchRequest.model_construct(video_id=row['vid'],query=row['query'],top_k=5)
                response=app._run_model(record,request)
            predictions=[[m['start'],m['end'],m['score']] for m in response['moments']]
            result={k:row[k] for k in ('qid','vid','query','duration')}
            result.update(ground_truth=row['relevant_windows'],late=all(a>=150 for a,b in row['relevant_windows']),
                          predictions=predictions,iou1=best_iou(predictions[:1],row['relevant_windows']),
                          iou5=best_iou(predictions[:5],row['relevant_windows']),window_ranking=response['window_ranking'])
            for k in (1,5):max_difference=max(max_difference,abs(result[f'iou{k}']-cached[row['qid']][f'iou{k}']))
            for a,b in zip(predictions,cached[row['qid']]['predictions']):max_score_difference=max(max_score_difference,abs(a[2]-b[2]))
            assert all(0<=p[0]<p[1]<=response['duration'] for p in predictions)
            assert all(a[2]>=b[2] for a,b in zip(predictions,predictions[1:]))
            assert response['window_ranking']['enabled']==(response['windowing']['window_count']>1 and ranker.strength>0)
            records.append(result);f.write(json.dumps(result,ensure_ascii=False)+'\n')
            if index%500==0:print('PROGRESS',index,'/',len(rows),round(time.monotonic()-started,1),flush=True)
    from ranking_metrics import summarize
    assert not predictor.model.transformer._forward_hooks
    groups={'all':records,'late':[r for r in records if r['late']],
            'early':[r for r in records if not r['late']],
            'long_150_to_600':[r for r in records if 150<r['duration']<=600],
            'up_to_600':[r for r in records if r['duration']<=600]}
    result={'metrics':{name:summarize(group) for name,group in groups.items()},
            'queries':len(records),'videos':len({r['vid'] for r in records}),
            'max_iou_difference_from_cached':max_difference,'max_score_difference_from_cached':max_score_difference,
            'seconds':round(time.monotonic()-started,2),'http':args.http,
            'checkpoint_sha256':file_sha256(args.checkpoint),'ranker_sha256':file_sha256(args.ranker),
            'backend_hashes':{p.name:file_sha256(p) for p in (PROJECT_ROOT/'backend').glob('*.py')},
            'scope':'Precomputed-feature backend inference; offline long benchmark annotations bypass request text-length validation. HTTP smoke retains that validation.'}
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2))
    assert max_difference<.001,(max_difference,max_score_difference)
    assert max_score_difference<.001,max_score_difference
    if not args.http:
        assert result['metrics']==cached_metrics,'Actual backend changes retrieval metrics'
    print('FINISHED',json.dumps(result),flush=True)


if __name__=='__main__':main()

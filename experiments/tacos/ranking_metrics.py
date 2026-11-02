"""Cached exact hard-NMS decisions for validation sweeps and independent audit."""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import json
from pathlib import Path

import numpy as np


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def iou_matrix(a,b):
    overlap=np.maximum(0,np.minimum(a[:,None,1],b[None,:,1])-np.maximum(a[:,None,0],b[None,:,0]))
    union=(a[:,1]-a[:,0])[:,None]+(b[:,1]-b[:,0])[None,:]-overlap
    return np.divide(overlap,union,out=np.zeros_like(overlap),where=union>0)


class Query:
    def __init__(self,row):
        self.row=row
        triples=[];window_ids=[];features=[]
        for index,window in enumerate(row['windows']):
            for p in window['candidates']:
                if round(p[1],4)>round(p[0],4):
                    triples.append(p);window_ids.append(index);features.append(window['feature_index'])
        self.candidates=np.array(triples,dtype=np.float64).reshape(-1,3)
        self.window_ids=np.array(window_ids);self.features=np.array(features)
        self.base=self.candidates[:,2]
        clipped=np.clip(self.base,1e-7,1-1e-7)
        self.base_logit=np.log(clipped/(1-clipped))
        self.suppression=iou_matrix(self.candidates[:,:2],self.candidates[:,:2])>.5
        self.ious=iou_matrix(np.round(self.candidates[:,:2],4),np.array(row['relevant_windows'])).max(1)
        self.late=all(a>=150 for a,b in row['relevant_windows'])

    def score(self,logits,strength):
        if strength==0 or len(self.row['windows'])==1:
            return self.base
        values=(self.base_logit+strength*logits[self.features])/(1+strength)
        return 1/(1+np.exp(-np.clip(values,-700,700)))

    def top(self,logits=None,strength=0,window=None,all_candidates=False):
        scores=self.score(logits,strength)
        order=np.argsort(-scores,kind='stable')
        if window is not None:order=order[self.window_ids[order]==window]
        keep=[]
        while len(order):
            chosen=order[0];keep.append(int(chosen))
            if not all_candidates and len(keep)==5:break
            order=order[1:][~self.suppression[chosen,order[1:]]]
        return keep,scores


def summarize(values):
    result={'queries':len(values)}
    for k in (1,5):
        for threshold in (.3,.5,.7):
            count=sum(v[f'iou{k}']>=threshold for v in values)
            result[f'R{k}@{threshold}']=round(100*count/len(values),3) if values else None
            result[f'R{k}@{threshold}_hits']=count
    return result


def evaluate(queries,logits=None,strength=0,return_rows=False):
    values=[]
    for q in queries:
        keep,scores=q.top(logits,strength)
        values.append({'qid':q.row['qid'],'vid':q.row['vid'],'duration':q.row['duration'],'late':q.late,
                       'iou1':float(q.ious[keep[0]]) if keep else 0.,
                       'iou5':float(q.ious[keep].max()) if keep else 0.,
                       'predictions':[[round(float(q.candidates[i,0]),4),round(float(q.candidates[i,1]),4),round(float(scores[i]),4)] for i in keep] if return_rows else []})
    groups={'all':values,'late':[v for v in values if v['late']],
            'early':[v for v in values if not v['late']],
            'long_150_to_600':[v for v in values if 150<v['duration']<=600],
            'up_to_600':[v for v in values if v['duration']<=600]}
    metrics={name:summarize(rows) for name,rows in groups.items()}
    return (metrics,values) if return_rows else metrics


def diagnose(queries):
    rows=[]
    for q in queries:
        a,b=q.row['relevant_windows'][0]
        def key(i):
            w=q.row['windows'][i];s,e=w['start'],w['end']
            return max(0,min(b,e)-max(a,s))/(b-a),min(a-s,e-b),-i
        chosen=max(range(len(q.row['windows'])),key=key)
        local,_=q.top(window=chosen);global_,_=q.top()
        rows.append({'qid':q.row['qid'],'duration':q.row['duration'],'fully_contained':key(chosen)[0]>=1-1e-10,
                     'local_iou1':float(q.ious[local[0]]),'local_iou5':float(q.ious[local].max()),
                     'global_iou1':float(q.ious[global_[0]]),'global_iou5':float(q.ious[global_].max())})
    out={}
    for name,group in [('all',rows),('contained_long',[r for r in rows if 150<r['duration']<=600 and r['fully_contained']])]:
        out[name]={'queries':len(group)}
        for mode in ('local','global'):
            for k in (1,5):
                out[name][f'{mode}_R{k}@0.5']=round(100*sum(r[f'{mode}_iou{k}']>=.5 for r in group)/len(group),3)
        out[name]['local_top5_hit_to_global_miss']=sum(r['local_iou5']>=.5 and r['global_iou5']<.5 for r in group)
    return out

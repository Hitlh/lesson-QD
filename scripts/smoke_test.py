"""Real HTTP upload/search contract check using a video supplied by the recipient."""
import argparse
import itertools
import json
import urllib.request
import uuid
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('video', type=Path)
    p.add_argument('--query', default='Chef makes pizza and cuts it up.')
    p.add_argument('--base-url', default='http://127.0.0.1:8000')
    p.add_argument('--output', type=Path)
    args=p.parse_args();base=args.base_url.rstrip('/')
    health=json.load(urllib.request.urlopen(base+'/api/health',timeout=30))
    assert health['postprocessing']['method']=='hard' and health['postprocessing']['iou_threshold']==.5
    assert health['windowing']['window_seconds']==150 and health['windowing']['overlap_seconds']==30
    boundary=uuid.uuid4().hex
    payload=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{args.video.name}"\r\nContent-Type: application/octet-stream\r\n\r\n').encode()+args.video.read_bytes()+f'\r\n--{boundary}--\r\n'.encode()
    req=urllib.request.Request(base+'/api/videos',data=payload,headers={'Content-Type':'multipart/form-data; boundary='+boundary})
    video=json.load(urllib.request.urlopen(req,timeout=120))
    results=[]
    for query in [args.query, 'A person prepares food.']:
        body={'video_id':video['video_id'],'query':query,'top_k':5}
        req=urllib.request.Request(base+'/api/search',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        result=json.load(urllib.request.urlopen(req,timeout=1800))
        assert result['postprocessing']['method']=='hard'
        assert all(0<=m['start']<m['end']<=result['duration'] for m in result['moments'])
        assert len(result['saliency_scores'])==result['windowing']['feature_clip_count']
        if result['duration']>150.05:assert result['windowing']['window_count']>1
        for a,b in itertools.combinations(result['moments'],2):
            inter=max(0,min(a['end'],b['end'])-max(a['start'],b['start']))
            union=a['end']-a['start']+b['end']-b['start']-inter
            assert inter/union<=.5001
        results.append(result)
        print(json.dumps({k:result[k] for k in ['duration','moments','postprocessing','video_feature_cache_hit','inference_time_ms']},ensure_ascii=False),flush=True)
    assert results[1]['video_feature_cache_hit']
    req=urllib.request.Request(base+video['stream_url'],headers={'Range':'bytes=0-1023'})
    with urllib.request.urlopen(req,timeout=30) as stream:
        assert stream.status==206
        assert len(stream.read())==1024
    if args.output:args.output.write_text(json.dumps({'health':health,'video':video,'results':results,'range_stream_ok':True},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Upload, real model inference, NMS, windowing, cache and range streaming checks passed.')

if __name__=='__main__':main()

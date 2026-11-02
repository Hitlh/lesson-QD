import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from backend import app as api
from backend.window_reranker import (CaptureWindow, FEATURE_DIM, FEATURE_VERSION,
                                     WindowHead, WindowRanker, combine_scores, file_sha256)


class Transformer(torch.nn.Module):
    def forward(self, signal, count):
        local=torch.full((1,count,256),float(signal))
        global_=torch.full((1,256),float(signal))
        return None,None,local,global_


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.transformer=Transformer()

    def forward(self,**inputs):
        count=inputs['src_vid'].shape[1]
        start=float(inputs['src_vid'][0,0,0])
        self.transformer(start/60,count)
        # The upstream model also invokes the transformer for a negative pair.
        self.transformer(999,count)
        logit=torch.logit(torch.tensor(.9 if start==0 else .8))
        return {'pred_logits':torch.tensor([[[logit,0.],[-2.,0.]]]),
                'pred_spans':torch.tensor([[[.5,.1],[.1,.05]]]),
                'saliency_scores':torch.ones(1,count)}


def artifact(path,base,**changes):
    head=WindowHead()
    with torch.no_grad():
        for p in head.parameters():p.zero_()
        head.net[0].weight[0,0]=1
        head.net[3].weight[0,0]=4
        head.net[3].bias[0]=-2
    record={'feature_version':FEATURE_VERSION,'feature_dim':FEATURE_DIM,'hidden':64,
            'strength':1.,'base_checkpoint_sha256':file_sha256(base),
            'state_dict':head.state_dict(),'mean':torch.zeros(FEATURE_DIM),'scale':torch.ones(FEATURE_DIM)}
    record.update(changes);torch.save(record,path)


class WindowRankerTests(unittest.TestCase):
    def test_reranking_fixes_cross_window_order_without_changing_local_order(self):
        early=combine_scores([.9,.6],-2,1)
        late=combine_scores([.8,.5],2,1)
        self.assertGreater(late[0],early[0])
        self.assertGreater(early[0],early[1]);self.assertGreater(late[0],late[1])
        self.assertTrue(all(0<v<1 for v in early+late))
        self.assertEqual(combine_scores([.123456789,1.,0.],4.,0),[.123456789,1.,0.])
        for scores,value,strength in [([.9],float('nan'),1),([.9],0,-1),([1.1],0,1)]:
            with self.assertRaises(ValueError):combine_scores(scores,value,strength)

    def test_capture_uses_positive_pass_and_removes_hooks_on_errors(self):
        model=Model();kwargs={'src_vid':torch.zeros(1,75,3)}
        with CaptureWindow(model) as capture:
            output=model(**kwargs);features=capture.features(output,75)
            self.assertEqual(float(features[0,0]),0.)
            self.assertEqual(tuple(features.shape),(1,FEATURE_DIM))
        self.assertFalse(model.transformer._forward_hooks)
        with self.assertRaisesRegex(RuntimeError,'interrupted'):
            with CaptureWindow(model):raise RuntimeError('interrupted')
        self.assertFalse(model.transformer._forward_hooks)

    def test_loader_rejects_wrong_checkpoint_and_invalid_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base';base.write_bytes(b'base');wrong=root/'wrong';wrong.write_bytes(b'wrong')
            ranker=root/'ranker.pt';artifact(ranker,base)
            loaded=WindowRanker(ranker,base,'cpu');self.assertTrue(torch.isfinite(loaded.score(torch.zeros(1,FEATURE_DIM))).all())
            with self.assertRaisesRegex(ValueError,'不匹配'):WindowRanker(ranker,wrong,'cpu')
            artifact(ranker,base,scale=torch.zeros(FEATURE_DIM))
            with self.assertRaisesRegex(ValueError,'normalization'):WindowRanker(ranker,base,'cpu')

    def test_existing_http_payload_receives_reranked_global_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base';base.write_bytes(b'base');path=root/'head.pt';artifact(path,base)
            predictor=SimpleNamespace(device='cpu',model=Model(),feature_extractor=SimpleNamespace(encode_text=lambda _:[torch.ones(4,2)]))
            record=api.VideoRecord(video_id='ranking',filename='long.mp4',path=Path('/tmp/long.mp4'),duration=200)
            with patch.object(api,'WINDOW_RANKER_PATH',path),patch.object(api,'CHECKPOINT_PATH',base),patch.object(api,'WINDOW_RANKER_STRENGTH',None),patch.object(api,'_window_ranker',None),patch.object(api,'_get_predictor',return_value=predictor),patch.object(api,'_get_video_features',return_value=(torch.arange(100).float().unsqueeze(1),True)),patch.dict(api._videos,{record.video_id:record}):
                body=TestClient(api.app).post('/api/search',json={'video_id':record.video_id,'query':'event','top_k':5})
                self.assertEqual(body.status_code,200,body.text);body=body.json()
                self.assertTrue(body['window_ranking']['enabled'])
                self.assertEqual(body['windowing']['score_aggregation'],'window_reranked')
                self.assertGreater(body['moments'][0]['start'],150)
                self.assertTrue(all(0<=m['start']<m['end']<=200 for m in body['moments']))
                self.assertTrue(all(a['score']>=b['score'] for a,b in zip(body['moments'],body['moments'][1:])))
                self.assertFalse(predictor.model.transformer._forward_hooks)

    def test_single_window_bypasses_ranker(self):
        predictor=SimpleNamespace(device='cpu',model=Model(),feature_extractor=SimpleNamespace(encode_text=lambda _:[torch.ones(4,2)]))
        record=api.VideoRecord(video_id='short',filename='short.mp4',path=Path('/tmp/short.mp4'),duration=150)
        with patch.object(api,'_get_window_ranker',side_effect=AssertionError('short video changed')),patch.object(api,'_get_video_features',return_value=(torch.arange(75).float().unsqueeze(1),True)):
            prediction,_=api._predict_with_cached_features(predictor,record,'event')
        self.assertFalse(prediction['window_ranking']['enabled'])
        self.assertAlmostEqual(prediction['pred_relevant_windows'][0][2],.9,places=6)


if __name__=='__main__':unittest.main()

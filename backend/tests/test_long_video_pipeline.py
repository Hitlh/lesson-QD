import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from fastapi.testclient import TestClient

from backend import app as api
from backend.video_features import extract_video_features

sys.path.insert(0, str(api.QD_DETR_ROOT))


class RecordingModel:
    def __init__(self):
        self.calls = []

    def __call__(self, **inputs):
        self.calls.append(inputs)
        start = inputs['src_vid'][0, 0, 0].item()
        count = inputs['src_vid'].shape[1]
        return {
            'pred_logits': torch.tensor([[[start / 100, 0.0], [-1.0, 0.0]]]),
            'pred_spans': torch.tensor([[[.5, .5], [1., .6]]]),
            'saliency_scores': torch.full((1, count), start),
        }


class LongVideoPipelineTests(unittest.TestCase):
    def test_api_suppresses_duplicate_global_candidates_from_different_windows(self):
        def model(**inputs):
            start = inputs['src_vid'][0, 0, 0].item() * 2
            scale = inputs['src_vid'].shape[1] * 2
            other = (20, 25) if start == 0 else (145, 150)
            spans = [[(137.5 - start) / scale, 15 / scale],
                     [((other[0] + other[1]) / 2 - start) / scale, (other[1] - other[0]) / scale]]
            return {
                'pred_logits': torch.tensor([[[3. if start else 2., 0.], [-1., 0.]]]),
                'pred_spans': torch.tensor([spans]),
                'saliency_scores': torch.ones(1, inputs['src_vid'].shape[1]),
            }
        predictor = SimpleNamespace(device='cpu', model=model,
                                    feature_extractor=SimpleNamespace(encode_text=lambda _: [torch.ones(4, 2)]))
        record = api.VideoRecord(video_id='cross-window', filename='long.mp4', path=Path('/tmp/long.mp4'), duration=151)
        with patch.dict(api._videos, {record.video_id: record}), patch.object(api, '_get_predictor', return_value=predictor), patch.object(
            api, '_get_video_features', return_value=(torch.arange(76).float().unsqueeze(1), True),
        ):
            response = TestClient(api.app).post('/api/search', json={'video_id': record.video_id, 'query': 'walk', 'top_k': 3})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['windowing']['window_count'], 2)
        self.assertEqual(body['postprocessing']['candidates_before'], 4)
        self.assertEqual(body['postprocessing']['candidates_after'], 3)
        self.assertEqual((body['moments'][0]['start'], body['moments'][0]['end']), (130, 145))

    def test_real_inference_loop_reuses_query_and_resets_time_encoding(self):
        model = RecordingModel()
        encode_text = Mock(return_value=[torch.ones(4, 2)])
        predictor = SimpleNamespace(device='cpu', model=model, feature_extractor=SimpleNamespace(encode_text=encode_text))
        record = api.VideoRecord(video_id='long-loop', filename='long.mp4', path=Path('/tmp/long.mp4'), duration=599)
        features = torch.arange(300).float().unsqueeze(1)
        with patch.object(api, '_get_video_features', return_value=(features, True)):
            prediction, hit = api._predict_with_cached_features(predictor, record, 'a person walks')
        self.assertTrue(hit)
        self.assertEqual(encode_text.call_count, 1)
        self.assertEqual(len(model.calls), 5)
        self.assertEqual([c['src_vid'].shape[1] for c in model.calls], [75, 75, 75, 75, 60])
        for call in model.calls:
            video = call['src_vid'][0]
            self.assertEqual(video[0, -2].item(), 0)
            self.assertAlmostEqual(video[-1, -1].item(), 1, places=6)
            self.assertEqual(call['src_txt'].data_ptr(), model.calls[0]['src_txt'].data_ptr())
        rows = prediction['pred_relevant_windows']
        self.assertTrue(all(0 <= row[0] <= row[1] <= 599 for row in rows))
        self.assertGreater(rows[0][0], 150)
        self.assertEqual(len(rows), 10)
        self.assertEqual(len(prediction['pred_saliency_scores']), 300)
        self.assertEqual(prediction['pred_saliency_scores'][60], 30)
        self.assertEqual(prediction['pred_saliency_scores'][-1], 240)

    def test_block_extraction_has_no_duplicate_work_or_lost_partial_tail(self):
        image_encoder = Mock(side_effect=lambda frames: torch.ones(len(frames), 2))
        extractor = SimpleNamespace(
            video_loader=SimpleNamespace(_get_video_info=lambda _: {'height': 1, 'width': 1, 'fps': 30}),
            video_preprocessor=lambda frames: frames,
            clip_extractor=SimpleNamespace(encode_image=image_encoder),
            encode_video=Mock(side_effect=AssertionError('Long video must not use whole-video decoding')),
        )
        predictor = SimpleNamespace(device='cpu', feature_extractor=extractor)
        loader = SimpleNamespace(_preprocess=lambda frames, info: frames[::60])
        slow_encoder = Mock(side_effect=lambda frames, bsz: torch.ones(len(frames), 3))
        decoder = Mock(side_effect=lambda path, start, duration, info, fps, frames: torch.ones(frames, 1, 1, 3, dtype=torch.uint8))
        with patch('backend.video_features._decode_block', decoder):
            features = extract_video_features(predictor, Path('/tmp/video.mp4'), 151,
                                              slowfast_components=(loader, slow_encoder), chunk_clips=30)
        self.assertEqual(features.shape, (76, 5))
        self.assertEqual(features.device.type, 'cpu')
        self.assertEqual([call.args[1] for call in decoder.call_args_list], [0, 0, 60, 60, 120, 120])
        self.assertEqual([call.args[2] for call in decoder.call_args_list], [60, 60, 60, 60, 31, 31])
        self.assertEqual([call.kwargs['frames'] for call in decoder.call_args_list], [30, 1800, 30, 1800, 16, 960])

    def test_short_extraction_retains_original_path_and_modality_trim(self):
        encode_video = Mock(return_value=torch.ones(75, 2))
        predictor = SimpleNamespace(feature_extractor=SimpleNamespace(encode_video=encode_video))
        loader = Mock(return_value=torch.ones(76, 1))
        encoder = Mock(return_value=torch.ones(76, 3))
        with patch('backend.video_features._decode_block', side_effect=AssertionError('Short video changed decoding path')):
            features = extract_video_features(predictor, Path('/tmp/short.mp4'), 150.0165,
                                              slowfast_components=(loader, encoder))
        self.assertEqual(features.shape, (75, 5))
        self.assertTrue(torch.allclose(features[0], torch.tensor([3 ** -.5] * 3 + [2 ** -.5] * 2)))
        encode_video.assert_called_once()

    def test_feature_cache_reuses_appearance_and_does_not_save_failed_extraction(self):
        predictor = SimpleNamespace(model=SimpleNamespace(input_vid_proj=[SimpleNamespace(net=[None, SimpleNamespace(in_features=7)])]))
        record = api.VideoRecord(video_id='cache-long', filename='long.mp4', path=Path('/tmp/long.mp4'), duration=599)
        with patch.dict(api._video_feature_cache, clear=True), patch.object(api, 'FEATURE_MODE', 'clip'), patch.object(
            api, 'extract_video_features', return_value=torch.ones(300, 5),
        ) as extract:
            first, hit1 = api._get_video_features(predictor, record)
            second, hit2 = api._get_video_features(predictor, record)
            self.assertFalse(hit1)
            self.assertTrue(hit2)
            self.assertIs(first, second)
            self.assertEqual(extract.call_count, 1)
        with patch.dict(api._video_feature_cache, clear=True), patch.object(api, 'FEATURE_MODE', 'clip'), patch.object(
            api, 'extract_video_features', side_effect=RuntimeError('interrupted'),
        ):
            with self.assertRaises(RuntimeError):
                api._get_video_features(predictor, record)
            self.assertNotIn(record.video_id, api._video_feature_cache)

    def test_upload_accepts_long_video_but_preserves_service_cap(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(api, 'STORAGE_DIR', Path(folder)), patch.dict(api._videos):
            client = TestClient(api.app)
            for duration, status in ((151, 201), (600.0165, 201), (601, 422)):
                with self.subTest(duration=duration), patch.object(api, '_probe_duration', return_value=duration):
                    response = client.post('/api/videos', files={'file': ('long.mp4', b'video', 'video/mp4')})
                    self.assertEqual(response.status_code, status)
            self.assertEqual(len(list(Path(folder).iterdir())), 2)


if __name__ == '__main__':
    unittest.main()

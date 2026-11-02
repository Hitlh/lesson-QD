"""HTTP contract tests with fixed predictions; real inference is smoke-tested separately."""
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import app as api


class SearchApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)
        self.record = api.VideoRecord(video_id='dedup-test', filename='test.mp4', path=Path('/tmp/test.mp4'), duration=100)
        self.video_patch = patch.dict(api._videos, {self.record.video_id: self.record})
        self.predictor_patch = patch.object(api, '_get_predictor', return_value=object())
        self.prediction_patch = patch.object(api, '_predict_with_cached_features', return_value=({
            'pred_relevant_windows': [[10, 20, .9], [10, 20, .89], [40, 50, .7], [60, 70, .6]],
            'pred_saliency_scores': [0.1, 0.5],
        }, True))
        for mock in (self.video_patch, self.predictor_patch, self.prediction_patch):
            mock.start()
            self.addCleanup(mock.stop)

    def search(self, **options):
        return self.client.post('/api/search', json={
            'video_id': self.record.video_id, 'query': 'a person walks', 'top_k': 3, **options,
        })

    def test_existing_frontend_payload_uses_server_defaults(self):
        response = self.search()
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['postprocessing']['method'], 'hard')
        self.assertEqual(body['postprocessing']['iou_threshold'], .5)
        self.assertEqual([m['start'] for m in body['moments']], [10, 40, 60])
        self.assertEqual(body['saliency_scores'], [.1, .5])
        self.assertTrue(body['video_feature_cache_hit'])

    def test_all_frontend_modes_and_optional_overrides(self):
        for mode in ('none', 'hard', 'soft_linear'):
            with self.subTest(mode=mode):
                response = self.search(nms_method=mode, nms_threshold=.5)
                self.assertEqual(response.status_code, 200)
                body = response.json()
                self.assertEqual(body['postprocessing']['method'], mode)
                self.assertEqual(body['postprocessing']['returned_count'], len(body['moments']))
                self.assertTrue(all(0 <= m['start'] < m['end'] <= body['duration'] for m in body['moments']))
                self.assertTrue(all('raw_score' in m for m in body['moments']))
                self.assertEqual(len(body['moments']), 3)
                self.assertEqual(body['moments'][1]['start'], 10 if mode == 'none' else 40)

    def test_invalid_options_return_validation_errors(self):
        for options in ({'nms_method': 'invalid'}, {'nms_threshold': -1},
                        {'nms_threshold': 1.01}, {'nms_min_score': -1}, {'top_k': 6}):
            with self.subTest(options=options):
                self.assertEqual(self.search(**options).status_code, 422)

    def test_fewer_results_and_empty_results_keep_array_contract(self):
        for rows, expected_count in (([[10, 20, .9], [10, 20, .8]], 1), ([], 0)):
            with self.subTest(expected_count=expected_count), patch.object(
                api, '_predict_with_cached_features', return_value=({
                    'pred_relevant_windows': rows, 'pred_saliency_scores': [],
                }, True),
            ):
                response = self.search(top_k=5)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(len(response.json()['moments']), expected_count)
                self.assertEqual(response.json()['postprocessing']['returned_count'], expected_count)

    def test_missing_video_and_health_defaults(self):
        self.assertEqual(self.search(video_id='does-not-exist').status_code, 404)
        self.assertEqual(self.client.get('/api/health').json()['postprocessing'], {
            'method': 'hard', 'iou_threshold': .5, 'min_score': 0.0,
        })


if __name__ == '__main__':
    unittest.main()

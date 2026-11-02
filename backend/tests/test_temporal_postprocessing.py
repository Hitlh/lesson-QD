import copy
import math
import unittest

from backend.temporal_postprocessing import postprocess_moments, temporal_iou


class TemporalPostprocessingTests(unittest.TestCase):
    def test_hard_suppresses_before_top_k(self):
        rows = [[10, 20, .9], [10, 20, .89], [10.5, 20, .88], [40, 50, .7], [60, 70, .6]]
        moments, info = postprocess_moments(rows, 100, top_k=3)
        self.assertEqual([m['start'] for m in moments], [10, 40, 60])
        self.assertEqual(info['candidates_before'], 5)
        self.assertEqual(info['candidates_after'], 3)

    def test_soft_reselects_after_decay_and_preserves_original_score(self):
        rows = [[0, 10, .9], [0, 9, .85], [20, 30, .7]]
        original = copy.deepcopy(rows)
        moments, _ = postprocess_moments(rows, 40, method='soft_linear', iou_threshold=.5)
        self.assertEqual([m['start'] for m in moments], [0, 20, 0])
        self.assertEqual(moments[2]['score'], .085)
        self.assertEqual(moments[2]['raw_score'], .85)
        self.assertEqual(rows, original)

    def test_soft_discards_zero_and_below_minimum_scores(self):
        moments, info = postprocess_moments(
            [[0, 10, .9], [0, 10, .8], [0, 9, .7]], 20,
            method='soft_linear', iou_threshold=.5, min_score=.1,
        )
        self.assertEqual(len(moments), 1)
        self.assertEqual(info['candidates_after'], 1)

    def test_cleanup_precedes_suppression_and_selection(self):
        rows = [[1, 2, math.nan], [10, 9, 1], [5, 5, 1], [math.inf, 5, 1],
                [90, 120, .9], [90, 100, .8], [-5, 5, .7], [30, 40, .6]]
        moments, info = postprocess_moments(rows, 100)
        self.assertEqual([(m['start'], m['end']) for m in moments], [(90, 100), (0, 5), (30, 40)])
        self.assertEqual(info['candidates_valid'], 4)
        self.assertEqual(info['candidates_after'], 3)

    def test_rounding_does_not_emit_empty_intervals(self):
        moments, _ = postprocess_moments([[1, 1.000001, .99], [2, 3, .9]], 10, top_k=1)
        self.assertEqual(moments[0]['start'], 2)

    def test_threshold_is_strict_and_nonoverlap_is_retained(self):
        rows = [[0, 3, .9], [1, 4, .8], [4, 5, .7]]
        self.assertEqual(temporal_iou(rows[0], rows[1]), .5)
        self.assertEqual(temporal_iou(rows[1], rows[2]), 0)
        self.assertEqual(len(postprocess_moments(rows, 10, iou_threshold=.5)[0]), 3)
        self.assertEqual(len(postprocess_moments(rows, 10, iou_threshold=.49)[0]), 2)

    def test_none_keeps_duplicates_and_stable_ties(self):
        rows = [[10, 20, .8], [30, 40, .9], [30, 40, .9]]
        original = copy.deepcopy(rows)
        moments, info = postprocess_moments(rows, 100, method='none')
        self.assertEqual([m['start'] for m in moments], [30, 30, 10])
        self.assertEqual(info['candidates_after'], 3)
        self.assertEqual(rows, original)

    def test_empty_and_single_candidate(self):
        for mode in ('none', 'hard', 'soft_linear'):
            self.assertEqual(postprocess_moments([], 100, method=mode)[0], [])
            self.assertEqual(len(postprocess_moments([[1, 2, .9]], 100, method=mode)[0]), 1)

    def test_invalid_configuration_is_rejected(self):
        for options in ({'method': 'invalid'}, {'iou_threshold': 1.1},
                        {'iou_threshold': math.nan}, {'min_score': -1}, {'top_k': 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                postprocess_moments([], 100, **options)


if __name__ == '__main__':
    unittest.main()

import math
import unittest

from backend.sliding_window import build_windows, clip_count, globalize_candidates, merge_saliency


class SlidingWindowTests(unittest.TestCase):
    def test_boundaries_cover_every_clip_without_exceeding_capacity(self):
        for count in (1, 74, 75, 76, 135, 136, 300):
            with self.subTest(count=count):
                windows = build_windows(count)
                self.assertEqual(windows[-1][1], count)
                self.assertEqual({i for start, end in windows for i in range(start, end)}, set(range(count)))
                self.assertTrue(all(0 < end - start <= 75 for start, end in windows))
                self.assertTrue(all(a[1] - b[0] == 15 for a, b in zip(windows, windows[1:])))
        self.assertEqual(build_windows(300), [(0, 75), (60, 135), (120, 195), (180, 255), (240, 300)])
        self.assertEqual(build_windows(75), [(0, 75)])

    def test_partial_clip_and_container_tolerance(self):
        self.assertEqual(clip_count(150.0165), 75)
        self.assertEqual(clip_count(151), 76)
        self.assertEqual(clip_count(.2), 1)
        self.assertEqual(clip_count(599.5), 300)
        for duration in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                clip_count(duration)

    def test_local_clipping_precedes_global_offset(self):
        self.assertEqual(globalize_candidates([[-10, 40, .9]], 60, 76, 151), [[120, 151, .9]])
        self.assertEqual(globalize_candidates([[10, 25, .8]], 60, 135, 300), [[130, 145, .8]])

    def test_saliency_averages_overlap_and_keeps_tail(self):
        self.assertEqual(merge_saliency(5, [(0, [1, 2, 3]), (2, [5, 6, 7])]), [1, 2, 4, 6, 7])
        with self.assertRaises(ValueError):
            merge_saliency(5, [(0, [1, 2])])
        with self.assertRaises(ValueError):
            merge_saliency(1, [(0, [math.nan])])

    def test_invalid_window_configuration(self):
        for args in ((0, 75, 15), (100, 76, 15), (100, 75, 75), (100, 75, -1)):
            with self.assertRaises(ValueError):
                build_windows(*args)

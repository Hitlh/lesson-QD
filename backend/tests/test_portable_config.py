import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

from backend.config import PROJECT_ROOT, configured_path
from start_backend import configure


class PortableConfigTests(unittest.TestCase):
    def test_upstream_training_and_evaluation_help_can_be_displayed(self):
        sys.path.insert(0, str(PROJECT_ROOT / 'QD-DETR'))
        from qd_detr.config import BaseOptions, TestOptions
        for kind in (BaseOptions, TestOptions):
            options = kind()
            options.initialize()
            self.assertIn('--v_feat_dirs', options.parser.format_help())

    def test_relative_paths_survive_launch_from_other_directory(self):
        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'QD_DETR_CHECKPOINT': 'models/custom.ckpt'}):
            try:
                os.chdir(folder)
                self.assertEqual(configured_path('QD_DETR_CHECKPOINT', 'unused'), PROJECT_ROOT / 'models/custom.ckpt')
            finally:
                os.chdir(cwd)

    def test_environment_override_and_profile_defaults(self):
        args = argparse.Namespace(profile='tacos', device=None, slowfast_batch_size=None, cors_origin=None)
        with patch.dict(os.environ, {'QD_DETR_CHECKPOINT': 'models/custom.ckpt'}, clear=True):
            configure(args)
            self.assertEqual(os.environ['QD_DETR_CHECKPOINT'], str(PROJECT_ROOT / 'models/custom.ckpt'))
            self.assertEqual(os.environ['QD_DETR_WINDOW_RANKER'], str(PROJECT_ROOT / 'models/window_ranker.pt'))
            self.assertEqual(os.environ['FRONTEND_DIST'], str(PROJECT_ROOT / 'frontend/dist'))

    def test_explicit_options_override_environment_and_empty_ranker_disables_it(self):
        args = argparse.Namespace(profile='tacos', device='cpu', slowfast_batch_size=2, cors_origin=['http://localhost:8080'])
        with patch.dict(os.environ, {'QD_DETR_DEVICE': 'cuda', 'QD_DETR_WINDOW_RANKER': ''}, clear=True):
            configure(args)
            self.assertEqual(os.environ['QD_DETR_DEVICE'], 'cpu')
            self.assertEqual(os.environ['SLOWFAST_BATCH_SIZE'], '2')
            self.assertEqual(os.environ['QD_DETR_WINDOW_RANKER'], '')

    def test_evaluation_relocates_saved_paths_and_honors_explicit_features(self):
        sys.path.insert(0, str(PROJECT_ROOT / 'QD-DETR'))
        from qd_detr.config import TestOptions
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'opt.json').write_text(json.dumps({
                'v_feat_dirs': ['previous_features'], 't_feat_dir': 'previous_text',
                'v_feat_dim': 2816, 't_feat_dim': 512, 'ctx_mode': 'video_tef',
                'results_dir': 'previous_results', 'eval_path': 'previous_annotations',
                'eval_split_name': 'test',
            }))
            argv = ['evaluate', '--resume', str(root / 'model_best.ckpt'),
                    '--v_feat_dirs', 'new_slowfast', 'new_clip', '--t_feat_dir', 'new_text',
                    '--eval_path', 'new_annotations', '--eval_split_name', 'val']
            with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
                options = TestOptions().parse()
            self.assertEqual(options.v_feat_dirs, ['new_slowfast', 'new_clip'])
            self.assertEqual(options.t_feat_dir, 'new_text')
            self.assertEqual(options.eval_path, 'new_annotations')
            self.assertEqual(options.eval_split_name, 'val')
            self.assertEqual(options.results_dir, str(root))

"""Configuration and no-allocation training entry checks."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vpp2_obj.config import PROJECT, check_data_sources, load_config
from vpp2_obj.train import main


class ObjectVideoConfigurationTests(unittest.TestCase):
    def test_rejects_wrong_objective_and_sampling(self):
        for override in ("model.condition_latent_loss_weight=0", "data.train.history_stride=25",
                         "data.train.prediction_stride=4", "data.train.history_frames=7",
                         "data.train.training=false", "data.test.balance_tasks=true"):
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_config(overrides=[override])

    def test_frozen_partition_and_cache_references(self):
        summary = check_data_sources(load_config())
        self.assertEqual((summary['train_episodes'], summary['test_episodes'], summary['excluded_episodes']),
                         (2155, 240, 105))
        self.assertGreater(summary['complete_future_windows']['train'], 0)

    def test_rejects_training_from_test(self):
        # Equal paths are rejected during config validation, before model loading.
        with self.assertRaises(ValueError):
            load_config(overrides=['data.train.manifest_path=${paths.test_manifest}'])

    def test_smoke_inherits_model_but_reduces_run_budget(self):
        cfg = load_config('configs/obj_video_smoke_v1.yaml')
        self.assertEqual(cfg.max_steps, 2)
        self.assertEqual(cfg.gradient_accumulation_steps, 1)
        self.assertEqual(cfg.model.condition_latent_loss_weight, 1)
        self.assertFalse(cfg.save_final_checkpoint)
        self.assertEqual(cfg.data.train.samples_per_epoch, 64)

    def test_dry_run_does_not_create_output_or_call_runtime(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {'WORLD_SIZE': '4'}):
            output = Path(temporary) / 'never-created'
            stdout = io.StringIO()
            with patch('vpp2_obj.train.run_training') as runtime, contextlib.redirect_stdout(stdout):
                self.assertEqual(main(['train', '--dry-run', f'output_dir={output}']), 0)
            runtime.assert_not_called()
            self.assertFalse(output.exists())
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary['global_batch'], 832)
            self.assertEqual(summary['condition_latent_loss_weight'], 1)

    def test_project_relative_config_works_outside_project(self):
        previous = Path.cwd()
        try:
            os.chdir('/tmp')
            cfg = load_config('configs/obj_video_baseline_v1.yaml')
            self.assertTrue(Path(cfg.data.train.manifest_path).is_absolute())
            self.assertEqual(Path(cfg.paths.project), PROJECT)
        finally:
            os.chdir(previous)

    def test_rejects_partial_accumulation_at_epoch_boundary(self):
        with patch.dict(os.environ, {'WORLD_SIZE': '4'}), self.assertRaisesRegex(ValueError, 'divisible'):
            main(['train', '--dry-run', 'data.train.samples_per_epoch=703130'])

    def test_resume_paths_are_checked_before_dry_run_returns(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {'WORLD_SIZE': '4'}):
            for name in ('resume', 'resume_ckpt'):
                with self.subTest(name=name), self.assertRaises(FileNotFoundError):
                    main(['train', '--dry-run', f'{name}={temporary}/missing'])


if __name__ == '__main__':
    unittest.main()

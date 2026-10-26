"""CPU-only inference contract checks; no pretrained models are loaded."""

from contextlib import ExitStack, redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from omegaconf import OmegaConf

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))

from vpp2_obj import infer


class InferenceCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / 'saved_dit.pth'
        self.checkpoint.write_bytes(b'checkpoint validation fixture')
        self.config = self.root / 'config.yaml'
        self.config.write_text('eval_num_inference_steps: 10\n')
        self.manifest = self.root / 'test.jsonl'
        self.manifest.write_text('{}\n')
        self.output = self.root / 'generated'
        self.cfg = OmegaConf.create({
            'mixed_precision': 'bf16', 'eval_num_inference_steps': 10,
            'data': {'test': {'_target_': 'vpp2_obj.dataset.PairedObjectVideoDataset',
                              'manifest_path': str(self.manifest), 'training': False, 'balance_tasks': False}},
            'model': {'_target_': 'vpp2_obj.model.create_model', 'dit_checkpoint_path': '/original/stage2.pth',
                      'video_dit_config': {'text_len': 128}, 'condition_latent_loss_weight': 1.0},
        })
        # Target paths intentionally do not exist, including the first-frame file.
        self.row = {'task': 'move_object', 'episode': 'episode7', 'condition_video_path': '/rgb/input.mp4',
                    'target_video_path': '/DO_NOT_READ/object.mp4', 'target_first_frame_path': '/DO_NOT_READ/frame.png',
                    'prompt': 'Move the object.', 'text_embedding_path': '/cache/prompt.pt',
                    'text_embedding_sha256': 'cache-file-sha', 'text_embedding_cache_signature': 'cache-contract-sha'}
        self.indices = {'task': 'move_object', 'episode': 'episode7', 'episode_index': 0, 'anchor_index': 98,
                        'deduplicated_length': 99, 'condition_indices': torch.tensor([0, 0, 0, 0, 0, 8, 38, 68, 98]),
                        'condition_source_indices': torch.tensor([0, 0, 0, 0, 0, 16, 76, 136, 196]),
                        'condition_image_is_pad': torch.tensor([False, True, True, True, True, False, False, False, False]),
                        'future_indices': torch.arange(103, 179, 5)}
        self.sample = {**self.indices, 'condition_video': torch.zeros(3, 9, 16, 16),
                       'context': torch.ones(128, 4096, dtype=torch.bfloat16),
                       'context_mask': torch.arange(128) < 57, 'prompt': self.row['prompt']}
        self.dataset = Mock(records=[self.row])
        self.dataset.window_indices.return_value = self.indices
        self.dataset.get_window.return_value = self.sample
        self.frames = [object() for _ in range(16)]
        self.model = Mock()
        self.model.infer_object_future.return_value = {'video': self.frames}

    def args(self, *extra):
        return infer.build_parser().parse_args([
            '--config', str(self.config), '--checkpoint', str(self.checkpoint),
            '--task', 'move_object', '--episode', 'episode7', '--anchor', '98',
            '--output', str(self.output), '--device', 'cpu', *extra,
        ])

    def dependencies(self):
        stack = ExitStack()
        stack.enter_context(patch.object(infer, 'load_config', return_value=self.cfg))
        dataset_factory = stack.enter_context(patch.object(infer, 'make_dataset', return_value=self.dataset))
        model_factory = stack.enter_context(patch.object(infer, 'make_model', return_value=self.model))
        writer = stack.enter_context(patch.object(infer, 'write_video',
                                                  side_effect=lambda frames, path, fps: path.write_bytes(b'mp4 fixture')))
        return stack, dataset_factory, model_factory, writer

    def test_dry_run_checks_indices_without_decoding_model_or_any_output(self):
        stack, dataset_factory, model_factory, writer = self.dependencies()
        with stack:
            result = infer.run_inference(self.args('--dry-run'))
        self.assertEqual(result['status'], 'dry_run_passed')
        self.assertEqual(result['num_inference_steps'], 10)
        self.assertEqual(result['sampling']['anchor_index'], 98)
        self.dataset.window_indices.assert_called_once_with(0, 98, include_target=False)
        self.dataset.get_window.assert_not_called()
        model_factory.assert_not_called()
        writer.assert_not_called()
        self.assertFalse(self.output.exists())
        dataset_factory.assert_called_once_with(self.cfg, self.manifest)

    def test_inference_uses_rgb_and_real_text_mask_without_target_dependency(self):
        stack, _, model_factory, writer = self.dependencies()
        with stack:
            result = infer.run_inference(self.args('--seed', '23', '--steps', '7'))
        self.dataset.get_window.assert_called_once_with(0, 98, include_target=False)
        model_factory.assert_called_once_with(self.cfg, self.checkpoint, 'cpu')
        self.model.eval.assert_called_once_with()
        call = self.model.infer_object_future.call_args.kwargs
        self.assertEqual(set(call), {'condition_video', 'context', 'context_mask', 'num_inference_steps', 'seed'})
        self.assertIs(call['condition_video'], self.sample['condition_video'])
        self.assertIs(call['context_mask'], self.sample['context_mask'])
        self.assertIs(call['context'], self.sample['context'])
        self.assertEqual(call['num_inference_steps'], 7)
        self.assertEqual(call['seed'], 23)
        self.assertIs(writer.call_args.args[0], self.frames)
        self.assertEqual(writer.call_args.args[2], 6)
        self.assertEqual(result['exported_frames'], 16)
        self.assertEqual(result['decoded_frame_slice'], [9, 25])
        self.assertFalse(result['target_video_read'])
        self.assertFalse(result['target_first_frame_read'])
        self.assertEqual(result['context_mask'], self.sample['context_mask'].tolist())
        self.assertEqual(result['model_visible_text_tokens'], 57)
        self.assertEqual(json.loads((self.output / 'trace.json').read_text())['status'], 'complete')
        self.assertTrue((self.output / 'future.mp4').is_file())
        self.assertFalse((self.output / '.future.tmp.mp4').exists())

    def test_model_returning_prefix_frames_is_rejected_and_failure_is_recorded(self):
        self.model.infer_object_future.return_value = {'video': [object() for _ in range(25)]}
        stack, _, _, writer = self.dependencies()
        with stack, self.assertRaisesRegex(ValueError, 'exactly sixteen'):
            infer.run_inference(self.args())
        writer.assert_not_called()
        self.assertFalse((self.output / 'future.mp4').exists())
        failure = json.loads((self.output / 'failure.json').read_text())
        self.assertEqual(failure['status'], 'failed')

    def test_existing_output_and_dangling_symlink_are_not_overwritten(self):
        for symlink in (False, True):
            with self.subTest(symlink=symlink):
                if symlink:
                    self.output.symlink_to(self.root / 'missing-directory', target_is_directory=True)
                else:
                    self.output.mkdir()
                    (self.output / 'keep.txt').write_text('keep')
                with patch.object(infer, 'load_config') as load, self.assertRaises(FileExistsError):
                    infer.run_inference(self.args('--dry-run'))
                load.assert_not_called()
                if symlink:
                    self.assertTrue(self.output.is_symlink())
                    self.output.unlink()
                else:
                    self.assertEqual((self.output / 'keep.txt').read_text(), 'keep')
                    (self.output / 'keep.txt').unlink()
                    self.output.rmdir()

    def test_invalid_cli_values_fail_before_loading(self):
        for flag, value in [('--anchor', '-1'), ('--steps', '0'), ('--steps', '1001'),
                            ('--seed', '-2'), ('--fps', '0'), ('--device', 'cuda:-1'),
                            ('--device', 'gpu4')]:
            with self.subTest(flag=flag, value=value), patch.object(infer, 'load_config') as load:
                with self.assertRaises(ValueError):
                    infer.run_inference(self.args(flag, value))
                load.assert_not_called()

    def test_missing_checkpoint_and_nonunique_episode_fail_without_model(self):
        with self.assertRaises(FileNotFoundError):
            infer.run_inference(self.args('--checkpoint', str(self.root / 'missing.pth')))
        for records in ([], [self.row, self.row]):
            with self.subTest(count=len(records)), self.assertRaisesRegex(ValueError, 'exactly one'):
                infer.select_episode(records, 'move_object', 'episode7')

    def test_corrupt_mask_and_nonfinite_context_are_rejected_before_model_load(self):
        for key, bad_value in [('context_mask', torch.ones(128, dtype=torch.int64)),
                               ('context', torch.full((128, 4096), float('nan')))]:
            with self.subTest(key=key):
                sample = {**self.sample, key: bad_value}
                with self.assertRaises(ValueError):
                    infer.validate_condition_sample(sample)

    def test_checkpoint_override_reuses_model_configuration_and_dtype(self):
        with patch('hydra.utils.instantiate', return_value=self.model) as factory:
            result = infer.make_model(self.cfg, self.checkpoint, 'cuda:0')
        options = factory.call_args.args[0]
        self.assertEqual(options['dit_checkpoint_path'], str(self.checkpoint))
        self.assertEqual(options['video_dit_config'], {'text_len': 128})
        self.assertEqual(options['condition_latent_loss_weight'], 1.0)
        self.assertEqual(factory.call_args.kwargs, {'model_dtype': torch.bfloat16, 'device': 'cuda:0'})
        self.assertEqual(self.cfg.model.dit_checkpoint_path, '/original/stage2.pth')
        self.assertIs(result, self.model)

    def test_main_reports_invalid_anchor_as_cli_error(self):
        stack, _, _, _ = self.dependencies()
        self.dataset.window_indices.side_effect = IndexError('anchor outside RGB history')
        args = self.args('--dry-run')
        with stack, patch.object(infer.build_parser().__class__, 'parse_args', return_value=args), \
             redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            infer.main([])
        self.assertEqual(caught.exception.code, 2)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()

"""Generate sixteen object-video frames from RGB history and cached text only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / 'configs/obj_video_baseline_v1.yaml'


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def atomic_json(path: Path, value: dict) -> None:
    payload = (json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_config(path):
    from .config import load_config as shared_load_config

    return shared_load_config(path)


def make_dataset(cfg, manifest):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    options = OmegaConf.to_container(cfg.data.test, resolve=True)
    options.update(manifest_path=str(manifest), training=False, balance_tasks=False)
    return instantiate(options)


def make_model(cfg, checkpoint, device):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    precision = str(cfg.get('mixed_precision', 'bf16'))
    dtypes = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32, 'no': torch.float32}
    if precision not in dtypes:
        raise ValueError(f'Unsupported configured mixed_precision: {precision}')
    options = OmegaConf.to_container(cfg.model, resolve=True)
    options['dit_checkpoint_path'] = str(checkpoint)
    return instantiate(options, model_dtype=dtypes[precision], device=device)


def write_video(frames, path, fps):
    from vpp2.utils.video_io import save_mp4

    save_mp4(frames, str(path), fps=fps)


def select_episode(records, task, episode):
    matches = [index for index, row in enumerate(records) if row.get('task') == task and row.get('episode') == episode]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one manifest episode {task}/{episode}; found {len(matches)}')
    return matches[0]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Saved DiT .pt/.pth checkpoint (or Stage-2 initializer)')
    parser.add_argument('--manifest', type=Path, help='Defaults to data.test.manifest_path from the training config')
    parser.add_argument('--task', required=True)
    parser.add_argument('--episode', required=True)
    parser.add_argument('--anchor', type=int, required=True, help='Anchor in deduplicated frame order, not source MP4 frame index')
    parser.add_argument('--output', type=Path, required=True, help='New directory containing future.mp4 and trace.json')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, help='Defaults to eval_num_inference_steps in the training config (10 for this baseline)')
    parser.add_argument('--device', default='cuda:0', help='Torch device; relative to CUDA_VISIBLE_DEVICES')
    parser.add_argument('--fps', type=int, default=6, help='Display FPS only; does not assert uniform simulation time')
    parser.add_argument('--dry-run', action='store_true', help='Check config, manifest and indices on CPU; decode no video and write nothing')
    return parser


def validate_args(args):
    if args.anchor < 0:
        raise ValueError('--anchor must be non-negative')
    if args.steps is not None and not 1 <= args.steps <= 1000:
        raise ValueError('--steps must be in [1, 1000]')
    if args.seed < 0:
        raise ValueError('--seed must be non-negative')
    if args.fps < 1:
        raise ValueError('--fps must be positive')
    if args.device != 'cpu' and not (args.device == 'cuda' or (args.device.startswith('cuda:') and args.device[5:].isdigit())):
        raise ValueError('--device must be cpu, cuda, or cuda:<visible-device-index>')
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing to overwrite output: {output}')
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return output, checkpoint


def validate_condition_sample(sample):
    import torch

    condition = sample['condition_video']
    context = sample['context']
    mask = sample['context_mask']
    if condition.ndim != 4 or tuple(condition.shape[:2]) != (3, 9):
        raise ValueError(f'Expected condition_video [3,9,H,W], got {tuple(condition.shape)}')
    if condition.shape[-2] % 16 or condition.shape[-1] % 16:
        raise ValueError('Condition height and width must be multiples of 16')
    if tuple(context.shape) != (128, 4096) or tuple(mask.shape) != (128,) or mask.dtype != torch.bool:
        raise ValueError('Expected cached context [128,4096] and a bool context_mask [128]')
    if not bool(torch.isfinite(condition).all()) or not bool(torch.isfinite(context).all()):
        raise ValueError('Non-finite inference input')
    if float(condition.min()) < -1.00001 or float(condition.max()) > 1.00001:
        raise ValueError('Condition RGB must use the training normalization [-1,1]')


def window_metadata(sample):
    excluded = {'video', 'condition_video', 'context', 'context_mask', 'prompt'}
    return {key: jsonable(value) for key, value in sample.items() if key not in excluded}


def run_inference(args):
    output, checkpoint = validate_args(args)
    cfg = load_config(args.config)
    steps = args.steps if args.steps is not None else int(cfg.get('eval_num_inference_steps', 10))
    if not 1 <= steps <= 1000:
        raise ValueError('Configured eval_num_inference_steps must be in [1, 1000]')
    config_path = (PROJECT_ROOT / args.config.expanduser()).resolve()
    manifest = (args.manifest or Path(str(cfg.data.test.manifest_path))).expanduser().resolve()
    dataset = make_dataset(cfg, manifest)
    episode_index = select_episode(dataset.records, args.task, args.episode)
    row = dataset.records[episode_index]
    indices = dataset.window_indices(episode_index, args.anchor, include_target=False)
    checkpoint_stat = checkpoint.stat()
    trace = {
        'schema': 'vpp2_obj.inference.v1', 'dry_run': bool(args.dry_run),
        'config': str(config_path), 'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'manifest': str(manifest), 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'checkpoint': str(checkpoint), 'checkpoint_bytes': checkpoint_stat.st_size,
        'checkpoint_mtime_ns': checkpoint_stat.st_mtime_ns,
        'task': args.task, 'episode': args.episode, 'anchor_index': args.anchor,
        'sampling': jsonable(indices), 'condition_video_path': row['condition_video_path'],
        'text_embedding_path': row.get('text_embedding_path'),
        'text_embedding_sha256': row.get('text_embedding_sha256'),
        'text_embedding_cache_signature': row.get('text_embedding_cache_signature'),
        'prompt': row.get('prompt'), 'seed': args.seed, 'num_inference_steps': steps,
        'device': args.device, 'output_directory': str(output), 'exported_frames': 16,
        'decoded_frame_slice': [9, 25], 'prefix_clamped': False,
        'target_video_read': False, 'target_first_frame_read': False,
        'display_fps': args.fps,
        'display_fps_note': 'Playback rate only; deduplicated source timestamps need not be uniformly spaced.',
    }
    if args.dry_run:
        trace['status'] = 'dry_run_passed'
        return trace

    # Reserve the directory atomically before model loading. A concurrent run
    # cannot replace it, and failed attempts retain an explicit failure trace.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    try:
        sample = dataset.get_window(episode_index, args.anchor, include_target=False)
        validate_condition_sample(sample)
        trace['sampling'] = window_metadata(sample)
        trace['condition_shape'] = list(sample['condition_video'].shape)
        trace['context_shape'] = list(sample['context'].shape)
        trace['context_mask'] = sample['context_mask'].tolist()
        trace['model_visible_text_tokens'] = int(sample['context_mask'].sum())
        model = make_model(cfg, checkpoint, args.device)
        model.eval()
        result = model.infer_object_future(
            condition_video=sample['condition_video'], context=sample['context'],
            context_mask=sample['context_mask'], num_inference_steps=steps, seed=args.seed,
        )
        frames = result['video']
        if len(frames) != 16:
            raise ValueError(f'Model must export exactly sixteen future frames, got {len(frames)}')
        temporary_video = output / '.future.tmp.mp4'
        write_video(frames, temporary_video, args.fps)
        os.replace(temporary_video, output / 'future.mp4')
        trace.update(status='complete', video_path=str(output / 'future.mp4'))
        atomic_json(output / 'trace.json', trace)
        return trace
    except BaseException as exc:
        trace.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        atomic_json(output / 'failure.json', trace)
        raise


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        trace = run_inference(args)
    except (ValueError, FileNotFoundError, FileExistsError, IndexError) as exc:
        parser.error(str(exc))
    print(json.dumps(jsonable(trace), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

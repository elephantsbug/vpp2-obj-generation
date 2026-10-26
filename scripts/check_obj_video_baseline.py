#!/usr/bin/env python3
"""Review fixed paired-video windows; optionally check the frozen VAE on CUDA.

The default path is CPU-only. It independently decodes the manifest's RGB/object
source frame ordinals and compares them with the dataset's actual tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'third_party/vpp2/src'))

from vpp2_obj.infer import atomic_json, jsonable, load_config, select_episode, window_metadata


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs/obj_video_baseline_v1.yaml')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--split', choices=('train', 'test'), default='test')
    parser.add_argument('--task')
    parser.add_argument('--episode')
    parser.add_argument('--anchors', nargs='+', type=int, help='Fixed deduplicated anchors; default: first, midpoint and last complete-future window')
    parser.add_argument('--max-episodes', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True, help='New directory for contact sheets, indices and verification.json')
    parser.add_argument('--vae-check', action='store_true', help='Additionally load only the frozen VAE, reconstruct 25 frames and compare the first 3 latents')
    parser.add_argument('--vae-device', help='Explicit CUDA device for --vae-check, e.g. cuda:0 with CUDA_VISIBLE_DEVICES=4')
    return parser


def row_path(manifest, value):
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def decode_reference(path, indices, expected_size):
    """Independent RGB24 decode, without calling a dataset decode helper."""
    import av
    import numpy as np
    import torch

    wanted = set(int(value) for value in indices)
    decoded = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        for ordinal, frame in enumerate(container.decode(stream)):
            if ordinal in wanted:
                array = frame.to_ndarray(format='rgb24')
                if tuple(array.shape[:2]) != tuple(expected_size):
                    raise ValueError(f'Reference frame has the wrong dimensions: {path}')
                decoded[ordinal] = array
                if len(decoded) == len(wanted):
                    break
    if set(decoded) != wanted:
        raise ValueError(f'Missing source ordinals in {path}: {sorted(wanted - set(decoded))}')
    pixels = np.stack([decoded[int(index)] for index in indices])
    return torch.from_numpy(pixels).permute(3, 0, 1, 2).contiguous().float().div(127.5).sub(1)


def tensor_frames(video):
    import torch
    from PIL import Image

    pixels = ((video.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return [Image.fromarray(pixels[:, index].permute(1, 2, 0).numpy()) for index in range(pixels.shape[1])]


def contact_sheet(video, metadata, destination, title):
    from PIL import Image, ImageDraw, ImageFont

    frames = tensor_frames(video)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
    except OSError:
        font = ImageFont.load_default()
    width, height, label_height = 224, 112, 34
    columns, rows = 5, (len(frames) + 4) // 5
    canvas = Image.new('RGB', (columns * width, 36 + rows * (height + label_height)), '#202020')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill='white', font=font)
    condition = metadata['condition_indices']
    future = metadata['future_indices']
    condition_source = metadata['condition_source_indices']
    future_source = metadata['future_source_indices']
    is_pad = metadata['image_is_pad']
    for index, frame in enumerate(frames):
        x = (index % columns) * width
        y = 36 + (index // columns) * (height + label_height)
        canvas.paste(frame.resize((width, height), Image.Resampling.BILINEAR), (x, y))
        if index < 9:
            role, logical, source = 'RGB', condition[index], condition_source[index]
            color = '#80e0b0'
        else:
            role, logical, source = 'OBJ', future[index - 9], future_source[index - 9]
            color = '#ffd080'
        caption = f'{index:02d} {role} dedup={logical} source={source}'
        draw.text((x + 4, y + height + 2), caption, fill=color, font=font)
        if is_pad[index]:
            draw.text((x + 4, y + height + 16), 'repeated history / padding', fill='#ff9090', font=font)
    canvas.save(destination)


def validate_window(dataset, episode_index, anchor, manifest):
    import numpy as np
    import torch

    sample = dataset.get_window(episode_index, anchor, include_target=True)
    row = dataset.records[episode_index]
    with np.load(row_path(manifest, row['dedup_index_path']), allow_pickle=False) as data:
        sources = data['source_frame_indices'].copy()
        timestamps = data['timestamps_us'].copy()
    requested_history = anchor + np.arange(-(dataset.history_frames - 1), 1) * dataset.history_stride
    expected_condition = np.concatenate(([0], np.maximum(requested_history, 0))).astype(np.int64)
    expected_future = anchor + np.arange(1, dataset.prediction_frames + 1) * dataset.prediction_stride
    expected_pad = np.concatenate(([False], requested_history < 0, np.zeros(16, dtype=bool)))
    expected = {
        'condition_indices': expected_condition, 'future_indices': expected_future,
        'condition_source_indices': sources[expected_condition],
        'future_source_indices': sources[expected_future],
        'condition_timestamps_us': timestamps[expected_condition],
        'future_timestamps_us': timestamps[expected_future], 'image_is_pad': expected_pad,
    }
    for key, value in expected.items():
        if not np.array_equal(sample[key].cpu().numpy(), value):
            raise ValueError(f'Dataset {key} differs from independently calculated NPZ indices')
    video, condition = sample['video'], sample['condition_video']
    if tuple(video.shape) != (3, 25, *dataset.video_size) or tuple(condition.shape) != (3, 9, *dataset.video_size):
        raise ValueError('The window is not nine RGB conditions plus sixteen object future frames')
    if not torch.equal(video[:, :9], condition):
        raise ValueError('Target RGB prefix must equal the condition pixel-for-pixel')
    reference_rgb = decode_reference(row_path(manifest, row['condition_video_path']), sources[expected_condition], dataset.video_size)
    reference_obj = decode_reference(row_path(manifest, row['target_video_path']), sources[expected_future], dataset.video_size)
    rgb_difference = float((condition - reference_rgb).abs().max())
    object_difference = float((video[:, 9:] - reference_obj).abs().max())
    if rgb_difference > 1e-6 or object_difference > 1e-6:
        raise ValueError(f'Actual decoded RGB/OBJ order mismatch: rgb={rgb_difference}, obj={object_difference}')
    if not bool(torch.isfinite(video).all()) or not bool(torch.isfinite(sample['context']).all()):
        raise ValueError('Non-finite video or text tensor')
    if tuple(sample['context'].shape) != (128, 4096) or tuple(sample['context_mask'].shape) != (128,) or sample['context_mask'].dtype != torch.bool:
        raise ValueError('Text cache shape or real mask dtype differs from the training contract')
    if float(video.min()) < -1.00001 or float(video.max()) > 1.00001:
        raise ValueError('Invalid RGB normalization')
    metadata = window_metadata(sample)
    latent_pad = np.concatenate((expected_pad[:1], expected_pad[1:].reshape(-1, 4).all(axis=1)))
    metadata.update(
        condition_shape=list(condition.shape), video_shape=list(video.shape),
        context_shape=list(sample['context'].shape), context_dtype=str(sample['context'].dtype),
        context_mask=sample['context_mask'].tolist(), visible_text_tokens=int(sample['context_mask'].sum()),
        latent_is_pad=latent_pad.tolist(),
        rgb_reference_max_abs_difference=rgb_difference,
        object_reference_max_abs_difference=object_difference,
        prefix_pixel_identity=True, independent_frame_order_check=True,
        condition_video_path=str(row_path(manifest, row['condition_video_path'])),
        target_video_path=str(row_path(manifest, row['target_video_path'])),
        prompt=sample['prompt'], text_embedding_path=row['text_embedding_path'],
        text_embedding_cache_signature=row['text_embedding_cache_signature'],
    )
    return sample, metadata


def load_vae(cfg, device):
    import torch
    from vpp2.models.wan21_14b.helpers.io import load_state_dict
    from vpp2.models.wan21_14b.helpers.loader import _convert_vae_state_dict
    from vpp2.models.wan21_14b.wan_video_vae import WanVideoVAE

    selected = torch.device(device)
    if selected.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('--vae-check requires the explicitly selected available CUDA device')
    torch.cuda.set_device(selected)
    checkpoint = Path(str(cfg.model.model_id)) / 'Wan2.1_VAE.pth'
    state = load_state_dict(str(checkpoint), torch_dtype=torch.float32, device='cpu')
    vae = WanVideoVAE(z_dim=16)
    vae.load_state_dict(_convert_vae_state_dict(state), strict=True, assign=True)
    del state
    return vae.eval().requires_grad_(False).to(device=selected, dtype=torch.float32)


def check_vae_window(vae, sample, metadata, output, device):
    import torch
    from vpp2.utils.video_io import save_mp4

    with torch.inference_mode():
        full_latents = vae.encode([sample['video'].float()], device=device, tiled=False)
        condition_latents = vae.encode([sample['condition_video'].float()], device=device, tiled=False)
        if full_latents.shape[2] != 7 or condition_latents.shape[2] != 3:
            raise ValueError('Expected 25 RGB -> 7 latents and 9 condition RGB -> 3 latents')
        difference = (full_latents[:, :, :3] - condition_latents).abs()
        prefix_equal = bool(torch.allclose(full_latents[:, :, :3], condition_latents, atol=1e-5, rtol=1e-5))
        reconstruction = vae.decode(full_latents, device=device, tiled=False)[0].float().cpu()
    if not bool(torch.isfinite(reconstruction).all()):
        raise ValueError('Non-finite VAE reconstruction')
    per_frame_mse = (reconstruction - sample['video']).square().mean(dim=(0, 2, 3))
    result = {
        'full_latent_shape': list(full_latents.shape), 'condition_latent_shape': list(condition_latents.shape),
        'prefix_latents_allclose': prefix_equal,
        'prefix_latents_max_abs_difference': float(difference.max()),
        'prefix_latents_mean_abs_difference': float(difference.mean()),
        'rgb_prefix_mse': float(per_frame_mse[:9].mean()),
        'obj_future_mse': float(per_frame_mse[9:].mean()),
        'first_obj_future_frame_mse': float(per_frame_mse[9]),
        'per_frame_mse': per_frame_mse.tolist(), 'vae_device': device, 'frozen_vae_only': True,
    }
    contact_sheet(reconstruction, metadata, output / 'vae_reconstruction.png', 'Frozen VAE reconstruction: RGB prefix then object future')
    frames = tensor_frames(reconstruction)
    save_mp4(frames, str(output / 'vae_reconstruction_25.mp4'), fps=6)
    save_mp4(frames[9:25], str(output / 'vae_reconstruction_future.mp4'), fps=6)
    atomic_json(output / 'vae_check.json', result)
    if not prefix_equal:
        raise ValueError('The independently encoded RGB condition is not the first three target latents; inspect vae_check.json')
    return result


def run_checks(args):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    if bool(args.task) != bool(args.episode):
        raise ValueError('--task and --episode must be provided together')
    if args.max_episodes < 1 or (args.anchors is not None and any(value < 0 for value in args.anchors)):
        raise ValueError('max-episodes must be positive and anchors non-negative')
    if args.vae_check and not args.vae_device:
        raise ValueError('--vae-check requires an explicit --vae-device')
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing to overwrite review directory: {output}')
    cfg = load_config(args.config)
    options = OmegaConf.to_container(cfg.data[args.split], resolve=True)
    manifest = (args.manifest or Path(options['manifest_path'])).expanduser().resolve()
    options.update(manifest_path=str(manifest), training=False, balance_tasks=False)
    dataset = instantiate(options)
    if args.task:
        episodes = [select_episode(dataset.records, args.task, args.episode)]
    else:
        episodes = [index for index in range(len(dataset.records)) if dataset.episode_length(index) > dataset.future_horizon][:args.max_episodes]
    if not episodes:
        raise ValueError('No complete-future episodes selected for review')
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        'schema': 'vpp2_obj.baseline_review.v1', 'status': 'running',
        'config': str(args.config.resolve()), 'manifest': str(manifest),
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'split': args.split, 'vae_requested': bool(args.vae_check),
        'gpu_used': False, 'windows': [],
    }
    atomic_json(output / 'verification.json', report)
    try:
        samples = []
        for episode_index in episodes:
            last_anchor = dataset.episode_length(episode_index) - dataset.future_horizon - 1
            anchors = list(dict.fromkeys(args.anchors if args.anchors is not None else [0, last_anchor // 2, last_anchor]))
            for anchor in anchors:
                sample, metadata = validate_window(dataset, episode_index, anchor, manifest)
                window_dir = output / f'window_{len(samples):03d}'
                window_dir.mkdir()
                atomic_json(window_dir / 'indices.json', metadata)
                contact_sheet(sample['video'], metadata, window_dir / 'contact_sheet.png', f"{metadata['task']}/{metadata['episode']} anchor={anchor} (deduplicated order)")
                report['windows'].append({
                    'episode_index': episode_index, 'task': metadata['task'], 'episode': metadata['episode'],
                    'anchor_index': anchor, 'directory': str(window_dir), 'cpu_validation_passed': True,
                    'rgb_reference_max_abs_difference': metadata['rgb_reference_max_abs_difference'],
                    'object_reference_max_abs_difference': metadata['object_reference_max_abs_difference'],
                })
                samples.append((sample, metadata, window_dir))
                atomic_json(output / 'verification.json', report)
        if args.vae_check:
            vae = load_vae(cfg, args.vae_device)
            report['gpu_used'] = True
            for record, (sample, metadata, window_dir) in zip(report['windows'], samples):
                record['vae'] = check_vae_window(vae, sample, metadata, window_dir, args.vae_device)
                atomic_json(output / 'verification.json', report)
            del vae
            torch.cuda.empty_cache()
        report.update(status='complete', windows_checked=len(samples), training_started=False)
        atomic_json(output / 'verification.json', report)
        return report
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        atomic_json(output / 'verification.json', report)
        raise


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_checks(args)
    except (ValueError, FileExistsError, FileNotFoundError, IndexError) as exc:
        parser.error(str(exc))
    print(json.dumps(jsonable(report), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

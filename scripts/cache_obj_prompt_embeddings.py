#!/usr/bin/env python3
"""Cache frozen Wan UMT5 embeddings with explicit prompt/tokenizer provenance.

This script neither edits dataset manifests nor fits a model. Run with the
isolated training Python and CUDA_VISIBLE_DEVICES=4. A fresh output directory is
required unless --resume is given; resume only accepts the identical contract.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback


PROJECT = Path(__file__).resolve().parents[1]
SCHEMA = 'vpp2.frozen_wan_text_embedding_cache.v1'
LENGTH = 128
WIDTH = 4096


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode('utf-8')).hexdigest()


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: dict) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


def atomic_torch_save(torch, path: Path, value: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=PROJECT / 'runs/prompt_obj_chronological_128_v1/exports/paired_training.jsonl')
    parser.add_argument('--tokenizer', type=Path, default=PROJECT / 'runs/prompt_obj_chronological_128_v1/tokenizer')
    parser.add_argument('--wan-root', type=Path, default=PROJECT / 'weights/checkpoints/Wan2.1-I2V-14B-480P')
    parser.add_argument('--output', type=Path, default=PROJECT / 'data/text_embeddings/prompt_obj_chronological_128_v1_20261019')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--physical-gpu', default='4')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.physical_gpu:
        raise ValueError(f'Explicitly set CUDA_VISIBLE_DEVICES={args.physical_gpu}; no other GPU may be visible')
    output = args.output.resolve()
    existed = output.exists()
    if existed and not args.resume:
        raise FileExistsError(f'Refusing to overwrite an existing cache: {output}; --resume requires an identical contract')
    if existed and not (output / 'cache_contract.json').is_file():
        raise FileExistsError(f'Refusing to adopt an unrecognized cache directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.cache.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    signature = None
    completed = 0
    total = None
    authorized_output = not existed

    def status(phase, state='running', **extra):
        payload = {
            'schema': SCHEMA, 'status': state, 'phase': phase,
            'cache_signature': signature, 'pid': os.getpid(),
            'updated_utc': datetime.now(timezone.utc).isoformat(),
            'completed_embeddings': completed, 'unique_prompts': total,
            'elapsed_seconds': round(time.monotonic() - started, 3), **extra,
        }
        atomic_json(output / 'status.json', payload)
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    try:
        if authorized_output:
            status('fingerprinting_inputs')
        source = args.source.resolve()
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError('Source contains no rows')
        episode_keys = [(row['task'], row['episode']) for row in rows]
        if len(episode_keys) != len(set(episode_keys)):
            raise ValueError('Source repeats a task/episode key')
        if any(not isinstance(row.get('prompt'), str) or not row['prompt'].strip() for row in rows):
            raise ValueError('Every source row must contain a non-empty prompt string')
        prompt_counts = Counter(row['prompt'] for row in rows)
        prompts_by_hash = {prompt_hash(prompt): prompt for prompt in prompt_counts}
        if len(prompts_by_hash) != len(prompt_counts):
            raise ValueError('Prompt hash collision')
        keys = sorted(prompts_by_hash)
        prompts = [prompts_by_hash[key] for key in keys]
        total = len(keys)
        source_sha256 = sha256_file(source)
        checkpoint = args.wan_root.resolve() / 'models_t5_umt5-xxl-enc-bf16.pth'
        encoder_source = PROJECT / 'third_party/vpp2/src/vpp2/models/wan21_14b/wan_video_text_encoder.py'
        tokenizer_source = args.tokenizer.resolve()
        tokenizer_fingerprints = {}
        for path in sorted(tokenizer_source.iterdir()):
            if not path.is_file():
                raise ValueError(f'Unexpected tokenizer directory entry: {path}')
            digest = sha256_file(path)
            corresponding_weight_file = args.wan_root.resolve() / 'google/umt5-xxl' / path.name
            if not corresponding_weight_file.is_file() or sha256_file(corresponding_weight_file) != digest:
                raise ValueError(f'Tokenizer snapshot differs from pretrained weights: {path.name}')
            tokenizer_fingerprints[path.name] = digest
        if not tokenizer_fingerprints:
            raise ValueError('Tokenizer snapshot is empty')

        sys.path.insert(0, str(PROJECT / 'third_party/vpp2/src'))
        import torch
        from vpp2.models.wan21_14b.wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder

        torch.set_num_threads(4)
        identity = {
            'schema': SCHEMA, 'context_length': LENGTH, 'context_width': WIDTH,
            'context_dtype': 'bfloat16', 'mask_dtype': 'bool', 'input_ids_dtype': 'int64',
            'prompt_hash_algorithm': 'sha256_utf8_exact_original_prompt',
            'source_sha256': source_sha256,
            'prompt_set_sha256': hashlib.sha256(('\n'.join(keys) + '\n').encode()).hexdigest(),
            'encoder_checkpoint_sha256': sha256_file(checkpoint),
            'encoder_source_sha256': sha256_file(encoder_source),
            'tokenizer_files_sha256': tokenizer_fingerprints,
            'tokenizer_class': 'vpp2.models.wan21_14b.wan_video_text_encoder.HuggingfaceTokenizer',
            'tokenizer_clean': 'whitespace', 'add_special_tokens': True,
            'padding': 'max_length', 'truncation_policy': 'reject_any_prompt_exceeding_128_including_EOS',
            'encoder_class': 'WanTextEncoder', 'encoder_training': False,
            'requires_grad': False, 'compute_dtype': 'bfloat16',
            'padded_context': 'preserve_encoder_output_with_real_attention_mask',
            'packages': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'tokenizers', 'ftfy', 'regex']},
        }
        signature = canonical_hash(identity)
        contract = {
            **identity, 'cache_signature': signature,
            'source_path': str(source), 'encoder_checkpoint_path': str(checkpoint),
            'tokenizer_source_path': str(tokenizer_source), 'tokenizer_snapshot_path': str(output / 'tokenizer'),
            'source_rows': len(rows), 'unique_prompts': total,
            'embedding_path_template': str(output / 'embeddings/<prompt_sha256>.pt'),
            'loading': 'meta_module + torch.load(weights_only=True,mmap=True,map_location=cpu) + strict_assign + cuda_bfloat16',
            'gpu_visibility': args.physical_gpu, 'batch_size': args.batch_size,
            'created_utc': datetime.now(timezone.utc).isoformat(),
        }
        if existed:
            existing_contract = json.loads((output / 'cache_contract.json').read_text())
            if existing_contract.get('cache_signature') != signature:
                raise ValueError('Existing cache contract signature differs; refusing to overwrite')
            authorized_output = True
        else:
            atomic_json(output / 'cache_contract.json', contract)
        previous_file_hashes = {}
        previous_index_path = output / 'index.jsonl'
        if existed and previous_index_path.exists():
            previous_summary_path = output / 'summary.json'
            if previous_summary_path.exists():
                previous_summary = json.loads(previous_summary_path.read_text())
                if previous_summary.get('cache_signature') != signature or previous_summary.get('index_sha256') != sha256_file(previous_index_path):
                    raise ValueError('Existing summary/index digest is invalid; refusing to resume')
            previous_entries = [json.loads(line) for line in previous_index_path.read_text().splitlines() if line.strip()]
            for entry in previous_entries:
                key = entry['prompt_sha256']
                if key in previous_file_hashes or key not in prompts_by_hash or entry.get('cache_signature') != signature or entry.get('prompt') != prompts_by_hash[key]:
                    raise ValueError('Existing cache index provenance is invalid')
                previous_file_hashes[key] = entry['sha256']
        (output / 'embeddings').mkdir(exist_ok=True)
        (output / 'tokenizer').mkdir(exist_ok=True)
        for name, expected_hash in tokenizer_fingerprints.items():
            destination = output / 'tokenizer' / name
            if destination.exists():
                if sha256_file(destination) != expected_hash:
                    raise ValueError(f'Existing tokenizer snapshot is invalid: {destination}')
            else:
                atomic_bytes(destination, (tokenizer_source / name).read_bytes())
        status('tokenizing_all_unique_prompts', source_rows=len(rows))
        tokenizer = HuggingfaceTokenizer(name=str(output / 'tokenizer'), seq_len=LENGTH, clean='whitespace', local_files_only=True)
        raw_ids = tokenizer.tokenizer([tokenizer._clean(prompt) for prompt in prompts], add_special_tokens=True, truncation=False, padding=False)['input_ids']
        token_counts = [len(value) for value in raw_ids]
        if any(count > LENGTH for count in token_counts):
            too_long = [(keys[i], token_counts[i]) for i in range(total) if token_counts[i] > LENGTH]
            raise ValueError(f'Prompts would be truncated: {too_long[:20]}')
        expected_ids, expected_mask = tokenizer(prompts, return_mask=True, add_special_tokens=True)
        expected_ids = expected_ids.to(dtype=torch.int64, device='cpu')
        expected_mask = expected_mask.to(dtype=torch.bool, device='cpu')
        if tuple(expected_ids.shape) != (total, LENGTH) or tuple(expected_mask.shape) != (total, LENGTH):
            raise ValueError('Tokenizer produced an unexpected shape')
        for index, values in enumerate(raw_ids):
            if expected_ids[index][expected_mask[index]].tolist() != values:
                raise ValueError(f'Padded token IDs do not match untruncated tokenization: {keys[index]}')
            if not values or values[-1] != tokenizer.tokenizer.eos_token_id:
                raise ValueError(f'Missing special EOS: {keys[index]}')
            if not torch.equal(expected_mask[index], torch.arange(LENGTH) < len(values)):
                raise ValueError('Expected contiguous right-padding mask')
            if bool((expected_ids[index][~expected_mask[index]] != tokenizer.tokenizer.pad_token_id).any()):
                raise ValueError('Padding token IDs are invalid')
        expected_by_hash = {key: index for index, key in enumerate(keys)}
        source_count_mismatches = [
            (row['task'], row['episode']) for row in rows
            if row.get('prompt_token_count') is not None
            and int(row['prompt_token_count']) != token_counts[expected_by_hash[prompt_hash(row['prompt'])]]
        ]
        if source_count_mismatches:
            raise ValueError(f'Token counts differ from prompt export: {source_count_mismatches[:20]}')

        def validate_file(index):
            key = keys[index]
            path = output / 'embeddings' / f'{key}.pt'
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if payload.get('schema') != SCHEMA or payload.get('cache_signature') != signature or payload.get('prompt_sha256') != key or payload.get('prompt') != prompts[index]:
                raise ValueError(f'Embedding provenance mismatch: {path}')
            context, mask, ids = payload['context'], payload['mask'], payload['input_ids']
            if tuple(context.shape) != (LENGTH, WIDTH) or context.dtype != torch.bfloat16 or context.device.type != 'cpu':
                raise ValueError(f'Invalid context shape/dtype/device: {path}')
            if tuple(mask.shape) != (LENGTH,) or mask.dtype != torch.bool or not torch.equal(mask, expected_mask[index]):
                raise ValueError(f'Invalid attention mask: {path}')
            if tuple(ids.shape) != (LENGTH,) or ids.dtype != torch.int64 or not torch.equal(ids, expected_ids[index]):
                raise ValueError(f'Invalid token IDs: {path}')
            if not bool(torch.isfinite(context).all()) or not bool(context[mask].abs().sum() > 0):
                raise ValueError(f'Non-finite or empty embedding: {path}')
            if payload.get('valid_token_count') != token_counts[index]:
                raise ValueError(f'Invalid token-count metadata: {path}')
            file_digest = sha256_file(path)
            if key in previous_file_hashes and file_digest != previous_file_hashes[key]:
                raise ValueError(f'Embedding differs from its previously recorded SHA256: {path}')
            return {
                'prompt_sha256': key, 'prompt': prompts[index], 'path': str(path),
                'sha256': file_digest, 'valid_token_count': token_counts[index],
                'context_shape': [LENGTH, WIDTH], 'context_dtype': 'bfloat16',
                'mask_dtype': 'bool', 'cache_signature': signature,
                'source_row_count': prompt_counts[prompts[index]],
            }

        pending = []
        for index, key in enumerate(keys):
            if (output / 'embeddings' / f'{key}.pt').exists():
                validate_file(index)
                completed += 1
            else:
                pending.append(index)
        peak_allocated = 0
        peak_reserved = 0
        if pending:
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError('Exactly the selected CUDA GPU must be available')
            torch.cuda.set_device(0)
            torch.manual_seed(0)
            status('loading_frozen_t5', gpu_name=torch.cuda.get_device_name(0), token_count_min=min(token_counts), token_count_max=max(token_counts))
            with torch.device('meta'):
                encoder = WanTextEncoder()
            weights = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
            if len(weights) == 1:
                for name in ['state_dict', 'module', 'model_state']:
                    if name in weights:
                        weights = weights[name]
                        break
            encoder.load_state_dict(weights, strict=True, assign=True)
            del weights
            encoder = encoder.to(device='cuda:0', dtype=torch.bfloat16).eval().requires_grad_(False)
            if any(parameter.requires_grad for parameter in encoder.parameters()) or encoder.training:
                raise RuntimeError('Encoder must remain frozen and in evaluation mode')
            torch.cuda.reset_peak_memory_stats(0)
            with torch.inference_mode():
                for offset in range(0, len(pending), args.batch_size):
                    indices = pending[offset:offset + args.batch_size]
                    ids = expected_ids[indices].to('cuda:0')
                    masks = expected_mask[indices].to('cuda:0')
                    contexts = encoder(ids, masks)
                    if tuple(contexts.shape) != (len(indices), LENGTH, WIDTH) or not bool(torch.isfinite(contexts).all()):
                        raise RuntimeError('Encoder output shape or finiteness check failed')
                    contexts = contexts.to(device='cpu', dtype=torch.bfloat16)
                    for batch_index, index in enumerate(indices):
                        key = keys[index]
                        payload = {
                            'schema': SCHEMA, 'schema_version': 1,
                            'context': contexts[batch_index].contiguous().clone(),
                            'mask': expected_mask[index].clone(), 'input_ids': expected_ids[index].clone(),
                            'prompt': prompts[index], 'prompt_hash': key, 'prompt_sha256': key,
                            'cache_signature': signature, 'valid_token_count': token_counts[index],
                        }
                        atomic_torch_save(torch, output / 'embeddings' / f'{key}.pt', payload)
                        completed += 1
                    status('encoding', batch_size=len(indices))
            peak_allocated = torch.cuda.max_memory_allocated(0)
            peak_reserved = torch.cuda.max_memory_reserved(0)
            del encoder, contexts, ids, masks
            torch.cuda.empty_cache()
        status('validating_all_saved_embeddings')
        index_rows = [validate_file(index) for index in range(total)]
        if set(path.name for path in (output / 'embeddings').glob('*.pt')) != {f'{key}.pt' for key in keys}:
            raise ValueError('Cache has missing or unexpected PT files')
        if sha256_file(source) != source_sha256:
            raise RuntimeError('Prompt source changed during caching')
        for name, digest in tokenizer_fingerprints.items():
            if sha256_file(output / 'tokenizer' / name) != digest:
                raise RuntimeError('Tokenizer snapshot changed during caching')
        index_content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in index_rows)
        atomic_bytes(output / 'index.jsonl', index_content.encode('utf-8'))
        summary = {
            'schema': SCHEMA, 'status': 'complete', 'cache_signature': signature,
            'source_rows': len(rows), 'unique_prompts': total, 'embedding_files': len(index_rows),
            'deduplicated_rows': len(rows) - total, 'verified_embeddings': len(index_rows),
            'context_shape': [LENGTH, WIDTH], 'context_dtype': 'bfloat16', 'mask_dtype': 'bool',
            'valid_token_count_min': min(token_counts), 'valid_token_count_max': max(token_counts),
            'valid_token_count_histogram': dict(sorted(Counter(token_counts).items())),
            'truncated_prompts': 0, 'nonfinite_embeddings': 0, 'source_token_count_mismatches': 0,
            'source_sha256': source_sha256, 'index_sha256': sha256_file(output / 'index.jsonl'),
            'encoder_checkpoint_sha256': identity['encoder_checkpoint_sha256'],
            'tokenizer_snapshot_matches_weights': True, 'physical_gpu': args.physical_gpu,
            'peak_cuda_allocated_bytes': peak_allocated, 'peak_cuda_reserved_bytes': peak_reserved,
            'bytes_embeddings': sum(Path(row['path']).stat().st_size for row in index_rows),
            'elapsed_seconds': round(time.monotonic() - started, 3),
            'encoder_frozen': True, 'model_training_performed': False,
            'validated': ['prompt_hash', 'cache_signature', 'context_shape', 'context_dtype', 'mask_shape', 'real_attention_mask', 'input_ids', 'EOS', 'padding_ids', 'finite_context', 'nonzero_valid_context', 'file_sha256'],
        }
        atomic_json(output / 'summary.json', summary)
        status('complete', state='complete', summary_path=str(output / 'summary.json'), index_path=str(output / 'index.jsonl'))
    except BaseException as exc:
        if authorized_output:
            status('failed', state='failed', error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()

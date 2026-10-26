#!/usr/bin/env python3
"""Independently validate refreshed frozen splits and cached text on CPU.

This never loads the UMT5 encoder checkpoint, decodes media, or changes datasets.
Use --write-report after the refresh transaction and cache generation have ended.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time


PROJECT = Path(__file__).resolve().parents[1]
SCHEMA = 'vpp2.frozen_wan_text_embedding_cache.v1'
ENCODER_SHA = '7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d'
IDENTITY_FIELDS = (
    'schema', 'context_length', 'context_width', 'context_dtype', 'mask_dtype',
    'input_ids_dtype', 'prompt_hash_algorithm', 'source_sha256',
    'prompt_set_sha256', 'encoder_checkpoint_sha256', 'encoder_source_sha256',
    'tokenizer_files_sha256', 'tokenizer_class', 'tokenizer_clean',
    'add_special_tokens', 'padding', 'truncation_policy', 'encoder_class',
    'encoder_training', 'requires_grad', 'compute_dtype', 'padded_context',
    'packages',
)
PROMPT_MUTABLE = {
    'prompt', 'prompt_records_path', 'prompt_records_sha256',
    'text_embedding_cache_must_be_rebuilt',
}
DEDUP_FIELDS = (
    'dedup_index_path', 'dedup_index_sha256', 'source_frame_count',
    'deduplicated_frame_count', 'sampling_index_space',
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical_sha(value):
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(',', ':')).encode('utf-8'))


def episode_key(row):
    return row['task'], row['episode']


class Audit:
    def __init__(self):
        self.inputs = {}
        self.checks = Counter()
        self.errors = []

    def check(self, condition, name, item=None):
        self.checks[name] += 1
        if not condition:
            self.errors.append({'check': name, 'item': item})

    def read(self, path):
        path = Path(path)
        data = path.read_bytes()
        self.inputs[str(path)] = {'sha256': sha(data), 'size_bytes': len(data)}
        return data

    def json(self, path):
        return json.loads(self.read(path))

    def rows(self, path, identity=episode_key):
        rows = [json.loads(line) for line in self.read(path).splitlines() if line]
        keys = [identity(row) for row in rows]
        self.check(len(keys) == len(set(keys)), 'unique_record_keys', str(path))
        return rows, dict(zip(keys, rows))


def atomic_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.validation-',
                                     delete=False) as stream:
        temporary = Path(stream.name)
        stream.write((json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate(args):
    started = time.monotonic()
    audit = Audit()
    split = args.split_root.resolve()
    backup = args.backup.resolve()
    cache = args.cache.resolve()
    source = args.prompt_source.resolve()
    # Shared locks prevent reading an in-progress refresh or cache writer.
    locks = []
    for path in (split / '.prompt_refresh.lock', cache / '.cache.lock'):
        stream = path.open('r')
        fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        locks.append(stream)

    cache_status = audit.json(cache / 'status.json')
    cache_summary = audit.json(cache / 'summary.json')
    cc = audit.json(cache / 'cache_contract.json')
    if cache_status['status'] != 'complete' or cache_summary['status'] != 'complete':
        raise RuntimeError('Cache generation is incomplete; active validation was not replaced')
    if not (backup / 'validation.json').is_file():
        raise RuntimeError('The previous validation report must be backed up first')

    backup_manifest = audit.json(backup / 'backup_manifest.json')
    for name, item in backup_manifest.items():
        path = source if name == 'prompt_source' else backup / (
            'obj_video_data_contract.json' if name == 'data_contract' else name)
        audit.check(sha(audit.read(path)) == item['sha256'], 'backup_exact_sha256', name)

    source_rows, source_map = audit.rows(source)
    index_rows, cache_index = audit.rows(cache / 'index.jsonl', lambda row: row['prompt_sha256'])
    old, current = {}, {}
    for name in ('train', 'test', 'excluded_prompt_review'):
        old[name] = audit.rows(backup / (name + '.jsonl'))
        current[name] = audit.rows(split / (name + '.jsonl'))
    excluded_bytes = audit.read(split / 'excluded_prompt_review.jsonl')
    audit.check(excluded_bytes == audit.read(backup / 'excluded_prompt_review.jsonl'),
                'excluded_105_bytes_unchanged')
    train_keys = set(current['train'][1])
    test_keys = set(current['test'][1])
    excluded_keys = set(current['excluded_prompt_review'][1])
    audit.check((len(train_keys), len(test_keys), len(excluded_keys)) == (2155, 240, 105),
                'expected_partition_counts')
    audit.check(not train_keys & test_keys and not (train_keys | test_keys) & excluded_keys,
                'episode_partitions_disjoint')
    audit.check(train_keys | test_keys == set(source_map) and len(source_map) == 2395,
                'new_prompt_exact_episode_coverage')
    audit.check({key[0] for key in train_keys} == {key[0] for key in test_keys}
                and len({key[0] for key in train_keys}) == 50, 'all_tasks_in_both_splits')

    counts = Counter(row['prompt'] for row in source_rows)
    prompts = {sha(text.encode('utf-8')): text for text in counts}
    keys = sorted(prompts)
    audit.check(len(prompts) == len(counts) == args.expected_unique_prompts,
                'expected_unique_prompt_count')
    audit.check(set(cache_index) == set(prompts), 'cache_index_exact_prompt_coverage')
    audit.check({p.name for p in (cache / 'embeddings').glob('*.pt')}
                == {key + '.pt' for key in keys}, 'cache_files_exact_coverage')
    signature = canonical_sha({key: cc[key] for key in IDENTITY_FIELDS})
    audit.check(signature == cc['cache_signature'] == cache_status['cache_signature']
                == cache_summary['cache_signature'], 'cache_signature_recomputed')
    audit.check(cc['schema'] == SCHEMA and cc['source_sha256'] == sha(audit.read(source))
                == cache_summary['source_sha256'], 'cache_source_fingerprint')
    audit.check(cc['prompt_set_sha256'] == sha(('\n'.join(keys) + '\n').encode()),
                'cache_prompt_set_fingerprint')
    audit.check(cc['encoder_checkpoint_sha256'] == ENCODER_SHA
                == cache_summary['encoder_checkpoint_sha256'], 'encoder_recorded_official_sha256')
    # Stat only: this verifier intentionally does not reread the 11 GB checkpoint.
    checkpoint = Path(cc['encoder_checkpoint_path'])
    audit.check(checkpoint.is_file() and checkpoint.stat().st_size == 11361920418,
                'encoder_checkpoint_size_only')
    encoder_code = PROJECT / 'third_party/vpp2/src/vpp2/models/wan21_14b/wan_video_text_encoder.py'
    audit.check(sha(audit.read(encoder_code)) == cc['encoder_source_sha256'],
                'encoder_tokenizer_source_fingerprint')
    audit.check(cc['encoder_training'] is False and cc['requires_grad'] is False
                and cache_summary['encoder_frozen'] is True
                and cache_summary['model_training_performed'] is False, 'frozen_encoder_metadata')
    audit.check(cc['context_length'] == 128 and cc['context_width'] == 4096
                and cc['tokenizer_clean'] == 'whitespace' and cc['add_special_tokens'] is True,
                'cache_encoding_contract')
    audit.check(cache_summary['index_sha256'] == sha(audit.read(cache / 'index.jsonl')),
                'cache_index_fingerprint')
    prompt_contract = audit.json(source.parent.parent / 'prompt_contract.json')
    for name, digest in cc['tokenizer_files_sha256'].items():
        for directory in (cache / 'tokenizer', Path(prompt_contract['tokenizer_snapshot_path'])):
            audit.check(sha(audit.read(directory / name)) == digest, 'tokenizer_current_sha256',
                        str(directory / name))
        audit.check(prompt_contract['tokenizer_sha256'].get(name) == digest,
                    'tokenizer_matches_prompt_contract', name)

    dedup_checked = set()
    frame_counts = {}
    for split_name in ('train', 'test'):
        previous_rows, previous_map = old[split_name]
        rows, _ = current[split_name]
        audit.check([episode_key(row) for row in rows] == [episode_key(row) for row in previous_rows],
                    'frozen_partition_and_row_order_unchanged', split_name)
        frame_counts[split_name] = {'original': 0, 'deduplicated': 0}
        for row in rows:
            key = episode_key(row)
            previous, rewritten = previous_map[key], source_map[key]
            for field, value in previous.items():
                if field not in PROMPT_MUTABLE:
                    audit.check(row.get(field) == value, 'original_nonprompt_field_unchanged',
                                [*key, field])
            for field, value in rewritten.items():
                if field != 'text_embedding_cache_must_be_rebuilt':
                    audit.check(row.get(field) == value, 'new_prompt_export_field_exact',
                                [*key, field])
            audit.check(not row['prompt_review_flags'], 'no_flagged_optimization_or_test_row', key)
            for field in DEDUP_FIELDS:
                audit.check(row[field] == previous[field], 'dedup_field_unchanged', [*key, field])
            index_path = Path(row['dedup_index_path'])
            if index_path not in dedup_checked:
                audit.check(sha(index_path.read_bytes()) == row['dedup_index_sha256'],
                            'dedup_current_file_sha256', key)
                dedup_checked.add(index_path)
            frame_counts[split_name]['original'] += row['source_frame_count']
            frame_counts[split_name]['deduplicated'] += row['deduplicated_frame_count']
            prompt_key = sha(row['prompt'].encode('utf-8'))
            entry = cache_index[prompt_key]
            expected = cache / 'embeddings' / (prompt_key + '.pt')
            audit.check(Path(row['text_embedding_path']) == Path(entry['path']) == expected,
                        'row_cache_path_matches_raw_prompt_sha', key)
            audit.check(row['text_embedding_prompt_sha256'] == prompt_key
                        and row['text_embedding_sha256'] == entry['sha256']
                        and row['text_embedding_cache_signature'] == entry['cache_signature'] == signature,
                        'row_cache_fingerprints_match', key)
            audit.check(row['text_embedding_context_length'] == 128
                        and row['text_embedding_valid_tokens'] == entry['valid_token_count']
                        == row['prompt_token_count']
                        and row['text_embedding_cache_must_be_rebuilt'] is False,
                        'row_cache_ready_token_metadata', key)
    for key, row in current['excluded_prompt_review'][1].items():
        audit.check(bool(row['prompt_review_flags']), 'excluded_rows_still_flagged', key)

    leakage = {}
    for domain in ('rgb', 'obj'):
        sets = [{row[domain + '_sha256']['video'] for row in current[name][0]}
                for name in ('train', 'test')]
        leakage[domain + '_video_hash_overlap'] = len(sets[0] & sets[1])
    sets = [{str(Path(row[field]).resolve()) for row in current[name][0]
             for field in ('condition_video_path', 'target_video_path')} for name in ('train', 'test')]
    leakage['any_video_path_overlap'] = len(sets[0] & sets[1])
    audit.check(not any(leakage.values()), 'no_episode_media_leakage')

    # Import only the tokenizer wrapper; never construct or load WanTextEncoder.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    sys.path.insert(0, str(PROJECT / 'third_party/vpp2/src'))
    import torch
    from vpp2.models.wan21_14b.wan_video_text_encoder import HuggingfaceTokenizer
    torch.set_num_threads(2)
    tokenizer = HuggingfaceTokenizer(str(cache / 'tokenizer'), seq_len=128,
                                    clean='whitespace', local_files_only=True)
    texts = [prompts[key] for key in keys]
    raw = tokenizer.tokenizer([tokenizer._clean(text) for text in texts],
                              add_special_tokens=True, padding=False, truncation=False)['input_ids']
    expected_ids, expected_masks = tokenizer(texts, return_mask=True, add_special_tokens=True)
    expected_ids, expected_masks = expected_ids.long(), expected_masks.bool()
    token_counts = []
    for index, key in enumerate(keys):
        entry = cache_index[key]
        path = cache / 'embeddings' / (key + '.pt')
        data = path.read_bytes()
        audit.check(sha(data) == entry['sha256'], 'cache_tensor_file_actual_sha256', key)
        payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
        context, mask, ids = payload['context'], payload['mask'], payload['input_ids']
        audit.check(payload['schema'] == SCHEMA and payload['cache_signature'] == signature
                    and payload['prompt'] == entry['prompt'] == prompts[key]
                    and payload['prompt_sha256'] == payload['prompt_hash'] == key,
                    'cache_payload_provenance', key)
        audit.check(tuple(context.shape) == (128, 4096) and context.dtype == torch.bfloat16
                    and context.device.type == 'cpu', 'cache_context_shape_dtype_cpu', key)
        audit.check(tuple(ids.shape) == (128,) and ids.dtype == torch.int64
                    and torch.equal(ids, expected_ids[index]), 'cache_input_ids_match_tokenizer', key)
        audit.check(tuple(mask.shape) == (128,) and mask.dtype == torch.bool
                    and torch.equal(mask, expected_masks[index]), 'cache_real_mask_matches_tokenizer', key)
        count = len(raw[index])
        token_counts.append(count)
        audit.check(count <= 128 and ids[mask].tolist() == raw[index],
                    'no_token_truncation', key)
        audit.check(raw[index][-1] == tokenizer.tokenizer.eos_token_id
                    and torch.equal(mask, torch.arange(128) < count)
                    and bool((ids[~mask] == tokenizer.tokenizer.pad_token_id).all()),
                    'eos_and_right_padding_correct', key)
        audit.check(bool(torch.isfinite(context).all()) and bool(context[mask].abs().sum() > 0),
                    'embedding_finite_and_nonzero_valid_context', key)
        audit.check(payload['valid_token_count'] == entry['valid_token_count'] == count
                    and entry['source_row_count'] == counts[prompts[key]],
                    'embedding_token_count_and_reuse_count', key)
        if (index + 1) % 200 == 0 or index + 1 == len(keys):
            print(json.dumps({'verified_unique_embeddings': index + 1,
                              'issues': len(audit.errors)}), flush=True)
    audit.check(not torch.cuda.is_initialized(), 'verification_never_initialized_cuda')
    audit.check(cache_summary['unique_prompts'] == cache_summary['embedding_files']
                == cache_summary['verified_embeddings'] == len(keys)
                and cache_summary['source_rows'] == len(source_rows), 'cache_summary_counts')
    audit.check(cache_summary['valid_token_count_min'] == min(token_counts)
                and cache_summary['valid_token_count_max'] == max(token_counts)
                and {int(k): v for k, v in cache_summary['valid_token_count_histogram'].items()}
                == dict(Counter(token_counts)), 'cache_summary_token_counts_recomputed')

    summary = audit.json(split / 'summary.json')
    config = audit.json(args.data_contract.resolve())
    for name in ('train', 'test', 'excluded_prompt_review'):
        filename = name + '.jsonl'
        audit.check(summary['artifacts'][filename]['sha256'] == sha(audit.read(split / filename)),
                    'summary_artifact_actual_sha256', filename)
    audit.check(summary['frames'] == frame_counts, 'summary_frame_totals')
    audit.check(summary['counts'] == audit.json(backup / 'summary.json')['counts'],
                'summary_partition_counts_unchanged')
    audit.check(summary['prompt_source']['sha256'] == sha(audit.read(source))
                and summary['prompt_revision'] == 'prompt_obj_chronological_128_v1',
                'summary_new_prompt_source')
    audit.check(summary['text_embedding_cache']['cache_signature'] == signature
                and summary['text_embedding_cache']['status'] == 'complete', 'summary_cache_ready')
    audit.check(Path(config['current_training_manifest']).resolve() == split / 'train.jsonl'
                == Path(config['manifests']['train']).resolve(), 'config_optimization_entry_only_train')
    audit.check(Path(config['current_test_manifest']).resolve() == split / 'test.jsonl'
                == Path(config['manifests']['test']).resolve(), 'config_held_out_test_entry')
    audit.check(Path(config['prompt_source_manifest']).resolve() == source
                and config['text_embeddings']['cache_signature'] == signature
                and config['text_embeddings']['status'] == 'complete', 'config_prompt_and_cache_identity')
    audit.check(config['reuse_previous_text_embeddings'] is False, 'old_prompt_cache_reuse_disabled')

    for path, item in audit.inputs.items():
        audit.check(sha(Path(path).read_bytes()) == item['sha256'],
                    'input_unchanged_during_validation', path)
    report = {
        'status': 'passed' if not audit.errors else 'failed',
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds': round(time.monotonic() - started, 3),
        'validator': {'path': str(Path(__file__).resolve()),
                      'sha256': sha(Path(__file__).read_bytes())},
        'counts': {'train': len(train_keys), 'test': len(test_keys), 'excluded': len(excluded_keys),
                   'unique_embeddings_checked': len(keys)},
        'partition_and_row_order_preserved': not any(
            x['check'] == 'frozen_partition_and_row_order_unchanged' for x in audit.errors),
        'cache_signature': signature,
        'unique_prompt_token_count_range': [min(token_counts), max(token_counts)],
        'media_leakage': leakage,
        'shared_text_cache_between_splits_is_allowed': True,
        'shared_text_cache_reason': 'Deterministic inference through a frozen pretrained encoder; no fitting on held-out episodes.',
        'previous_validation_backup': str(backup / 'validation.json'),
        'scope': 'All refreshed rows, frozen backups, dedup file hashes, all unique cache tensors, local tokenizer IDs/masks, configuration and artifact hashes.',
        'limitations': ['No media decoding or video/image rehashing.',
                       'The 11 GB encoder checkpoint was not rehashed or loaded; its generator-recorded SHA is compared with the known official SHA.',
                       'This validates cached tensor integrity and provenance, not an independent rerun of the encoder or a trained model.'],
        'inputs': audit.inputs, 'checks_performed': dict(audit.checks), 'issues': audit.errors,
        'source_media_or_training_files_modified': False,
    }
    if args.write_report:
        atomic_report(args.output.resolve(), report)
    for stream in locks:
        stream.close()
    return report


def main():
    split = PROJECT / 'data/robotwin_observer_ego_obj_splits_v1'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-root', type=Path, default=split)
    parser.add_argument('--backup', type=Path,
                        default=split / 'revisions/before_chronological_128_20261019')
    parser.add_argument('--cache', type=Path,
                        default=PROJECT / 'data/text_embeddings/prompt_obj_chronological_128_v1_20261019')
    parser.add_argument('--prompt-source', type=Path,
                        default=PROJECT / 'runs/prompt_obj_chronological_128_v1/exports/paired_training.jsonl')
    parser.add_argument('--data-contract', type=Path, default=PROJECT / 'configs/obj_video_data_contract.json')
    parser.add_argument('--expected-unique-prompts', type=int, default=1382)
    parser.add_argument('--output', type=Path, default=split / 'validation.json')
    parser.add_argument('--write-report', action='store_true')
    args = parser.parse_args()
    report = validate(args)
    print(json.dumps({key: report[key] for key in
                      ('status', 'counts', 'cache_signature', 'unique_prompt_token_count_range', 'issues')},
                     ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

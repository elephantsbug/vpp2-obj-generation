#!/usr/bin/env python3
"""Refresh prompt/cache fields while preserving frozen episode assignments."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
SPLIT = PROJECT / 'data/robotwin_observer_ego_obj_splits_v1'
REVISION = 'prompt_obj_chronological_128_v1'
PROMPTS = PROJECT / 'runs' / REVISION / 'exports/paired_training.jsonl'
CACHE = PROJECT / 'data/text_embeddings/prompt_obj_chronological_128_v1_20261019'
CONTRACT = PROJECT / 'configs/obj_video_data_contract.json'
BACKUP = SPLIT / 'revisions/before_chronological_128_20261019'
PROMPT_FIELDS = {'prompt', 'prompt_records_path', 'prompt_records_sha256',
                 'text_embedding_cache_must_be_rebuilt'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def key(row):
    return row['task'], row['episode']


def rows(data):
    return [json.loads(line) for line in data.splitlines() if line]


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def jsonl_bytes(value):
    return ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in value).encode()


def write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name, delete=False) as f:
        temporary = Path(f.name)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run(commit=False):
    tracked = {name: SPLIT / name for name in ['train.jsonl', 'test.jsonl', 'excluded_prompt_review.jsonl', 'summary.json', 'validation.json']}
    tracked.update(data_contract=CONTRACT, prompt_source=PROMPTS)
    before = {name: path.read_bytes() for name, path in tracked.items()}
    new_rows = rows(before['prompt_source'])
    new = {key(row): row for row in new_rows}
    require(len(new) == len(new_rows) == 2395, 'Expected 2395 unique current prompts')
    old = {name: rows(before[name + '.jsonl']) for name in ['train', 'test']}
    excluded = {key(row) for row in rows(before['excluded_prompt_review.jsonl'])}
    keys = {name: [key(row) for row in rs] for name, rs in old.items()}
    require(len(keys['train']) == len(set(keys['train'])) == 2155, 'Unexpected frozen train partition')
    require(len(keys['test']) == len(set(keys['test'])) == 240, 'Unexpected frozen test partition')
    require(not set(keys['train']) & set(keys['test']), 'Train/test overlap')
    require(set(keys['train']) | set(keys['test']) == set(new), 'New prompt keys differ from frozen eligible episodes')
    require(len(excluded) == 105 and not excluded & set(new), 'Excluded episodes mixed into eligible prompts')
    cache_status, cache_index = {}, {}
    if commit:
        cache_status = json.loads((CACHE / 'status.json').read_text())
        require(cache_status.get('status') == 'complete', 'Embedding cache must be complete before updating active manifests')
        for entry in rows((CACHE / 'index.jsonl').read_bytes()):
            require(entry['prompt_sha256'] not in cache_index, 'Duplicate prompt cache key')
            cache_index[entry['prompt_sha256']] = entry
        require(set(cache_index) == {sha(row['prompt'].encode()) for row in new_rows}, 'Cache does not exactly cover current unique prompts')
        require(cache_status.get('cache_signature'), 'Missing cache signature')
    updated = {}
    for split, old_rows in old.items():
        output = []
        for original in old_rows:
            current = new[key(original)]
            require(not original['prompt_review_flags'] and not current['prompt_review_flags'], 'Flagged prompt found')
            for field in original.keys() & current.keys() - PROMPT_FIELDS:
                require(original[field] == current[field], f'Unexpected non-prompt change: {key(original)} / {field}')
            row = dict(original)
            row.update(current)
            prompt_hash = sha(row['prompt'].encode())
            path = CACHE / 'embeddings' / (prompt_hash + '.pt')
            row.update(text_embedding_path=str(path), text_embedding_prompt_sha256=prompt_hash,
                       text_embedding_context_length=128)
            if commit:
                entry = cache_index[prompt_hash]
                require(entry['prompt'] == row['prompt'], 'Cache prompt collision/mismatch')
                require(Path(entry['path']) == path and path.is_file(), 'Cache path mismatch/missing')
                require(entry['valid_token_count'] == row['prompt_token_count'], 'Cached token count mismatch')
                row.update(text_embedding_sha256=entry['sha256'],
                           text_embedding_cache_signature=cache_status['cache_signature'],
                           text_embedding_valid_tokens=entry['valid_token_count'],
                           text_embedding_cache_must_be_rebuilt=False)
            for field in ['dedup_index_path', 'dedup_index_sha256', 'source_frame_count',
                          'deduplicated_frame_count', 'sampling_index_space']:
                require(row[field] == original[field], f'Dedup field changed: {field}')
            output.append(row)
        require([key(row) for row in output] == keys[split], 'Episode assignment/order changed')
        updated[split] = output
    result = {'train': len(updated['train']), 'test': len(updated['test']), 'excluded': len(excluded),
              'unique_prompts': len({row['prompt'] for row in new_rows}),
              'revision': REVISION, 'partition_and_order_preserved': True, 'commit': commit}
    if not commit:
        return result
    require(not BACKUP.exists(), f'Backup exists; inspect previous refresh before rerunning: {BACKUP}')
    for name, path in tracked.items():
        require(path.read_bytes() == before[name], f'Source changed during refresh: {name}')
    BACKUP.mkdir(parents=True)
    for name, data in before.items():
        if name != 'prompt_source':
            filename = 'obj_video_data_contract.json' if name == 'data_contract' else name
            (BACKUP / filename).write_bytes(data)
    (BACKUP / 'backup_manifest.json').write_bytes(json_bytes({name: {'source': str(tracked[name]), 'sha256': sha(data)}
                                                           for name, data in before.items()}))
    now = datetime.now(timezone.utc).isoformat()
    payloads = {name + '.jsonl': jsonl_bytes(rs) for name, rs in updated.items()}
    summary = json.loads(before['summary.json'])
    summary.update(prompt_revision=REVISION, prompt_updated_at=now,
                   prompt_source={'path': str(PROMPTS), 'sha256': sha(before['prompt_source'])},
                   previous_revision_backup=str(BACKUP),
                   text_embedding_cache={'root': str(CACHE), 'cache_signature': cache_status['cache_signature'],
                                         'status': 'complete', 'unique_prompts': result['unique_prompts']},
                   prompt_refresh_builder={'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__).read_bytes())},
                   reproduction_note='Original builder reproduces the frozen partition and original prompt snapshot; this refresh preserves that partition and updates text/cache fields.')
    for name, data in payloads.items():
        summary['artifacts'][name]['sha256'] = sha(data)
    summary['checks'].update(frozen_episode_order_preserved=True, prompt_revision_matches=True,
                             text_embedding_references_complete=True)
    contract = json.loads(before['data_contract'])
    contract.update(current_training_manifest=str(SPLIT / 'train.jsonl'),
                    current_test_manifest=str(SPLIT / 'test.jsonl'), prompt_source_manifest=str(PROMPTS))
    contract['dataset_split']['status'] = 'current_prompts_and_embeddings_ready; training loader not yet connected'
    contract['text_embeddings'] = {'status': 'complete', 'root': str(CACHE), 'index': str(CACHE / 'index.jsonl'),
                                    'cache_signature': cache_status['cache_signature'], 'context_length': 128,
                                    'context_width': 4096, 'context_dtype': 'bfloat16', 'mask_dtype': 'bool',
                                    'prompt_revision': REVISION, 'unique_prompts': result['unique_prompts'],
                                    'encoding': 'Frozen pretrained UMT5 encoder, eval/inference mode; real padding mask preserved.'}
    report = dict(result, completed_at=now, backup=str(BACKUP), cache_signature=cache_status['cache_signature'],
                  artifacts={name: sha(data) for name, data in payloads.items()}, training_started=False)
    validation = {'status': 'pending_independent_validation', 'prompt_revision': REVISION,
                  'updated_at': now, 'previous_report': str(BACKUP / 'validation.json'),
                  'note': 'The previous validation covers the original prompts. Validate the refreshed manifests and embedding references before training.'}
    replacements = {SPLIT / name: data for name, data in payloads.items()}
    replacements.update({SPLIT / 'summary.json': json_bytes(summary), CONTRACT: json_bytes(contract),
                         SPLIT / 'validation.json': json_bytes(validation),
                         SPLIT / 'prompt_refresh_20261019.json': json_bytes(report)})
    rollback = {path: path.read_bytes() if path.exists() else None for path in replacements}
    try:
        for path, data in replacements.items():
            write_atomic(path, data)
    except BaseException as error:
        failures = {}
        for path, data in rollback.items():
            try:
                if data is None:
                    path.unlink(missing_ok=True)
                else:
                    write_atomic(path, data)
            except BaseException as restore_error:
                failures[str(path)] = repr(restore_error)
        write_atomic(BACKUP / 'refresh_failure.json', json_bytes({
            'error': repr(error), 'rollback_errors': failures,
            'active_files_restored': not failures}))
        raise
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', action='store_true', help='Require completed cache and replace train/test with backed-up revisions.')
    args = parser.parse_args()
    with (SPLIT / '.prompt_refresh.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(run(args.commit), ensure_ascii=False, indent=2))

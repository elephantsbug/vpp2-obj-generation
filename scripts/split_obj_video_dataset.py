#!/usr/bin/env python3
"""Exclude flagged prompts and split paired RGB/object episodes by task.

Only manifests and split metadata are written. Original media is unchanged.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import tempfile


PROJECT = Path(__file__).resolve().parents[1]
ALGORITHM = 'task_stratified_episode_sha256_rank_largest_remainder_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def key(row):
    return row['task'], row['episode']


def unique_map(rows, label):
    result = {key(row): row for row in rows}
    require(len(result) == len(rows), f'Duplicate episode key in {label}')
    return result


def rank(seed, purpose, *parts):
    text = json.dumps([seed, purpose, *parts], ensure_ascii=False, separators=(',', ':'))
    return digest(text.encode())


def serialize(rows):
    return ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n'
                   for row in sorted(rows, key=key)).encode()


def build(args):
    output = args.output_root.resolve()
    if not args.dry_run and output.exists():
        raise FileExistsError(f'Choose a new split directory; refusing to overwrite {output}')
    fraction = Decimal(args.test_fraction)
    require(fraction.is_finite() and 0 < fraction < 1, 'test-fraction must be between 0 and 1')
    sources = {
        'paired_all': args.paired_root / 'paired_all.jsonl',
        'paired_no_flags': args.paired_root / 'paired_no_flags.jsonl',
        'paired_needs_review': args.paired_root / 'paired_needs_review.jsonl',
        'dedup_manifest': args.dedup_root / 'manifest.jsonl',
    }
    source_bytes = {name: path.read_bytes() for name, path in sources.items()}
    parsed = {name: [json.loads(line) for line in data.splitlines() if line]
              for name, data in source_bytes.items()}
    maps = {name: unique_map(rows, name) for name, rows in parsed.items()}
    all_rows = maps['paired_all']
    clean = {k: row for k, row in all_rows.items() if not row['prompt_review_flags']}
    excluded = {k: row for k, row in all_rows.items() if row['prompt_review_flags']}
    require(clean == maps['paired_no_flags'], 'Clean manifest differs from current flags in paired_all')
    require(excluded == maps['paired_needs_review'], 'Excluded manifest differs from current flags in paired_all')
    require(set(all_rows) == set(maps['dedup_manifest']), 'Paired/dedup episode sets differ')
    grouped = defaultdict(list)
    for row in clean.values():
        grouped[row['task']].append(row)
    require(grouped, 'No eligible episodes')
    require(all(len(rows) >= 2 for rows in grouped.values()), 'Each task needs at least two eligible episodes')
    test_total = int((Decimal(len(clean)) * fraction).to_integral_value(rounding=ROUND_HALF_UP))
    require(len(grouped) <= test_total <= len(clean) - len(grouped),
            'Requested fraction cannot leave every task in both splits')
    quotas = {task: Decimal(len(rows)) * fraction for task, rows in grouped.items()}
    test_counts = {task: int(q.to_integral_value(rounding=ROUND_FLOOR)) for task, q in quotas.items()}
    allocation_order = sorted(grouped, key=lambda task: (-(quotas[task] - test_counts[task]),
                                                        rank(args.seed, 'task_quota_tie', task)))
    for task in allocation_order[:test_total - sum(test_counts.values())]:
        test_counts[task] += 1
    require(all(0 < test_counts[task] < len(rows) for task, rows in grouped.items()),
            'A task has an empty train/test split; choose another fraction')
    train, test, task_counts = [], [], {}
    for task in sorted(grouped):
        ordered = sorted(grouped[task], key=lambda row: (rank(args.seed, 'episode', *key(row)), key(row)))
        count = test_counts[task]
        test.extend(ordered[:count])
        train.extend(ordered[count:])
        task_counts[task] = {'eligible': len(ordered), 'train': len(ordered) - count,
                             'test': count, 'excluded_prompt_flags': sum(k[0] == task for k in excluded)}
    train_keys, test_keys = {key(r) for r in train}, {key(r) for r in test}
    require(not train_keys & test_keys, 'Episode leakage')
    require(train_keys | test_keys == set(clean), 'Eligible coverage mismatch')
    require(not (train_keys | test_keys) & set(excluded), 'Flagged episode included')
    require(len(test) == test_total, 'Global test count mismatch')
    for domain in ['rgb', 'obj']:
        train_sha = {r[f'{domain}_sha256']['video'] for r in train}
        test_sha = {r[f'{domain}_sha256']['video'] for r in test}
        require(not train_sha & test_sha, f'Identical {domain} video in train and test')
    for field in ['condition_video_path', 'target_video_path']:
        require(not {Path(r[field]).resolve() for r in train} & {Path(r[field]).resolve() for r in test},
                f'Video path shared across splits: {field}')

    def attach_dedup(row):
        entry = maps['dedup_manifest'][key(row)]
        index_path = (args.dedup_root / entry['index_path']).resolve()
        require(index_path.is_file(), f'Missing dedup index: {index_path}')
        require(digest(index_path.read_bytes()) == entry['index_sha256'], f'Dedup hash changed: {index_path}')
        require(entry['condition_video_path'] == row['condition_video_path'], 'Dedup RGB path mismatch')
        require(entry['source_frame_count'] == row['video_metadata']['frame_count'], 'Dedup frame count mismatch')
        require(entry['source_timestamps_sha256'] == row['obj_sha256']['timestamps'], 'Dedup timestamp mismatch')
        require(entry['source_timestamps_sha256'] == row['rgb_sha256']['timestamps'], 'RGB timestamp mismatch')
        require(entry['policy'] == 'equal_timestamp_keep_first_v1', 'Unexpected dedup policy')
        for field in ['condition_video_path', 'condition_episode_first_frame_path',
                      'target_video_path', 'target_first_frame_path', 'timestamps_path', 'record_path']:
            require(Path(row[field]).is_file(), f'Missing media/record: {row[field]}')
        # Keep all original paired fields; source media metadata retains original frame counts.
        return dict(row, dedup_index_path=str(index_path), dedup_index_sha256=entry['index_sha256'],
                    source_frame_count=entry['source_frame_count'],
                    deduplicated_frame_count=entry['frame_count'],
                    sampling_index_space='deduplicated_frame_order')

    train, test = [attach_dedup(r) for r in train], [attach_dedup(r) for r in test]
    artifacts = {'train.jsonl': serialize(train), 'test.jsonl': serialize(test),
                 'excluded_prompt_review.jsonl': serialize(list(excluded.values()))}
    summary = {
        'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
        'algorithm': ALGORITHM, 'seed': args.seed, 'requested_test_fraction': str(fraction),
        'rounding': 'Global test count rounds half up; per-task counts use floor plus largest remainders, seeded tie order.',
        'assignment': 'Within each task sort by SHA256 of JSON [seed, episode, task, episode_id]; first quota entries are test.',
        'split_unit': ['task', 'episode'], 'stratification': 'task',
        'evaluation_scope': 'Held-out episodes from the same tasks; not unseen-task generalization.',
        'counts': {'source': len(all_rows), 'excluded_prompt_review': len(excluded),
                   'eligible': len(clean), 'train': len(train), 'test': len(test), 'tasks': len(grouped)},
        'per_task': task_counts,
        'frames': {name: {'original': sum(r['source_frame_count'] for r in rows),
                         'deduplicated': sum(r['deduplicated_frame_count'] for r in rows)}
                   for name, rows in [('train', train), ('test', test)]},
        'sources': {name: {'path': str(sources[name].resolve()), 'sha256': digest(data)}
                    for name, data in source_bytes.items()},
        'builder': {'path': str(Path(__file__).resolve()), 'sha256': digest(Path(__file__).read_bytes())},
        'artifacts': {name: {'path': str(output / name), 'sha256': digest(data)}
                      for name, data in artifacts.items()},
        'checks': {'episode_overlap': 0, 'flagged_episodes_in_train_or_test': 0,
                   'eligible_episodes_missing': 0, 'shared_rgb_video_hashes': 0,
                   'shared_obj_video_hashes': 0, 'shared_video_paths': 0,
                   'dedup_indexes_hash_verified': len(clean), 'every_task_in_both_splits': True},
        'usage': {'train': 'Only train.jsonl may provide optimization samples or fitted dataset statistics.',
                  'test': 'Keep test.jsonl held out from optimization; all windows of each test episode remain test.',
                  'excluded': 'Excluded from both splits; original media and prompts are retained.',
                  'sampling': 'Sample in deduplicated order, then map through dedup_index_path for both RGB and obj.',
                  'dedup_source_frame_indices': 'NPZ source_frame_indices refer to unchanged original MP4 frame order.'},
        'source_media_modified': False, 'training_started': False,
    }
    for name, path in sources.items():
        require(path.read_bytes() == source_bytes[name], f'Source manifest changed during split: {name}')
    if not args.dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f'.{output.name}.building-', dir=output.parent) as tmp:
            stage = Path(tmp) / 'result'
            stage.mkdir()
            for name, data in artifacts.items():
                (stage / name).write_bytes(data)
            (stage / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
            require(not output.exists(), 'Output appeared during split; refusing overwrite')
            stage.rename(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired-root', type=Path, default=PROJECT / 'data/robotwin_observer_ego_obj_full_vpp2')
    parser.add_argument('--dedup-root', type=Path, default=PROJECT / 'data/robotwin_observer_ego_dedup_index_v1')
    parser.add_argument('--output-root', type=Path, default=PROJECT / 'data/robotwin_observer_ego_obj_splits_v1')
    parser.add_argument('--test-fraction', default='0.10')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    summary = build(args)
    print(json.dumps({k: summary[k] for k in ['algorithm', 'seed', 'counts', 'checks', 'artifacts']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

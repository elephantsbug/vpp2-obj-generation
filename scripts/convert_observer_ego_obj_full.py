#!/usr/bin/env python3
"""Convert observer/ego video_object_full targets, paired with existing RGB conditions."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import av
from PIL import Image

import convert_observer as common
import convert_observer_ego as pair

PROJECT = Path(__file__).resolve().parents[1]
CONTENT = {'source_video_kind': 'video_object_full', 'role': 'object_video_target',
           'condition_video_kind': 'video', 'background': 'preserved_from_source',
           'geometry': pair.LAYOUT}


def source_files(rgb_observer, rgb_root):
    base = rgb_observer.parents[2]
    task, episode = rgb_observer.parents[3].name, rgb_observer.stem
    obj_o = base / 'observer/video_object_full' / rgb_observer.name
    obj_e = base / 'ego/video_object_full' / rgb_observer.name
    rgb = common.output_paths(rgb_root, task, episode)
    return {'observer_video': obj_o, 'ego_video': obj_e,
            'observer_timestamps': obj_o.with_suffix('.csv'),
            'ego_timestamps': obj_e.with_suffix('.csv'),
            'instruction': base / 'instruction' / (episode + '.json'),
            **{'rgb_' + key: path for key, path in rgb.items()}}


def fingerprint(files, config):
    provenance = {'inputs': {k: {**common.file_info(p), 'sha256': common.digest(p)}
                             for k, p in files.items()}, 'config': config}
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    return signature, provenance


def check_rgb_reference(files, provenance, instruction, timestamps):
    record = json.loads(files['rgb_record'].read_text())
    if record['geometry'] != pair.LAYOUT or record['instruction'] != instruction:
        raise ValueError('RGB reference geometry/instruction differs')
    for key in ('video', 'image', 'timestamps'):
        if provenance['inputs']['rgb_' + key]['sha256'] != record['output_sha256'][key]:
            raise ValueError('RGB reference output hash changed: ' + key)
    for key in ('observer_video', 'ego_video', 'observer_timestamps', 'ego_timestamps', 'instruction'):
        original = record['inputs'][key]
        path = Path(original['path'])
        if path.parent.name == 'video_object_full':
            raise ValueError('Condition reference must be full-scene RGB, not obj')
        current = common.file_info(path)
        if any(current[k] != original[k] for k in ('path', 'size_bytes', 'mtime_ns')):
            raise ValueError('Original RGB source changed: ' + key)
        if 'sha256' in original and common.digest(path) != original['sha256']:
            raise ValueError('Original RGB annotation hash changed: ' + key)
    _, rgb_times = common.read_annotations({'instruction': files['instruction'],
                                          'timestamps': files['rgb_timestamps']})
    if timestamps != rgb_times:
        raise ValueError('OBJ and RGB simulation timestamps/frame counts differ')
    with Image.open(files['rgb_image']) as image:
        if image.size != (448, 224) or image.mode != 'RGB':
            raise ValueError('RGB condition first-frame geometry differs')
    return record


def run_job(rgb_observer, output_root, rgb_root, config, overwrite=False):
    files = source_files(rgb_observer, rgb_root)
    signature, provenance = fingerprint(files, config)
    task, episode = rgb_observer.parents[3].name, rgb_observer.stem
    paths = common.output_paths(output_root, task, episode)
    instruction, timestamps = common.read_annotations({
        'instruction': files['instruction'], 'timestamps': files['observer_timestamps']})
    _, ego_times = common.read_annotations({
        'instruction': files['instruction'], 'timestamps': files['ego_timestamps']})
    if timestamps != ego_times:
        raise ValueError('Observer and ego OBJ timestamps/frame counts differ')
    rgb = check_rgb_reference(files, provenance, instruction, timestamps)
    if (rgb['task'], rgb['episode']) != (task, episode):
        raise ValueError('RGB reference task/episode differs')
    if not overwrite:
        cached = pair.cached_record(paths, signature)
        if cached is not None and cached.get('content') == CONTENT:
            return cached, 'skipped'
    temporary = {}
    try:
        token = uuid.uuid4().hex
        for key, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary[key] = path.with_name(f'.{path.name}.{token}.tmp')
        sources = pair.encode_pair(files, temporary['video'], temporary['image'])
        count = sources['observer']['frame_count']
        if count != len(timestamps):
            raise ValueError(f'OBJ videos have {count} frames, timestamps have {len(timestamps)}')
        for view in ('observer', 'ego'):
            if sources[view] != rgb['source_videos'][view]:
                raise ValueError(f'OBJ/RGB source geometry, fps or frame count differs: {view}')
        output = common.probe_video(temporary['video'])
        expected = {'width': 448, 'height': 224, 'fps': sources['observer']['fps'],
                    'frame_count': count}
        if output != expected or output != rgb['output_video']:
            raise ValueError('Encoded OBJ video does not match RGB dimensions/fps/frame count')
        shutil.copyfile(files['observer_timestamps'], temporary['timestamps'])
        if fingerprint(files, config)[0] != signature:
            raise ValueError('Source or RGB reference changed during conversion')
        record = {
            'schema_version': 3, 'task': task, 'episode': episode,
            'fingerprint': signature, **provenance, 'content': CONTENT,
            'outputs': {k: str(p.relative_to(output_root)) for k, p in paths.items()},
            'source_videos': sources, 'output_video': output, 'geometry': pair.LAYOUT,
            'view_order': ['observer', 'ego'], 'view_size_wh': [224, 224],
            'first_frame_index': 0, 'instruction': instruction,
            'instruction_key': 'seen', 'instruction_index': 0,
            'rgb_reference': {'root': str(rgb_root), 'record_path': str(files['rgb_record']),
                              'output_sha256': rgb['output_sha256'], 'output_video': rgb['output_video']},
            'timing': {'resampled': False, 'encoded_cadence_preserved': True,
                       'both_camera_relative_pts_verified': True,
                       'both_camera_csv_timestamps_equal': True,
                       'rgb_obj_csv_timestamps_equal': True,
                       'rgb_obj_frame_count_and_fps_equal': True,
                       'simulation_timestamps_copied_verbatim': True,
                       'timestamp_rows': count, 'first_timestamp_s': timestamps[0],
                       'last_timestamp_s': timestamps[-1]},
            'output_sha256': {k: common.digest(temporary[k]) for k in ('video', 'image', 'timestamps')},
        }
        temporary['record'].write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        for key in ('video', 'image', 'timestamps', 'record'):
            os.replace(temporary[key], paths[key])
        return record, 'converted'
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def guard_output(root, rgb_root, source_root):
    for protected in (rgb_root, source_root):
        if root == protected or root.is_relative_to(protected) or protected.is_relative_to(root):
            raise ValueError('OBJ output must be separate from RGB and source directories')
    root.mkdir(parents=True, exist_ok=True)
    marker = root / '.content.json'
    expected = {**CONTENT, 'rgb_reference_root': str(rgb_root)}
    if marker.exists():
        if json.loads(marker.read_text()) != expected:
            raise ValueError('Output directory belongs to a different content/reference')
    elif any(p.name != '.convert.lock' for p in root.iterdir()):
        raise ValueError('Refusing nonempty output directory without OBJ content marker')
    else:
        common.atomic_text(marker, json.dumps(expected, indent=2) + '\n')


def paired_rows(records, output_root, rgb_root, prompt_records):
    prompts = {}
    for line in prompt_records.read_text().splitlines():
        p = json.loads(line); key = (p['task'], p['episode'])
        if key in prompts:
            raise ValueError('Duplicate prompt episode: ' + str(key))
        prompts[key] = p
    rows = []
    prompt_hash = common.digest(prompt_records)
    for r in records:
        p = prompts[(r['task'], r['episode'])]
        if ' '.join(p['original_instruction'].split()) != r['instruction']:
            raise ValueError('Prompt source instruction differs')
        rgb = common.output_paths(rgb_root, r['task'], r['episode'])
        if p['video_sha256'] != r['rgb_reference']['output_sha256']['video'] or p['image_sha256'] != r['rgb_reference']['output_sha256']['image']:
            raise ValueError('Prompt was reviewed against different RGB media')
        obj = common.output_paths(output_root, r['task'], r['episode'])
        rows.append({'task': r['task'], 'episode': r['episode'],
                     'condition_video_path': str(rgb['video']),
                     'condition_episode_first_frame_path': str(rgb['image']),
                     'target_video_path': str(obj['video']),
                     'target_first_frame_path': str(obj['image']),
                     'timestamps_path': str(obj['timestamps']),
                     'condition_source_kind': 'video', 'target_source_kind': 'video_object_full',
                     'prompt': p['enhanced_instruction'], 'prompt_review_flags': p['review_flags'],
                     'prompt_records_path': str(prompt_records), 'prompt_records_sha256': prompt_hash,
                     'rgb_sha256': r['rgb_reference']['output_sha256'],
                     'obj_sha256': r['output_sha256'], 'video_metadata': r['output_video'],
                     'view_order': r['view_order'], 'record_path': str(obj['record'])})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=Path, default=Path('/data2/yaozhongbo/robotwin_clean_50*50/demo_clean'))
    ap.add_argument('--rgb-root', type=Path, default=PROJECT / 'data/robotwin_observer_ego_vpp2')
    ap.add_argument('--output-root', type=Path, default=PROJECT / 'data/robotwin_observer_ego_obj_full_vpp2')
    ap.add_argument('--vpp2-root', type=Path, default=PROJECT / 'third_party/vpp2')
    ap.add_argument('--prompt-records', type=Path, default=PROJECT / 'runs/prompt_all_video_agent_v1/exports/records.jsonl')
    ap.add_argument('--workers', type=common.positive_int, default=16)
    ap.add_argument('--task', action='append')
    ap.add_argument('--limit', type=common.positive_int)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    source_root, videos = common.discover(args.source_root, args.task, args.limit)
    rgb_root, output_root = args.rgb_root.resolve(), args.output_root.resolve()
    scope = {'source_root': str(source_root), 'rgb_root': str(rgb_root), 'output_root': str(output_root),
             'selected_episodes': len(videos), 'tasks': sorted({v.parents[3].name for v in videos}),
             'manifest_scope': 'successful episodes selected by this invocation', 'content': CONTENT}
    missing = [str(p) for v in videos for p in source_files(v, rgb_root).values() if not p.is_file()]
    if not args.prompt_records.is_file(): missing.append(str(args.prompt_records))
    if args.dry_run:
        print(json.dumps({**scope, 'missing_count': len(missing), 'missing_examples': missing[:10]}, indent=2))
        return int(bool(missing))
    if missing: raise ValueError(f'Missing {len(missing)} inputs; first: {missing[0]}')
    for executable in ('ffmpeg', 'ffprobe'):
        if not shutil.which(executable): raise ValueError('Missing executable: ' + executable)
    pair.init_worker(args.vpp2_root)
    config = {'script_sha256': common.digest(__file__), 'pair_helper_sha256': common.digest(pair.__file__),
              'common_helper_sha256': common.digest(common.__file__),
              'official_helper_sha256': common.digest(args.vpp2_root / 'src/vpp2/libero/eval_helpers.py'),
              'composition': pair.LAYOUT, 'encoding': common.ENCODING, 'source_video_kind': 'video_object_full',
              'frame_selection': 'all', 'av_version': av.__version__}
    # Reject protected destinations before creating a lock or any marker there.
    for protected in (rgb_root, source_root):
        if output_root == protected or output_root.is_relative_to(protected) or protected.is_relative_to(output_root):
            raise ValueError('OBJ output must be separate from RGB and source directories')
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / '.convert.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard_output(output_root, rgb_root, source_root)
        counts = {'converted': 0, 'skipped': 0, 'failed': 0}; records = []; failures = []
        start = time.monotonic()
        with ProcessPoolExecutor(max_workers=min(args.workers, len(videos)),
                                 mp_context=multiprocessing.get_context('spawn'),
                                 initializer=pair.init_worker, initargs=(str(args.vpp2_root),)) as executor:
            futures = {executor.submit(run_job, v, output_root, rgb_root, config, args.overwrite): v for v in videos}
            for done, future in enumerate(as_completed(futures), 1):
                video = futures[future]
                try:
                    record, status = future.result(); records.append(record)
                except Exception as error:
                    status = 'failed'; failures.append({'source': str(video), 'error': str(error)})
                    print('ERROR ' + str(video) + ': ' + str(error), file=sys.stderr, flush=True)
                counts[status] += 1
                print(f'[{done}/{len(videos)}] {status}: {video.parents[3].name}/{video.stem}', flush=True)
        records.sort(key=lambda r: (r['task'], r['episode']))
        common.atomic_text(output_root / 'manifest.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
        pairs = paired_rows(records, output_root, rgb_root, args.prompt_records.resolve())
        for name, rows in [('paired_all.jsonl', pairs),
                           ('paired_no_flags.jsonl', [r for r in pairs if not r['prompt_review_flags']]),
                           ('paired_needs_review.jsonl', [r for r in pairs if r['prompt_review_flags']])]:
            common.atomic_text(output_root / name, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        summary = {**scope, **counts, 'elapsed_seconds': round(time.monotonic() - start, 3),
                   'total_frames': sum(r['output_video']['frame_count'] for r in records),
                   'paired_episodes': len(pairs), 'without_prompt_flags': sum(not r['prompt_review_flags'] for r in pairs),
                   'with_prompt_flags': sum(bool(r['prompt_review_flags']) for r in pairs),
                   'failures': failures, 'training_started': False, 'training_window_sampler_implemented': False,
                   'config': config}
        common.atomic_text(output_root / 'summary.json', json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())

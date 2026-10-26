#!/usr/bin/env python3
"""Resumable two-step prompt generation, with explicit --all authorization.

Uses only physical GPUs 4,5 and 6,7. Source images and tasks remain unchanged.
Check-free captions are a screened subset, not a claim of semantic correctness.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import signal
import sys
import time
import traceback

from enhance_prompts import atomic_text, check_caption, record_path, select_rows, sha256, write_json
from enhance_prompts_two_step import data_image, worker
from prompt_batched_worker import worker as batched_worker

PROJECT = Path(__file__).resolve().parents[1]
GPU_GROUPS = ['4,5', '6,7']
SNAPSHOT_FILES = ['enhance_prompts_two_step_batch.py', 'enhance_prompts_two_step.py',
                  'prompt_plan.py', 'enhance_prompts.py', 'prompt_batched_worker.py']


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_lines(values):
    return ''.join(json.dumps(value, ensure_ascii=False) + '\n' for value in values)


def choose_rows(dataset, selection, mode, tasks=None):
    """Validate inputs with the existing loader and select reproducibly."""
    rows = select_rows(dataset, selection, mode != 'preview')
    for row in rows:
        for name in ('task', 'episode'):
            value = row[name]
            if not isinstance(value, str) or not value or value in {'.', '..'} or Path(value).name != value:
                raise ValueError(f'Unsafe {name}: {value!r}')
    if mode in {'all', 'review_100', 'tasks'}:
        counts = Counter(row['task'] for row in rows)
        if len(rows) != 2500 or len(counts) != 50 or set(counts.values()) != {50}:
            raise ValueError(f'Expected 50 tasks x 50 episodes = 2500; found {len(rows)} episodes / {len(counts)} tasks')
        rows.sort(key=lambda row: (row['task'], row['episode']))
    if mode == 'tasks':
        if not tasks:
            raise ValueError('Task subset mode requires at least one task')
        if len(set(tasks)) != len(tasks):
            raise ValueError('Duplicate task selection')
        unknown = sorted(set(tasks) - set(counts))
        if unknown:
            raise ValueError('Unknown task selection: ' + ', '.join(unknown))
        requested = set(tasks)
        rows = [row for row in rows if row['task'] in requested]
    if mode == 'review_100':
        by_task = defaultdict(list)
        for row in rows:
            by_task[row['task']].append(row)
        rows = [row for task_rows in by_task.values() for row in (task_rows[0], task_rows[-1])]
    return rows


def validate_record(record, row, config, output=None):
    expected = {'task': row['task'], 'episode': row['episode'], 'image': row['outputs']['image'],
                'image_sha256': row['output_sha256']['image'], 'original_instruction': row['instruction'],
                'instruction_source': row['inputs']['instruction'], 'instruction_key': row['instruction_key'],
                'instruction_index': row['instruction_index'], 'config_fingerprint': config['fingerprint']}
    for name, value in expected.items():
        if record.get(name) != value:
            raise ValueError(f'Record provenance mismatch: {row["task"]}/{row["episode"]}: {name}')
    if not isinstance(record.get('review_flags'), list) or not all(isinstance(flag, str) for flag in record['review_flags']):
        raise ValueError('Record has invalid review_flags')
    caption = record.get('enhanced_instruction')
    if not isinstance(caption, str) or '\n' in caption or '\r' in caption or '@@' in caption:
        raise ValueError('Record contains an invalid caption line')
    if not caption and not record['review_flags']:
        raise ValueError('Empty caption must have diagnostic review_flags')
    if caption:
        _, errors = check_caption(caption)
        if any(error not in record['review_flags'] for error in errors):
            raise ValueError('Unrecorded caption format failures')
    if 'elapsed_seconds' not in record or not isinstance(record.get('plan_attempts'), list):
        raise ValueError('Record is not terminal')
    if record.get('physical_gpus') not in GPU_GROUPS:
        raise ValueError('Unexpected GPU group in record')
    if output is not None:
        for view in ('observer', 'ego'):
            relative = record.get(f'{view}_crop')
            if relative:
                crop = output / relative
                if not crop.resolve().is_relative_to(output.resolve()):
                    raise ValueError('Crop outside output directory')
                if not crop.is_file() or sha256(crop) != record.get(f'{view}_crop_sha256'):
                    raise ValueError(f'Missing or changed {view} crop')


def collect_records(output, rows, config, verify_crops=False):
    results, pending = [], []
    expected_paths = {record_path(output, row) for row in rows}
    extra = set((output / 'records').glob('*/*.json')) - expected_paths
    if extra:
        raise ValueError(f'Unexpected records outside selection: {sorted(str(path) for path in extra)[:3]}')
    for row in rows:
        path = record_path(output, row)
        if path.exists():
            record = json.loads(path.read_text())
            validate_record(record, row, config, output if verify_crops else None)
            results.append(record)
        else:
            pending.append(row)
    return results, pending


def ensure_image_link(output, dataset):
    target = dataset / 'images'
    link = output / 'images'
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise ValueError('Existing images symlink points to a different dataset')
    elif link.exists():
        raise ValueError('Unexpected existing images path in output')
    else:
        link.symlink_to(target, target_is_directory=True)


def summarize(rows, records, config, state, started_at, failure=None):
    generated = [record for record in records if record['enhanced_instruction']]
    screened = [record for record in generated if not record['review_flags']]
    needs_review = [record for record in records if record['review_flags'] or not record['enhanced_instruction']]
    counts = {task: {'selected': count, 'terminal_records': 0, 'caption_generated': 0,
                    'automatic_checks_passed': 0, 'empty_caption': 0, 'needs_review_by_checks': 0}
              for task, count in sorted(Counter(row['task'] for row in rows).items())}
    for record in records:
        entry = counts[record['task']]
        entry['terminal_records'] += 1
        entry['caption_generated'] += bool(record['enhanced_instruction'])
        entry['empty_caption'] += not bool(record['enhanced_instruction'])
        entry['automatic_checks_passed'] += bool(record['enhanced_instruction']) and not bool(record['review_flags'])
        entry['needs_review_by_checks'] += bool(record['review_flags']) or not bool(record['enhanced_instruction'])
    return {'status': state, 'mode': config['mode'], 'started_at': started_at, 'updated_at': utc_now(),
            'selected': len(rows), 'terminal_records': len(records), 'pending': len(rows) - len(records),
            'caption_generated': len(generated), 'empty_caption': len(records) - len(generated),
            'automatic_checks_passed': len(screened), 'needs_review_by_checks': len(needs_review),
            'processing_complete': len(records) == len(rows) and state in {'completed', 'completed_with_review'},
            'generation_authorization': ('user_authorized_full_run' if config['mode'] == 'all' else
                                         'user_authorized_task_subset' if config['mode'] == 'tasks' else 'user_authorized_preview'),
            'semantic_review': 'pending_human_review; automated checks do not establish semantic correctness',
            'full_generation_started': config['mode'] == 'all', 'model': config['model'],
            'gpu_groups': config['gpu_groups'], 'batch_size': config.get('batch_size', 1), 'config_fingerprint': config['fingerprint'],
            'failure': failure, 'tasks': counts}


PAGE_STYLE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<style>body{font:16px/1.6 system-ui;max-width:1100px;margin:30px auto;padding:0 20px;background:#f4f5f7;color:#182332}
article{background:white;border:1px solid #dce0e6;border-radius:10px;padding:24px;margin:24px 0}img{max-width:100%;height:auto}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}details{margin:12px 0}
td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse;font-size:14px}.status{background:#fff5dc;padding:12px;white-space:pre-wrap}</style>'''


def export_task_pages(output, dataset, rows, records, summary):
    by_task = defaultdict(list)
    for record in records:
        by_task[record['task']].append(record)
    table = ['<table><tr><th>Task</th><th>已完成 / 总数</th><th>已生成</th><th>检查通过</th><th>待复核</th><th>空文本</th></tr>']
    for task, counts in summary['tasks'].items():
        cards = []
        for number, record in enumerate(by_task[task], 1):
            flags = '\n'.join(record['review_flags']) or '程序检查未发现问题；仍需语义审核'
            ground = json.dumps(record.get('grounding'), ensure_ascii=False, indent=2)
            plan = json.dumps(record.get('plan'), ensure_ascii=False, indent=2)
            cards.append(f'<article id="{html.escape(record["episode"], quote=True)}"><h2>{number:02d}. {html.escape(record["episode"])}</h2>'
                         f'<img loading="lazy" width="672" src="{data_image(dataset / record["image"])}" alt="observer left, ego right">'
                         f'<h3>原任务</h3><p>{html.escape(record["original_instruction"])}</p>'
                         f'<h3>扩写候选</h3><p>{html.escape(record["enhanced_instruction"] or "未生成可渲染文本")}</p>'
                         f'<p class="status">{html.escape(flags)}</p>'
                         f'<details><summary>第一步：observer 定位</summary><pre>{html.escape(ground)}</pre></details>'
                         f'<details><summary>第二步：动作计划（{len(record["plan_attempts"])}次）</summary><pre>{html.escape(plan)}</pre></details>'
                         f'<p><a href="../records/{html.escape(task, quote=True)}/{html.escape(record["episode"], quote=True)}.json">原始记录和模型回答</a></p></article>')
        destination = output / 'review' / f'{task}.html'
        title = f'{task}：{counts["terminal_records"]}/{counts["selected"]}'
        atomic_text(destination, PAGE_STYLE + f'<title>{html.escape(title)}</title><h1>{html.escape(title)}</h1>'
                    '<p><a href="../index.html">返回全量索引</a> · 图片左 observer / 右 ego；实际右臂先写，实际左臂后写。</p>'
                    '<p>第一步：observer 896×896 定位、ego 448×448 辅助识别；第二步只根据原任务和定位结果生成动作计划。'
                    '自动检查不是语义正确性保证。带标记或空文本的记录保留在 needs_review.jsonl。</p>' + ''.join(cards) + '</html>\n')
        table.append('<tr><td><a href="review/' + html.escape(task, quote=True) + '.html">' + html.escape(task) + '</a></td>'
                     f'<td>{counts["terminal_records"]} / {counts["selected"]}</td><td>{counts["caption_generated"]}</td>'
                     f'<td>{counts["automatic_checks_passed"]}</td><td>{counts["needs_review_by_checks"]}</td><td>{counts["empty_caption"]}</td></tr>')
    table.append('</table>')
    atomic_text(output / 'index.html', PAGE_STYLE + '<title>两步 prompt 全量报告</title><h1>两步 prompt 生成报告</h1>'
                f'<p>状态：{html.escape(summary["status"])} · 已完成 {summary["terminal_records"]}/{summary["selected"]} · '
                f'有效文本 {summary["caption_generated"]} · 自动检查通过 {summary["automatic_checks_passed"]} · '
                f'待复核 {summary["needs_review_by_checks"]} · 空文本 {summary["empty_caption"]}</p>'
                '<p>Qwen2.5-VL-72B BF16 · GPU 4–7。通过自动检查仍不等于语义正确；原始回答完整保留。</p>'
                '<p><a href="text_prompt_enhanced.txt">检查通过文本</a> · <a href="text_candidates_all.txt">所有非空候选</a> · '
                '<a href="text_original.txt">原始任务</a> · <a href="needs_review.jsonl">待复核记录</a> · '
                '<a href="summary.json">汇总 JSON</a></p>' + ''.join(table) + '</html>\n')


def export_aggregate(output, dataset, rows, records, config, state, started_at, failure=None, pages=False):
    summary = summarize(rows, records, config, state, started_at, failure)
    screened = [record for record in records if record['enhanced_instruction'] and not record['review_flags']]
    needs_review = [record for record in records if record['review_flags'] or not record['enhanced_instruction']]
    # Full original manifest stays available during partial progress, while results contain only terminal records.
    atomic_text(output / 'text_original.txt', ''.join(f'{row["instruction"]}@@{row["outputs"]["image"]}\n' for row in rows))
    atomic_text(output / 'text_prompt_enhanced.txt', ''.join(f'{record["enhanced_instruction"]}@@{record["image"]}\n' for record in screened))
    atomic_text(output / 'text_candidates_all.txt', ''.join(f'{record["enhanced_instruction"]}@@{record["image"]}\n'
                                                         for record in records if record['enhanced_instruction']))
    atomic_text(output / 'results.jsonl', json_lines(records))
    atomic_text(output / 'needs_review.jsonl', json_lines(needs_review))
    completed_keys = {(record['task'], record['episode']) for record in records}
    atomic_text(output / 'pending.jsonl', json_lines({'task': row['task'], 'episode': row['episode'], 'image': row['outputs']['image']}
                for row in rows if (row['task'], row['episode']) not in completed_keys))
    write_json(output / 'summary.json', summary)
    write_json(output / 'progress.json', summary)
    if pages:
        export_task_pages(output, dataset, rows, records, summary)
    return summary


def replica_entry(group, rows, config, dataset, output, templates, status_file):
    status = {'physical_gpus': group, 'assigned': len(rows), 'started_at': utc_now(), 'status': 'running'}
    write_json(status_file, status)
    try:
        selected_worker = worker if config.get('batch_size', 1) == 1 else batched_worker
        count = selected_worker(group, rows, config, dataset, output, templates)
        status.update(status='completed', completed=count, ended_at=utc_now())
        write_json(status_file, status)
    except BaseException as exc:
        status.update(status='failed', ended_at=utc_now(), error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        write_json(status_file, status)
        raise


def run_workers(output, dataset, rows, pending, config, templates, started_at):
    context = mp.get_context('spawn')
    processes = []
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + f'_{os.getpid()}'
    failures = []
    last_export = 0.0
    last_pages = time.monotonic()
    try:
        for index, group in enumerate(GPU_GROUPS):
            assigned = pending[index::2]
            if not assigned:
                continue
            status_file = output / 'worker_status' / f'{attempt}_{group.replace(",", "_")}.json'
            process = context.Process(target=replica_entry, args=(group, assigned, config, str(dataset), str(output), templates, str(status_file)),
                                      name=f'prompt_{group.replace(",", "_")}')
            process.start()
            processes.append((process, group, status_file))
        active = {process.sentinel: (process, group, status_file) for process, group, status_file in processes}
        while active:
            for sentinel in wait(list(active), timeout=30):
                process, group, status_file = active.pop(sentinel)
                process.join()
                if process.exitcode != 0:
                    detail = json.loads(status_file.read_text()) if status_file.exists() else {}
                    failures.append({'physical_gpus': group, 'exit_code': process.exitcode, 'worker_status': str(status_file.relative_to(output)),
                                     'error': detail.get('error', 'Worker exited without a complete diagnostic; see worker log')})
                    write_json(output / 'failure.json', {'attempt': attempt, 'failures': failures, 'updated_at': utc_now()})
                    print(json.dumps({'worker_failure': failures[-1]}, ensure_ascii=False), flush=True)
                else:
                    print(f'Replica {group} completed', flush=True)
            if time.monotonic() - last_export >= 30 or not active:
                records, _ = collect_records(output, rows, config)
                refresh_pages = time.monotonic() - last_pages >= 300 or not active
                summary = export_aggregate(output, dataset, rows, records, config, 'running_with_worker_failure' if failures else 'running',
                                           started_at, failure=failures or None, pages=refresh_pages)
                print(json.dumps({key: summary[key] for key in ['status', 'terminal_records', 'selected', 'caption_generated', 'empty_caption', 'needs_review_by_checks']}, ensure_ascii=False), flush=True)
                last_export = time.monotonic()
                if refresh_pages:
                    last_pages = last_export
    except BaseException:
        for process, _, _ in processes:
            if process.is_alive():
                process.terminate()
        for process, _, _ in processes:
            process.join(timeout=15)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        raise
    finally:
        for process, _, _ in processes:
            if not process.is_alive():
                process.close()
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--all', action='store_true', help='Explicitly generate all 2500 episodes')
    mode_group.add_argument('--review-100', action='store_true', help='First and last episode of each of 50 tasks')
    mode_group.add_argument('--tasks', nargs='+', metavar='TASK', help='Generate all 50 episodes of each named task')
    parser.add_argument('--dataset-root', type=Path, default=PROJECT / 'data/robotwin_observer_ego_vpp2')
    parser.add_argument('--selection', type=Path, default=PROJECT / 'prompts/preview_selection.json')
    parser.add_argument('--model', type=Path, default=Path('/data2/models/Qwen2.5-VL-72B-Instruct'))
    parser.add_argument('--prompt-dir', type=Path, default=PROJECT / 'prompts/two_step')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--ground-max-tokens', type=int, default=768)
    parser.add_argument('--plan-max-tokens', type=int, default=2048)
    parser.add_argument('--plan-attempts', type=int, choices=[1, 2], default=2)
    parser.add_argument('--batch-size', type=int, choices=[1, 2, 4], default=4,
                        help='Episodes per replica batch; 1 uses the original serial worker')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    mode = 'all' if args.all else 'review_100' if args.review_100 else 'tasks' if args.tasks else 'preview'
    default_name = {'all': 'prompt_two_step_full_v1', 'review_100': 'prompt_two_step_review100_v1',
                    'tasks': 'prompt_two_step_tasks_v1', 'preview': 'prompt_two_step_batch_preview_v1'}[mode]
    dataset = args.dataset_root.resolve()
    output = (args.output_root or PROJECT / 'runs' / default_name).resolve()
    if output == dataset or output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError('Output must be separate from the source dataset')
    if args.ground_max_tokens < 1 or args.plan_max_tokens < 1:
        raise ValueError('Token limits must be positive')
    rows = choose_rows(dataset, args.selection, mode, args.tasks)
    templates = {name: (args.prompt_dir / f'{name}.txt').read_text() for name in ['ground_observer', 'plan_actions']}
    if any(template.count('{task_instruction}') != 1 for template in templates.values()) or templates['plan_actions'].count('{grounding_json}') != 1:
        raise ValueError('Invalid template placeholders')
    # Hash the modules actually imported, including when this entry point is staged separately.
    source_paths = {filename: Path(__file__) if filename == Path(__file__).name else Path(sys.modules[Path(filename).stem].__file__)
                    for filename in SNAPSHOT_FILES}
    source_hashes = {filename: sha256(path) for filename, path in source_paths.items()}
    config = {'model': str(args.model.resolve()), 'model_config_sha256': sha256(args.model / 'config.json'),
              'model_index_sha256': sha256(args.model / 'model.safetensors.index.json'), 'source_sha256': source_hashes,
              'templates_sha256': {name: hashlib.sha256(template.encode()).hexdigest() for name, template in templates.items()},
              'dataset': str(dataset), 'manifest_sha256': sha256(dataset / 'manifest.jsonl'), 'mode': mode, 'gpu_groups': GPU_GROUPS,
              'dtype': 'bfloat16', 'device_map': 'balanced', 'attention': 'sdpa', 'do_sample': False,
              'ground_max_tokens': args.ground_max_tokens, 'plan_max_tokens': args.plan_max_tokens, 'plan_attempts': args.plan_attempts,
              'batch_size': args.batch_size, 'padding_side': 'left' if args.batch_size > 1 else 'single_sample',
              'vlm_ground_observer_size': [896, 896], 'vlm_ground_ego_size': [448, 448],
              'plan_input': 'text_only', 'original_view_size': [224, 224],
              'selection': [{'task': row['task'], 'episode': row['episode'], 'image': row['outputs']['image'],
                             'image_sha256': row['output_sha256']['image'], 'instruction': row['instruction']} for row in rows]}
    if mode == 'tasks':
        config['tasks'] = sorted(args.tasks)
    config['fingerprint'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if args.dry_run:
        print(json.dumps({'mode': mode, 'selected': len(rows), 'tasks': len({row['task'] for row in rows}),
                          'output': str(output), 'gpu_groups': GPU_GROUPS, 'batch_size': args.batch_size,
                          'config_fingerprint': config['fingerprint']}, ensure_ascii=False, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.generation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = output / 'run_config.json'
        if config_path.exists():
            if json.loads(config_path.read_text()) != config:
                raise ValueError('Different configuration in output; choose a new output directory')
        elif any(output.iterdir()) and any(path.name != '.generation.lock' for path in output.iterdir()):
            raise ValueError('Output has files without a run_config.json; choose a clean directory')
        write_json(config_path, config)
        for name, template in templates.items():
            atomic_text(output / 'prompts' / f'{name}.txt', template)
        for filename in SNAPSHOT_FILES:
            atomic_text(output / 'source_snapshot' / filename, source_paths[filename].read_text())
        atomic_text(output / 'selection.jsonl', json_lines(rows))
        ensure_image_link(output, dataset)
        records, pending = collect_records(output, rows, config, verify_crops=True)
        started_at = utc_now()
        if (output / 'failure.json').exists():
            archived_failure = output / 'failure_history' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + f'_{os.getpid()}.json')
            archived_failure.parent.mkdir(parents=True, exist_ok=True)
            (output / 'failure.json').replace(archived_failure)
        export_aggregate(output, dataset, rows, records, config, 'starting', started_at, pages=True)
        print(f'Selected {len(rows)} episodes ({mode}); terminal {len(records)}, pending {len(pending)}', flush=True)
        original_handler = signal.getsignal(signal.SIGTERM)
        def interrupted(signum, frame):
            raise InterruptedError(f'Received signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        try:
            failures = run_workers(output, dataset, rows, pending, config, templates, started_at) if pending else []
            records, missing = collect_records(output, rows, config, verify_crops=True)
            # Revalidate source image hashes and exact selection before publishing the final report.
            final_rows = choose_rows(dataset, args.selection, mode, args.tasks)
            if sha256(dataset / 'manifest.jsonl') != config['manifest_sha256'] or final_rows != rows:
                raise ValueError('Source manifest or selection changed while generation was running')
            if failures or missing:
                state = 'failed'
                failure = {'worker_failures': failures, 'missing_records': len(missing)}
                write_json(output / 'failure.json', failure)
            else:
                state = 'completed_with_review' if any(record['review_flags'] or not record['enhanced_instruction'] for record in records) else 'completed'
                failure = None
            summary = export_aggregate(output, dataset, rows, records, config, state, started_at, failure=failure, pages=True)
            write_json(output / 'validation_report.json', {'checked_at': utc_now(), 'input_count': len(rows), 'terminal_record_count': len(records),
                        'all_inputs_match': True, 'all_terminal_records_present': not missing, 'image_paths_and_hashes_verified': True,
                        'crop_hashes_verified': True, 'worker_failures': failures, 'processing_complete': summary['processing_complete'],
                        'caption_generated': summary['caption_generated'], 'empty_caption': summary['empty_caption'],
                        'automatic_checks_passed': summary['automatic_checks_passed'], 'needs_review_by_checks': summary['needs_review_by_checks']})
            print(json.dumps({key: value for key, value in summary.items() if key != 'tasks'}, ensure_ascii=False, indent=2), flush=True)
            if state == 'failed':
                print('Generation incomplete; see failure.json, progress.json and worker logs', file=sys.stderr, flush=True)
                return 1
        except BaseException as exc:
            failure = {'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc(), 'updated_at': utc_now()}
            write_json(output / 'failure.json', failure)
            try:
                records, _ = collect_records(output, rows, config)
                export_aggregate(output, dataset, rows, records, config, 'interrupted' if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else 'failed',
                                 started_at, failure=failure, pages=True)
            except Exception as report_error:
                write_json(output / 'progress.json', {'status': 'failed', 'failure': failure, 'report_error': str(report_error),
                                                      'selected': len(rows), 'processing_complete': False, 'updated_at': utc_now()})
            raise
        finally:
            signal.signal(signal.SIGTERM, original_handler)


if __name__ == '__main__':
    raise SystemExit(main())

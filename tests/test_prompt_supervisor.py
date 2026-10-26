"""Actual CPU subprocess checks for supervisor resume and bounded empty retry.

The fake batch executable writes realistic counts/terminal records but never
imports a model or accesses a GPU. The production supervisor source is imported
unchanged; only its project root points to the temporary fake batch executable.
"""
import json
import hashlib
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap


SUPERVISOR = Path(__file__).resolve().parents[1] / 'scripts/run_prompt_full.py'
PROJECT_HELPERS = '/data2/niurc/vpp2-obj-generation/scripts'


FAKE_BATCH = r'''
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--all', action='store_true')
parser.add_argument('--batch-size')
parser.add_argument('--output-root', type=Path)
args = parser.parse_args()
output = args.output_root
output.mkdir(parents=True, exist_ok=True)
records = output / 'records' / 'task'
records.mkdir(parents=True, exist_ok=True)
calls_path = output / 'fake_batch_calls.jsonl'
attempt = len(calls_path.read_text().splitlines()) + 1 if calls_path.exists() else 1
existing = sorted(records.glob('*.json'))
with calls_path.open('a') as calls:
    calls.write(json.dumps({'attempt': attempt, 'existing': len(existing), 'pid': os.getpid()}) + '\n')
mode = os.environ['FAKE_BATCH_SCENARIO']

def write_record(index, caption, flags):
    path = records / f'episode_{index:07d}.json'
    path.write_text(json.dumps({'task': 'task', 'episode': f'episode_{index:07d}',
        'enhanced_instruction': caption, 'review_flags': flags,
        'raw_response': f'raw answer for episode {index}, attempt {attempt}', 'written_attempt': attempt}))

if mode == 'always_fatal':
    (output / 'summary.json').write_text(json.dumps({'processing_complete': False, 'selected': 2500, 'terminal_records': 0, 'pending': 2500}))
    raise SystemExit(17)
if mode == 'fatal_then_resume' and attempt == 1:
    write_record(0, 'preserve successful first record', [])
    (output / 'first_record_saved.json').write_bytes((records / 'episode_0000000.json').read_bytes())
    (output / 'summary.json').write_text(json.dumps({'processing_complete': False, 'selected': 2500, 'terminal_records': 1, 'pending': 2499}))
    raise SystemExit(17)

for index in range(2500):
    path = records / f'episode_{index:07d}.json'
    if path.exists():
        continue
    if mode == 'empty_retry' and (index == 1 or (index == 0 and attempt == 1)):
        write_record(index, '', ['pipeline_error: simulated empty output'])
    elif mode == 'empty_retry' and index == 2:
        write_record(index, 'nonempty caption with a semantic review flag', ['uncertain arm assignment'])
    else:
        write_record(index, f'caption for episode {index}', [])

if mode == 'empty_retry' and attempt == 1:
    (output / 'flagged_record_saved.json').write_bytes((records / 'episode_0000002.json').read_bytes())

values = [json.loads(path.read_text()) for path in records.glob('*.json')]
empty = sum(not value['enhanced_instruction'] for value in values)
passed = sum(bool(value['enhanced_instruction']) and not value['review_flags'] for value in values)
summary = {'processing_complete': len(values) == 2500, 'selected': 2500, 'terminal_records': len(values),
    'pending': 2500 - len(values), 'caption_generated': len(values) - empty, 'empty_caption': empty,
    'automatic_checks_passed': passed, 'needs_review_by_checks': len(values) - passed}
(output / 'summary.json').write_text(json.dumps(summary))
(output / 'progress.json').write_text(json.dumps(summary))
'''


def run_fake_supervisor(root, scenario, max_restarts):
    project = root / 'fake_project'
    scripts = project / 'scripts'
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / 'enhance_prompts_two_step_batch.py').write_text(textwrap.dedent(FAKE_BATCH))
    runner = root / 'run_test_supervisor.py'
    runner.write_text(textwrap.dedent(f'''
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location('supervisor_under_test', {str(SUPERVISOR)!r})
        supervisor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(supervisor)
        supervisor.PROJECT = Path({str(project)!r})
        raise SystemExit(supervisor.main())
    '''))
    output = root / 'output'
    environment = {**os.environ, 'PYTHONPATH': PROJECT_HELPERS, 'PYTHONDONTWRITEBYTECODE': '1',
                   'FAKE_BATCH_SCENARIO': scenario}
    process = subprocess.Popen([sys.executable, str(runner), '--output-root', str(output), '--max-restarts', str(max_restarts)],
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, env=environment, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=100)
    except subprocess.TimeoutExpired:
        # Cleanup only the process/session started by this test and the child PGID
        # explicitly recorded by its supervisor. Never inspect or signal GPUs.
        status_path = output.with_name(output.name + '_control') / 'status.json'
        if status_path.exists():
            child_pid = json.loads(status_path.read_text()).get('child_pid')
            if child_pid:
                try:
                    os.killpg(child_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        os.killpg(process.pid, signal.SIGTERM)
        process.communicate(timeout=10)
        raise
    status = json.loads((output.with_name(output.name + '_control') / 'status.json').read_text())
    return output, process.returncode, status, stdout, stderr


def test_actual_fatal_process_resumes_completed_rows_and_restart_limit_is_bounded(tmp_path):
    output, code, status, stdout, stderr = run_fake_supervisor(tmp_path / 'resume', 'fatal_then_resume', max_restarts=1)
    assert code == 0, (stdout, stderr, status)
    assert status['status'] == 'completed'
    assert status['fatal_restarts'] == 1
    assert [attempt['returncode'] for attempt in status['attempts']] == [17, 0]
    calls = [json.loads(line) for line in (output / 'fake_batch_calls.jsonl').read_text().splitlines()]
    assert [call['existing'] for call in calls] == [0, 1]
    assert (output / 'records/task/episode_0000000.json').read_bytes() == (output / 'first_record_saved.json').read_bytes()
    assert len(list((output / 'records/task').glob('*.json'))) == 2500
    assert (output.with_name(output.name + '_control') / 'completion.json').exists()
    # A permanently failing subprocess with a zero restart budget gets one attempt.
    failed_output, code, status, _, _ = run_fake_supervisor(tmp_path / 'bounded', 'always_fatal', max_restarts=0)
    assert code == 1 and status['status'] == 'failed_after_bounded_restarts'
    assert status['fatal_restarts'] == 0 and len(status['attempts']) == 1
    assert not (failed_output.with_name(failed_output.name + '_control') / 'completion.json').exists()
    # Relaunching the supervisor itself cannot reset an exhausted restart budget.
    _, code, resumed_status, _, _ = run_fake_supervisor(tmp_path / 'bounded', 'always_fatal', max_restarts=0)
    assert code == 1 and len(resumed_status['attempts']) == 1
    assert len((failed_output / 'fake_batch_calls.jsonl').read_text().splitlines()) == 1
    # A completed supervisor restart is idempotent and preserves its prior count.
    _, code, resumed_status, _, _ = run_fake_supervisor(tmp_path / 'resume', 'fatal_then_resume', max_restarts=1)
    assert code == 0 and resumed_status['fatal_restarts'] == 1
    assert len(resumed_status['attempts']) == 2


def test_actual_empty_retry_archives_only_empty_records_once(tmp_path):
    output, code, status, stdout, stderr = run_fake_supervisor(tmp_path, 'empty_retry', max_restarts=0)
    assert code == 0, (stdout, stderr, status)
    assert status['status'] == 'completed_with_empty_captions'
    assert status['empty_caption_retry_used'] is True
    assert len(status['attempts']) == 2
    assert status['fatal_restarts'] == 0
    calls = [json.loads(line) for line in (output / 'fake_batch_calls.jsonl').read_text().splitlines()]
    assert [call['existing'] for call in calls] == [0, 2498]
    archived = sorted((output / 'empty_caption_retry_archive/task').glob('*.json'))
    assert [path.name for path in archived] == ['episode_0000000.json', 'episode_0000001.json']
    assert all(json.loads(path.read_text())['enhanced_instruction'] == '' for path in archived)
    assert all(json.loads(path.read_text())['written_attempt'] == 1 for path in archived)
    # Nonempty flagged output is never rewritten or placed in the empty retry archive.
    flagged = output / 'records/task/episode_0000002.json'
    assert flagged.read_bytes() == (output / 'flagged_record_saved.json').read_bytes()
    repaired = json.loads((output / 'records/task/episode_0000000.json').read_text())
    still_empty = json.loads((output / 'records/task/episode_0000001.json').read_text())
    assert repaired['enhanced_instruction'] and repaired['written_attempt'] == 2
    assert still_empty['enhanced_instruction'] == '' and still_empty['written_attempt'] == 2
    assert status['result_counts']['empty_caption'] == 1
    assert status['result_counts']['caption_generated'] == 2499
    assert status['result_counts']['needs_review_by_checks'] == 2


def test_partial_archive_intent_recovers_under_generation_lock_without_resetting_retry(tmp_path):
    output = tmp_path / 'output'
    records = output / 'records/task'
    records.mkdir(parents=True)
    entries = []
    for index in range(2500):
        path = records / f'episode_{index:07d}.json'
        path.write_text(json.dumps({'task': 'task', 'episode': f'episode_{index:07d}',
            'enhanced_instruction': '' if index < 2 else f'original caption {index}',
            'review_flags': ['pipeline_error: original empty'] if index < 2 else [],
            'raw_response': f'original raw response {index}', 'written_attempt': 1}))
        if index < 2:
            entries.append({'source': str(path.relative_to(output)),
                'destination': f'empty_caption_retry_archive/task/{path.name}',
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    # Simulate a crash after the durable intent and the first of two atomic renames.
    first_archive = output / entries[0]['destination']
    first_archive.parent.mkdir(parents=True)
    (output / entries[0]['source']).replace(first_archive)
    (output / 'fake_batch_calls.jsonl').write_text(json.dumps({'attempt': 1, 'existing': 0, 'pid': 99999}) + '\n')
    control = output.with_name(output.name + '_control')
    control.mkdir()
    command = [sys.executable, '-u', str(tmp_path / 'fake_project/scripts/enhance_prompts_two_step_batch.py'),
               '--all', '--batch-size', '4', '--output-root', str(output)]
    prior = {'supervisor_pid': 99999, 'started_at': 'earlier', 'output_root': str(output),
        'gpu_groups': ['4,5', '6,7'], 'batch_size': 4, 'max_restarts': 0, 'status': 'interrupted',
        'attempts': [{'attempt': 1, 'pid': 99999, 'returncode': 0}], 'fatal_restarts': 0,
        'empty_caption_retry_used': True, 'next_launch_authorized': False, 'child_pid': None, 'command': command,
        'empty_caption_retry_intent': {'state': 'archiving', 'created_at': 'earlier', 'entries': entries}}
    (control / 'status.json').write_text(json.dumps(prior))
    # A concurrently held generation lock prevents any archive mutation or launch.
    with (output / '.generation.lock').open('a') as generation_lock:
        fcntl.flock(generation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _, code, status, _, _ = run_fake_supervisor(tmp_path, 'empty_retry', max_restarts=0)
        assert code != 0 and status['empty_caption_retry_intent']['state'] == 'archiving'
        assert (output / entries[1]['source']).exists()
        assert len((output / 'fake_batch_calls.jsonl').read_text().splitlines()) == 1
    # Restart continues exactly the unfinished rename, then launches one repair pass.
    _, code, status, stdout, stderr = run_fake_supervisor(tmp_path, 'empty_retry', max_restarts=0)
    assert code == 0, (stdout, stderr, status)
    assert status['empty_caption_retry_intent']['state'] == 'complete'
    assert status['fatal_restarts'] == 0 and status['empty_caption_retry_used']
    assert len(status['attempts']) == 2 and status['status'] == 'completed_with_empty_captions'
    for entry in entries:
        assert hashlib.sha256((output / entry['destination']).read_bytes()).hexdigest() == entry['sha256']
    assert len((output / 'fake_batch_calls.jsonl').read_text().splitlines()) == 2

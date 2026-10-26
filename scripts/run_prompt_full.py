#!/usr/bin/env python3
"""Durable full-run supervisor: bounded resume, diagnostics and final status.

Run in a detached session. Only the requested local batch process is managed.
No training, source-data writes, external messages, or unbounded retries occur.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from enhance_prompts import sha256, write_json

PROJECT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def read_status(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def cleanup_owned_group(pgid):
    """Clean only a session group created by this supervisor, even if its leader died."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.25)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def archive_empty_records(output, status, save):
    """Journal one empty-caption repair pass before moving any terminal records."""
    with (output / '.generation.lock').open('a') as generation_lock:
        fcntl.flock(generation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        intent = status.get('empty_caption_retry_intent')
        if intent is None:
            entries = []
            for path in sorted((output / 'records').glob('*/*.json')):
                record = json.loads(path.read_text())
                if record['enhanced_instruction']:
                    continue
                destination = output / 'empty_caption_retry_archive' / path.relative_to(output / 'records')
                if destination.exists():
                    raise ValueError('Unjournaled empty-caption archive exists; refusing to overwrite')
                entries.append({'source': str(path.relative_to(output)), 'destination': str(destination.relative_to(output)),
                                'sha256': sha256(path)})
            intent = {'state': 'archiving', 'created_at': now(), 'entries': entries}
            status.update(empty_caption_retry_intent=intent, empty_caption_retry_used=True,
                          next_launch_authorized=False, status='archiving_empty_captions')
            save()  # Durable intent must precede the first atomic rename.
        if intent['state'] == 'complete':
            return
        if intent['state'] != 'archiving':
            raise ValueError('Unknown empty-caption archive transaction state')
        for entry in intent['entries']:
            source, destination = output / entry['source'], output / entry['destination']
            if not source.resolve().is_relative_to((output / 'records').resolve()) or not destination.resolve().is_relative_to((output / 'empty_caption_retry_archive').resolve()):
                raise ValueError('Archive intent path outside its expected directory')
            if destination.exists():
                if source.exists() or sha256(destination) != entry['sha256']:
                    raise ValueError('Archived empty-caption record differs from the durable intent')
                continue  # A previous supervisor already completed this rename.
            if not source.exists() or sha256(source) != entry['sha256']:
                raise ValueError('Empty-caption record changed since archive intent was saved')
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
        intent.update(state='complete', completed_at=now())
        status.update(archived_empty_records=[entry['destination'] for entry in intent['entries']],
                      next_launch_authorized=True, status='retrying_empty_captions')
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=PROJECT / 'runs/prompt_two_step_full_v1')
    parser.add_argument('--batch-size', type=int, choices=[1, 2, 4], default=4)
    parser.add_argument('--max-restarts', type=int, choices=[0, 1, 2], default=2)
    args = parser.parse_args()
    output = args.output_root.resolve()
    control = output.with_name(output.name + '_control')
    control.mkdir(parents=True, exist_ok=True)
    status = {'supervisor_pid': os.getpid(), 'started_at': now(), 'output_root': str(output),
        'gpu_groups': ['4,5', '6,7'], 'batch_size': args.batch_size, 'status': 'starting',
        'attempts': [], 'fatal_restarts': 0, 'empty_caption_retry_used': False,
        'max_restarts': args.max_restarts, 'next_launch_authorized': True}
    command = [sys.executable, '-u', str(PROJECT / 'scripts/enhance_prompts_two_step_batch.py'),
        '--all', '--batch-size', str(args.batch_size), '--output-root', str(output)]
    status['command'] = command
    child = None
    def save():
        status['updated_at'] = now()
        status['batch_progress'] = read_status(output / 'progress.json')
        write_json(control / 'status.json', status)
    def interrupted(signum, frame):
        raise InterruptedError(f'Received signal {signum}')
    with (control / '.supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior_path = control / 'status.json'
        prior = read_status(prior_path)
        if prior_path.exists() and prior is None:
            raise ValueError('Existing supervisor status is unreadable; refusing to reset retry counters')
        if prior is not None:
            for key, value in {'output_root': str(output), 'batch_size': args.batch_size, 'command': command,
                               'max_restarts': args.max_restarts}.items():
                if key in prior and prior[key] != value:
                    raise ValueError(f'Existing supervisor configuration differs: {key}')
            status.update(prior)
            status.update(supervisor_pid=os.getpid(), resumed_at=now(), max_restarts=args.max_restarts)
            if prior.get('error'):
                status.setdefault('supervisor_error_history', []).append({
                    'error': prior['error'], 'traceback': prior.get('traceback'), 'updated_at': prior.get('updated_at')})
                status.pop('error', None)
                status.pop('traceback', None)
            status['next_launch_authorized'] = prior.get('next_launch_authorized',
                prior.get('status') in {'starting', 'resuming_after_failure', 'retrying_empty_captions'})
            if prior.get('status') in {'completed', 'completed_with_review', 'completed_with_empty_captions'}:
                save()
                return 0
            if prior.get('status') == 'failed_after_bounded_restarts':
                save()
                return 1
            # Do not signal a PID recovered from disk: it could have been reused.
            # Refuse overlapping supervisors if the previous batch group remains.
            if prior.get('child_pid'):
                try:
                    os.killpg(prior['child_pid'], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError('Previous batch process group still exists; refusing to launch an overlapping batch')
                status['child_pid'] = None
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            if status.get('empty_caption_retry_intent', {}).get('state') == 'archiving':
                archive_empty_records(output, status, save)
            if not status['next_launch_authorized']:
                if status['fatal_restarts'] >= args.max_restarts:
                    status['status'] = 'failed_after_bounded_restarts'
                    save()
                    return 1
                status['fatal_restarts'] += 1
                status.update(next_launch_authorized=True, status='resuming_after_supervisor_interruption')
                save()
            while True:
                attempt = len(status['attempts']) + 1
                log_path = control / f'batch_attempt_{attempt:02d}.log'
                status.update(next_launch_authorized=False, status='launching')
                save()
                with log_path.open('a', buffering=1) as log:
                    child = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
                    current = {'attempt': attempt, 'pid': child.pid, 'started_at': now(), 'log': str(log_path)}
                    status['attempts'].append(current)
                    status.update(status='running', child_pid=child.pid)
                    save()
                    while True:
                        try:
                            returncode = child.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            save()
                    current.update(returncode=returncode, ended_at=now())
                    cleanup_owned_group(child.pid)
                child = None
                status['child_pid'] = None
                save()
                summary = read_status(output / 'summary.json') or {}
                if returncode != 0 or not summary.get('processing_complete'):
                    if status['fatal_restarts'] >= args.max_restarts:
                        status['status'] = 'failed_after_bounded_restarts'
                        save()
                        return 1
                    status['fatal_restarts'] += 1
                    status.update(status='resuming_after_failure', next_launch_authorized=True)
                    save()
                    time.sleep(30)
                    continue
                # One bounded repair pass for rows with no renderable caption.
                # Existing nonempty captions (including flagged ones) are retained.
                if summary.get('empty_caption', 0) and not status['empty_caption_retry_used']:
                    archive_empty_records(output, status, save)
                    continue
                if summary.get('selected') != 2500 or summary.get('terminal_records') != 2500 or summary.get('pending'):
                    raise ValueError('Full-run completion counters do not match 2500 input episodes')
                status['status'] = ('completed_with_empty_captions' if summary.get('empty_caption') else
                    'completed_with_review' if summary.get('needs_review_by_checks') else 'completed')
                status['finished_at'] = now()
                status['result_counts'] = {key: summary[key] for key in ['selected', 'terminal_records',
                    'caption_generated', 'empty_caption', 'automatic_checks_passed', 'needs_review_by_checks']}
                save()
                write_json(control / 'completion.json', status)
                return 0
        except BaseException as exc:
            if child is not None:
                # This process group was created by this supervisor alone.
                cleanup_owned_group(child.pid)
                try:
                    child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    cleanup_owned_group(child.pid)
                    child.wait(timeout=10)
            status.update(status='interrupted' if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else 'failed',
                error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(), child_pid=None)
            save()
            raise


if __name__ == '__main__':
    raise SystemExit(main())

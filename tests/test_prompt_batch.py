import copy
import json
import os
from pathlib import Path
import sys

import pytest
from PIL import Image

sys.path.insert(0, '/data2/niurc/vpp2-obj-generation/scripts')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import enhance_prompts_two_step_batch as batch


@pytest.fixture
def sample(tmp_path):
    dataset, output = tmp_path / 'dataset', tmp_path / 'output'
    output.mkdir()
    rows = []
    for index in range(3):
        task, episode = 'task_a' if index < 2 else 'task_b', f'episode_{index:07d}'
        image = f'images/{task}/{episode}.png'
        path = dataset / image
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (448, 224), color=(index * 70, 120, 30)).save(path)
        rows.append({'task': task, 'episode': episode, 'outputs': {'image': image},
                     'output_sha256': {'image': batch.sha256(path)}, 'instruction': f'Lift object {index}',
                     'instruction_key': 'seen', 'instruction_index': 0,
                     'inputs': {'instruction': {'path': f'/source/{task}/{episode}.json', 'sha256': f'hash{index}'}},
                     'view_order': ['observer', 'ego'], 'geometry': {'output_size_wh': [448, 224]}})
    config = {'fingerprint': 'run_hash', 'mode': 'all', 'model': '/local/Qwen2.5-VL-72B-Instruct', 'gpu_groups': batch.GPU_GROUPS}
    records = []
    for row in rows:
        records.append({'task': row['task'], 'episode': row['episode'], 'image': row['outputs']['image'],
                        'image_sha256': row['output_sha256']['image'], 'original_instruction': row['instruction'],
                        'instruction_source': row['inputs']['instruction'], 'instruction_key': 'seen', 'instruction_index': 0,
                        'config_fingerprint': 'run_hash', 'physical_gpus': '4,5', 'review_flags': [],
                        'elapsed_seconds': 0.1, 'plan_attempts': [], 'grounding': None, 'plan': None,
                        'enhanced_instruction': 'The right robot gripper lifts the object, while the left robot gripper keeps static. '
                        'This is a composed robot video with the observer view on the left and the robot ego view on the right.'})
    return dataset, output, rows, config, records


@pytest.mark.parametrize('field,value', [
    ('task', 'different_task'), ('episode', 'different_episode'), ('image', 'images/wrong.png'),
    ('image_sha256', 'changed'), ('original_instruction', 'different task'), ('config_fingerprint', 'different_run'),
    ('instruction_source', {'path': '/different.json'}), ('physical_gpus', '0,1')])
def test_resume_rejects_provenance_changes(sample, field, value):
    _, output, rows, config, records = sample
    record = copy.deepcopy(records[0])
    record[field] = value
    batch.write_json(batch.record_path(output, rows[0]), record)
    with pytest.raises(ValueError):
        batch.collect_records(output, rows, config)


def test_partial_exports_keep_flagged_and_empty_records_distinct(sample):
    dataset, output, rows, config, records = sample
    records[1]['review_flags'] = ['uncertain arm assignment']
    records[2]['enhanced_instruction'] = ''
    records[2]['review_flags'] = ['pipeline_error: test']
    # Only 2 records are currently complete; the third should remain pending.
    for row, record in zip(rows[:2], records[:2]):
        batch.write_json(batch.record_path(output, row), record)
    results, pending = batch.collect_records(output, rows, config)
    assert len(pending) == 1
    summary = batch.export_aggregate(output, dataset, rows, results, config, 'running', 'start', pages=True)
    assert summary['terminal_records'] == 2 and summary['pending'] == 1
    assert summary['automatic_checks_passed'] == 1 and summary['needs_review_by_checks'] == 1
    assert not summary['processing_complete']
    assert len((output / 'text_original.txt').read_text().splitlines()) == 3
    assert len((output / 'text_prompt_enhanced.txt').read_text().splitlines()) == 1
    assert len((output / 'text_candidates_all.txt').read_text().splitlines()) == 2
    assert len((output / 'pending.jsonl').read_text().splitlines()) == 1
    # Terminal empty records are retained in needs_review, never written as empty prompt lines.
    batch.write_json(batch.record_path(output, rows[2]), records[2])
    results, pending = batch.collect_records(output, rows, config)
    summary = batch.export_aggregate(output, dataset, rows, results, config, 'completed_with_review', 'start', pages=True)
    assert not pending and summary['processing_complete']
    assert summary['caption_generated'] == 2 and summary['empty_caption'] == 1
    assert summary['needs_review_by_checks'] == 2
    assert len((output / 'needs_review.jsonl').read_text().splitlines()) == 2
    assert len((output / 'results.jsonl').read_text().splitlines()) == 3
    assert '<img' not in (output / 'index.html').read_text()
    page = (output / 'review/task_a.html').read_text()
    assert '../records/task_a/episode_0000000.json' in page
    assert 'data:image/png;base64,' in page
    assert (output / 'review/task_b.html').exists()


def test_failed_run_never_claims_complete_even_when_all_records_exist(sample):
    _, _, rows, config, records = sample
    summary = batch.summarize(rows, records, config, 'failed', 'start', failure={'exit_code': 1})
    assert summary['pending'] == 0
    assert not summary['processing_complete']
    assert summary['failure'] == {'exit_code': 1}


def test_empty_record_needs_diagnostic_and_crop_hash_is_checked(sample):
    _, output, rows, config, records = sample
    record = records[0]
    record['enhanced_instruction'] = ''
    with pytest.raises(ValueError, match='diagnostic'):
        batch.validate_record(record, rows[0], config)
    record['review_flags'] = ['pipeline_error: example']
    batch.validate_record(record, rows[0], config)
    crop = output / 'views/test.png'
    crop.parent.mkdir()
    Image.new('RGB', (224, 224)).save(crop)
    record['observer_crop'] = 'views/test.png'
    record['observer_crop_sha256'] = batch.sha256(crop)
    batch.validate_record(record, rows[0], config, output)
    crop.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed observer crop'):
        batch.validate_record(record, rows[0], config, output)


def test_full_count_and_review100_selection(monkeypatch):
    rows = [{'task': f'task_{task:02d}', 'episode': f'episode_{ep:07d}'} for task in range(50) for ep in range(50)]
    monkeypatch.setattr(batch, 'select_rows', lambda *args: list(reversed(rows)))
    selected = batch.choose_rows(Path('/unused'), Path('/unused'), 'review_100')
    assert len(selected) == 100
    assert selected[0]['episode'] == 'episode_0000000'
    assert selected[1]['episode'] == 'episode_0000049'
    assert len(batch.choose_rows(Path('/unused'), Path('/unused'), 'all')) == 2500
    monkeypatch.setattr(batch, 'select_rows', lambda *args: rows[:-1])
    with pytest.raises(ValueError, match='2500'):
        batch.choose_rows(Path('/unused'), Path('/unused'), 'all')


def test_task_subset_selection_is_sorted_complete_and_rejects_unknown_tasks(monkeypatch):
    rows = [{'task': f'task_{task:02d}', 'episode': f'episode_{ep:07d}'} for task in range(50) for ep in range(50)]
    monkeypatch.setattr(batch, 'select_rows', lambda *args: list(reversed(rows)))
    selected = batch.choose_rows(Path('/unused'), Path('/unused'), 'tasks', ['task_07', 'task_02'])
    assert len(selected) == 100
    assert selected == [row for row in rows if row['task'] in {'task_02', 'task_07'}]
    assert len(batch.choose_rows(Path('/unused'), Path('/unused'), 'tasks', ['task_02'])) == 50
    with pytest.raises(ValueError, match='Unknown task selection: nonexistent'):
        batch.choose_rows(Path('/unused'), Path('/unused'), 'tasks', ['nonexistent'])
    with pytest.raises(ValueError, match='Duplicate task selection'):
        batch.choose_rows(Path('/unused'), Path('/unused'), 'tasks', ['task_02', 'task_02'])
    with pytest.raises(ValueError, match='at least one task'):
        batch.choose_rows(Path('/unused'), Path('/unused'), 'tasks', [])
    # An incomplete unselected task invalidates the full manifest as well.
    monkeypatch.setattr(batch, 'select_rows', lambda *args: rows[:-1])
    with pytest.raises(ValueError, match='2500'):
        batch.choose_rows(Path('/unused'), Path('/unused'), 'tasks', ['task_02'])


@pytest.fixture
def full_manifest(sample):
    dataset, _, example_rows, _, _ = sample
    original = example_rows[0]
    original_image = dataset / original['outputs']['image']
    rows = []
    for task in range(50):
        for episode in range(50):
            row = copy.deepcopy(original)
            row.update(task=f'task_{task:02d}', episode=f'episode_{episode:07d}')
            row['outputs']['image'] = f'images/{row["task"]}/{row["episode"]}.png'
            image = dataset / row['outputs']['image']
            image.parent.mkdir(parents=True, exist_ok=True)
            os.link(original_image, image)
            rows.append(row)
    batch.atomic_text(dataset / 'manifest.jsonl', batch.json_lines(rows))
    return dataset, rows


def test_task_subset_validates_images_outside_requested_task_and_preserves_preview_limit(full_manifest, tmp_path):
    dataset, rows = full_manifest
    assert len(batch.choose_rows(dataset, Path('/unused'), 'tasks', ['task_02'])) == 50
    selection = tmp_path / 'preview_21.json'
    batch.write_json(selection, [{'task': row['task'], 'episode': row['episode']} for row in rows[:21]])
    with pytest.raises(ValueError, match='1–20 episodes'):
        batch.choose_rows(dataset, selection, 'preview')
    damaged = dataset / rows[-1]['outputs']['image']
    damaged.unlink()  # Replace just this link so the requested task images stay intact.
    damaged.write_bytes(b'changed outside selected task')
    with pytest.raises(ValueError, match='Changed initial image'):
        batch.choose_rows(dataset, Path('/unused'), 'tasks', ['task_02'])


@pytest.mark.parametrize('other_mode', ['--all', '--review-100'])
def test_tasks_cli_is_mutually_exclusive(monkeypatch, other_mode):
    monkeypatch.setattr(sys, 'argv', ['batch', '--tasks', 'task_02', other_mode, '--dry-run'])
    with pytest.raises(SystemExit) as error:
        batch.main()
    assert error.value.code == 2


def test_task_subset_cli_records_scope_default_output_and_authorization(full_manifest, tmp_path, monkeypatch):
    dataset, _ = full_manifest
    project, model, prompts = tmp_path / 'project', tmp_path / 'model', tmp_path / 'prompts'
    model.mkdir()
    prompts.mkdir()
    batch.write_json(model / 'config.json', {})
    batch.write_json(model / 'model.safetensors.index.json', {})
    (prompts / 'ground_observer.txt').write_text('{task_instruction}')
    (prompts / 'plan_actions.txt').write_text('{task_instruction} {grounding_json}')
    monkeypatch.setattr(batch, 'PROJECT', project)
    monkeypatch.setattr(batch, 'SNAPSHOT_FILES', [])
    monkeypatch.setattr(sys, 'argv', ['batch', '--tasks', 'task_07', 'task_02', '--dataset-root', str(dataset),
                                    '--model', str(model), '--prompt-dir', str(prompts)])
    def no_model(output, dataset, rows, pending, config, templates, started_at):
        assert len(rows) == len(pending) == 100
        assert rows[0]['task'] == 'task_02' and rows[-1]['task'] == 'task_07'
        return [{'physical_gpus': '4,5', 'error': 'CPU test: no model started'}]
    monkeypatch.setattr(batch, 'run_workers', no_model)
    assert batch.main() == 1
    output = project / 'runs/prompt_two_step_tasks_v1'
    config = json.loads((output / 'run_config.json').read_text())
    summary = json.loads((output / 'summary.json').read_text())
    assert config['mode'] == 'tasks' and config['tasks'] == ['task_02', 'task_07']
    assert len(config['selection']) == 100
    assert summary['generation_authorization'] == 'user_authorized_task_subset'
    assert not summary['full_generation_started']
    assert summary['selected'] == 100 and summary['pending'] == 100


def test_worker_failure_writes_traceback_without_loading_model(sample, monkeypatch):
    dataset, output, rows, config, _ = sample
    def fail(*args):
        raise RuntimeError('simulated model startup failure')
    monkeypatch.setattr(batch, 'worker', fail)
    status = output / 'worker_status/test.json'
    with pytest.raises(RuntimeError, match='startup'):
        batch.replica_entry('4,5', rows, config, str(dataset), str(output), {}, str(status))
    result = json.loads(status.read_text())
    assert result['status'] == 'failed'
    assert 'simulated model startup failure' in result['traceback']


def test_dry_run_verifies_images_but_does_not_write_output(sample, tmp_path, monkeypatch, capsys):
    dataset, _, rows, _, _ = sample
    batch.atomic_text(dataset / 'manifest.jsonl', batch.json_lines(rows))
    selection = tmp_path / 'selection.json'
    batch.write_json(selection, [{'task': row['task'], 'episode': row['episode']} for row in rows])
    model = tmp_path / 'model'
    model.mkdir()
    batch.write_json(model / 'config.json', {})
    batch.write_json(model / 'model.safetensors.index.json', {})
    prompts = tmp_path / 'prompts'
    prompts.mkdir()
    (prompts / 'ground_observer.txt').write_text('{task_instruction}')
    (prompts / 'plan_actions.txt').write_text('{task_instruction} {grounding_json}')
    output = tmp_path / 'not_created'
    monkeypatch.setattr(batch, 'SNAPSHOT_FILES', [])
    monkeypatch.setattr(sys, 'argv', ['batch', '--dataset-root', str(dataset), '--selection', str(selection),
                                    '--model', str(model), '--prompt-dir', str(prompts), '--output-root', str(output), '--dry-run'])
    batch.main()
    assert json.loads(capsys.readouterr().out)['selected'] == 3
    assert not output.exists()
    (dataset / rows[0]['outputs']['image']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='Changed initial image'):
        batch.main()


@pytest.mark.parametrize('fail_replica', [False, True])
def test_main_final_state_and_safe_resume_without_gpus(sample, tmp_path, monkeypatch, fail_replica):
    dataset, _, rows, _, records = sample
    batch.atomic_text(dataset / 'manifest.jsonl', batch.json_lines(rows))
    selection = tmp_path / 'selection.json'
    batch.write_json(selection, [{'task': row['task'], 'episode': row['episode']} for row in rows])
    model, prompts = tmp_path / 'model', tmp_path / 'prompts'
    model.mkdir()
    prompts.mkdir()
    batch.write_json(model / 'config.json', {})
    batch.write_json(model / 'model.safetensors.index.json', {})
    (prompts / 'ground_observer.txt').write_text('{task_instruction}')
    (prompts / 'plan_actions.txt').write_text('{task_instruction} {grounding_json}')
    output = tmp_path / 'run'
    monkeypatch.setattr(batch, 'SNAPSHOT_FILES', [])
    monkeypatch.setattr(sys, 'argv', ['batch', '--dataset-root', str(dataset), '--selection', str(selection),
                                    '--model', str(model), '--prompt-dir', str(prompts), '--output-root', str(output)])
    calls = []
    def fake_run(output, dataset, selected, pending, config, templates, started_at):
        calls.append(len(pending))
        if fail_replica:
            return [{'physical_gpus': '4,5', 'exit_code': 1, 'error': 'simulated fatal worker failure'}]
        for row, record in zip(selected, records):
            record['config_fingerprint'] = config['fingerprint']
            batch.write_json(batch.record_path(output, row), record)
        return []
    monkeypatch.setattr(batch, 'run_workers', fake_run)
    result = batch.main()
    summary = json.loads((output / 'summary.json').read_text())
    validation = json.loads((output / 'validation_report.json').read_text())
    if fail_replica:
        assert result == 1
        assert summary['status'] == 'failed' and not summary['processing_complete']
        assert summary['pending'] == 3
        assert not validation['all_terminal_records_present']
        assert json.loads((output / 'failure.json').read_text())['worker_failures'][0]['exit_code'] == 1
    else:
        assert result is None
        assert summary['status'] == 'completed' and summary['processing_complete']
        assert validation['all_terminal_records_present']
        assert (output / 'images').resolve() == (dataset / 'images').resolve()
        assert batch.main() is None  # Same config and exact input hashes resume without generation.
        assert calls == [3]

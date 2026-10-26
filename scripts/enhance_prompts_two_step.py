#!/usr/bin/env python3
"""Observer grounding -> structured action plan -> deterministic VPP2 rendering.

Preview only (at most 20 episodes). Source data and prior generations are kept.
An automatic check is a screening rule, never a claim of semantic correctness.
"""
from __future__ import annotations

import argparse
import base64
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import html
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys
import time

from enhance_prompts import atomic_text, check_caption, record_path, select_rows, sha256, write_json
from prompt_plan import enrich_grounding, parse_json, render_caption, resolve_assignment_modes, validate_plan
from prompt_plan import _both_evidence, _explicit_arms, _requires_handover, required_rgb_objects

PROJECT = Path(__file__).resolve().parents[1]


def split_views(image):
    if image.size != (448, 224) or image.mode != 'RGB':
        raise ValueError('Expected RGB observer-left / ego-right image at 448x224')
    return image.crop((0, 0, 224, 224)), image.crop((224, 0, 448, 224))


def unique(values):
    return list(dict.fromkeys(values))


def build_plan_prompt(template, grounding, task):
    """Make source-explicit participation constraints prominent before generation."""
    prompt = template.replace('{task_instruction}', task).replace(
        '{grounding_json}', json.dumps(grounding, ensure_ascii=False, indent=2))
    handover = _requires_handover(task)
    explicit = _explicit_arms(task)
    constraints = {'explicit_actual_arms': sorted(explicit), 'handover_requested': handover}
    instruction = ''
    if len(explicit) == 1 and not handover and not _both_evidence(task):
        active = next(iter(explicit))
        inactive = 'left' if active == 'right' else 'right'
        constraints.update(required_active_arms=[active], required_inactive_arms=[inactive])
        instruction = (f'ALL requested actions belong to the actual {active.upper()} arm. '
            f'The actual {inactive.upper()} arm MUST have active=false, actions=[], '
            'assignment.mode="inactive", assignment.object_id=null, verb_phrase="keeps static". '
            'The illustrative schema activity pattern above must be replaced accordingly. ')
    if not handover:
        instruction += 'Do not invent an arm-to-arm handover, transfer, receive, or assistance action. '
    required_colors = required_rgb_objects(task)
    if required_colors:
        constraints['required_manipulation_targets'] = [
            {'task_object': color + ' block', 'grounding_object_ids': [
                obj['id'] for obj in grounding.get('objects', [])
                if isinstance(obj, dict) and isinstance(obj.get('id'), str)
                and re.search(r'\b' + color + r'\b', str(obj.get('label', '')), re.I)
                and re.search(r'\b(?:block|cube)s?\b', str(obj.get('label', '')), re.I)]}
            for color in required_colors]
        instruction += ('The original task explicitly requires manipulating ALL THREE listed blocks. '
            'Plan a concrete manipulation for each required object_id and describe that operation '
            'in its owning arm\'s verb_phrase. Each arm can operate multiple blocks in sequence. '
            'A final list of colors, destination reference, or requirements entry is NOT a substitute '
            'for operating on the third block. Retain all original per-object arm assignments, '
            'requested relative positions, and explicit action dependencies. ')
    return (prompt + '\n\nTASK-SPECIFIC CONSTRAINTS DERIVED FROM THE ORIGINAL INSTRUCTION:\n'
        + json.dumps(constraints, ensure_ascii=False) + '\n' + instruction
        + '\nReturn the complete plan JSON, respecting these actual-arm constraints.')


def normalize_observer_pixels(raw):
    """Qwen2.5-VL grounds in actual resized image pixels, not 0..1000."""
    if raw.get('coordinate_system') != 'observer_pixels_896x896':
        raise ValueError('Grounding must explicitly use observer 896x896 pixel coordinates')
    normalized = copy.deepcopy(raw)
    for obj in normalized.get('objects', []):
        center = obj.get('center')
        obj['observer_center_px'] = center
        if center is None:
            continue
        if not isinstance(center, list) or len(center) != 2 or any(
            type(v) not in (float, int) or not 0 <= v <= 896 for v in center):
            raise ValueError('Observer coordinates outside the actual 896x896 image')
        obj['center'] = [round(v * 1000 / 896, 3) for v in center]
    normalized['coordinate_system'] = 'observer_normalized_0_1000_by_program'
    return normalized


def worker(gpus, rows, config, dataset_str, output_str, templates):
    os.environ.update(CUDA_VISIBLE_DEVICES=gpus, CUDA_DEVICE_ORDER='PCI_BUS_ID',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    dataset, output = Path(dataset_str), Path(output_str)
    with (output / f'worker_{gpus.replace(",", "_")}.log').open('a', buffering=1) as log:
        sys.stdout = log
        sys.stderr = log
        import torch
        import transformers
        from PIL import Image
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        torch.set_num_threads(4)
        torch.manual_seed(0)
        if torch.cuda.device_count() != 2:
            raise RuntimeError('Exactly two visible GPUs required per replica')
        print(f'Loading Qwen BF16 on physical GPUs {gpus}', flush=True)
        started = time.monotonic()
        processor = AutoProcessor.from_pretrained(config['model'], local_files_only=True,
            min_pixels=448 * 448, max_pixels=896 * 896)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config['model'],
            local_files_only=True, dtype=torch.bfloat16, device_map='balanced',
            max_memory={0: '120GiB', 1: '120GiB'}, attn_implementation='sdpa')
        model.eval()
        model.generation_config.temperature = 1.0
        if any(str(v) in {'cpu', 'disk'} for v in model.hf_device_map.values()):
            raise RuntimeError('Unexpected model CPU/disk offload')
        write_json(output / f'runtime_{gpus.replace(",", "_")}.json', {
            'physical_gpus': gpus, 'torch': torch.__version__, 'transformers': transformers.__version__,
            'device_map': {k: str(v) for k, v in model.hf_device_map.items()},
            'load_seconds': round(time.monotonic() - started, 3)})
        print('Model loaded', flush=True)

        def generate(prompt, pil_images, max_tokens):
            begin = time.monotonic()
            content = [{'type': 'image'} for _ in pil_images] + [{'type': 'text', 'text': prompt}]
            messages = [{'role': 'user', 'content': content}]
            chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            kwargs = {'images': pil_images} if pil_images else {}
            inputs = processor(text=[chat], padding=True, return_tensors='pt', **kwargs)
            grid = inputs['image_grid_thw'].tolist() if 'image_grid_thw' in inputs else []
            inputs = inputs.to(model.device)
            n_input = inputs.input_ids.shape[1]
            with torch.inference_mode():
                tokens = model.generate(**inputs, do_sample=False, max_new_tokens=max_tokens, use_cache=True)
            new_tokens = tokens[0, n_input:]
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            result = {'raw_response': processor.decode(new_tokens, skip_special_tokens=True,
                clean_up_tokenization_spaces=False), 'eos_terminated': bool(len(new_tokens) and new_tokens[-1].item() in eos),
                'input_tokens': n_input, 'generated_tokens': len(new_tokens), 'image_grid_thw': grid,
                'elapsed_seconds': round(time.monotonic() - begin, 3)}
            del inputs, tokens, new_tokens
            return result

        for row in rows:
            begin = time.monotonic()
            image_path = dataset / row['outputs']['image']
            record = {'task': row['task'], 'episode': row['episode'], 'image': row['outputs']['image'],
                'image_sha256': sha256(image_path), 'original_instruction': row['instruction'],
                'instruction_source': row['inputs']['instruction'], 'instruction_key': row['instruction_key'],
                'instruction_index': row['instruction_index'], 'physical_gpus': gpus,
                'config_fingerprint': config['fingerprint'], 'grounding': None, 'plan': None,
                'grounding_generation': None, 'plan_attempts': [], 'enhanced_instruction': '',
                'review_flags': [], 'semantic_review': 'pending_human_review'}
            try:
                with Image.open(image_path) as im:
                    observer, ego = split_views(im)
                view_dir = output / 'views' / row['task']
                view_dir.mkdir(parents=True, exist_ok=True)
                for name, im in [('observer', observer), ('ego', ego)]:
                    path = view_dir / f'{row["episode"]}_{name}.png'
                    im.save(path)
                    record[f'{name}_crop'] = str(path.relative_to(output))
                    record[f'{name}_crop_sha256'] = sha256(path)
                prompt = templates['ground_observer'].replace('{task_instruction}', row['instruction'])
                grounding_generation = generate(prompt, [observer.resize((896, 896), Image.Resampling.BICUBIC),
                    ego.resize((448, 448), Image.Resampling.BICUBIC)], config['ground_max_tokens'])
                record['grounding_generation'] = grounding_generation
                ground_raw = parse_json(grounding_generation['raw_response'])
                grounding, ground_flags = enrich_grounding(normalize_observer_pixels(ground_raw))
                record['grounding'] = grounding
                if not grounding_generation['eos_terminated']:
                    ground_flags.append('grounding_generation_truncated')
                prompt = build_plan_prompt(templates['plan_actions'], grounding, row['instruction'])
                feedback = ''
                for attempt in range(config['plan_attempts']):
                    generated = generate(prompt + feedback, [], config['plan_max_tokens'])
                    generated['attempt'] = attempt + 1
                    generated['feedback_supplied'] = feedback
                    plan, caption = None, ''
                    flags = []
                    try:
                        plan = resolve_assignment_modes(parse_json(generated['raw_response']), grounding, row['instruction'])
                        flags.extend(validate_plan(plan, grounding, row['instruction']))
                        caption = render_caption(plan)
                        _, format_flags = check_caption(caption)
                        flags.extend(format_flags)
                    except (ValueError, TypeError, KeyError, AttributeError) as exc:
                        flags.append(f'invalid_plan: {exc}')
                    if not generated['eos_terminated']:
                        flags.append('plan_generation_truncated')
                    generated['plan'] = plan
                    generated['validation_flags'] = unique(flags)
                    generated['rendered_caption'] = caption
                    record['plan_attempts'].append(generated)
                    record['plan'] = plan
                    record['enhanced_instruction'] = caption
                    record['review_flags'] = unique(ground_flags + flags)
                    # A plan retry cannot repair observer grounding; retain such
                    # flags for review without encouraging invented coordinates.
                    repairable_flags = [flag for flag in flags if flag not in ground_flags]
                    if not repairable_flags:
                        break
                    feedback = ('\n\nYour previous candidate below failed checks. Correct these issues using only '
                        'the original task, fixed observer grounding, and schema. Do not invent missing evidence. '
                        'Return the entire corrected JSON. If uncertainty cannot be resolved, retain it.\n'
                        'Check findings: ' + json.dumps(unique(repairable_flags), ensure_ascii=False) + '\n'
                        'Previous candidate: ' + generated['raw_response'])
            except Exception as exc:
                record['review_flags'] = unique(record['review_flags'] + [f'pipeline_error: {type(exc).__name__}: {exc}'])
            record['elapsed_seconds'] = round(time.monotonic() - begin, 3)
            write_json(record_path(output, row), record)
            print(json.dumps({'task': row['task'], 'episode': row['episode'],
                'seconds': record['elapsed_seconds'], 'attempts': len(record['plan_attempts']),
                'flags': record['review_flags'], 'caption': record['enhanced_instruction']}, ensure_ascii=False), flush=True)
    return len(rows)


def data_image(path):
    return 'data:image/png;base64,' + base64.b64encode(Path(path).read_bytes()).decode('ascii')


def export_review(output, dataset, rows, config):
    results = [json.loads(record_path(output, row).read_text()) for row in rows]
    write_json(output / 'results.json', results)
    image_link = output / 'images'
    if image_link.is_symlink():
        if image_link.resolve() != (dataset / 'images').resolve():
            raise ValueError('Wrong images symlink')
    elif image_link.exists():
        raise ValueError('Unexpected existing output images directory')
    else:
        image_link.symlink_to(dataset / 'images', target_is_directory=True)
    atomic_text(output / 'text_original.txt', ''.join(f"{r['original_instruction']}@@{r['image']}\n" for r in results))
    screened = [r for r in results if r['enhanced_instruction'] and not r['review_flags']]
    atomic_text(output / 'text_prompt_enhanced.txt', ''.join(f"{r['enhanced_instruction']}@@{r['image']}\n" for r in screened))
    atomic_text(output / 'text_candidates_all.txt', ''.join(f"{r['enhanced_instruction']}@@{r['image']}\n" for r in results if r['enhanced_instruction']))
    previous = {}
    previous_path = PROJECT / 'runs/prompt_preview_v2/results.json'
    if previous_path.exists():
        previous = {(r['task'], r['episode']): r for r in json.loads(previous_path.read_text())}
    cards, markdown = [], ['# 两步 prompt 样本审核', '',
        '只运行12条已选样本，所有结果仍待用户审核。程序检查通过不等于语义正确。', '']
    for i, r in enumerate(results, 1):
        title = f"{i:02d}. {r['task']} / {r['episode']}"
        flags = '\n'.join(r['review_flags']) or '自动检查未发现问题；仍待语义审核'
        prior = previous.get((r['task'], r['episode']), {}).get('enhanced_instruction', '')
        ground = r.get('grounding') or {'objects': []}
        table = '<table><tr><th>物体</th><th>observer 坐标</th><th>画面位置</th><th>机械臂参考</th><th>置信度</th></tr>'
        for obj in ground.get('objects', []):
            table += '<tr>' + ''.join('<td>' + html.escape(str(obj.get(k))) + '</td>'
                for k in ['label', 'center', 'observer_side', 'preferred_arm', 'confidence']) + '</tr>'
        table += '</table>'
        planning = json.dumps(r.get('plan'), ensure_ascii=False, indent=2)
        grounding_img = ('<img width="448" src="' + data_image(output / r['observer_crop']) + '" alt="observer crop">') if r.get('observer_crop') else ''
        cards.append(f'<article><h2>{html.escape(title)}</h2><img width="672" src="{data_image(dataset / r["image"])}" alt="original composed frame">'
            f'<h3>原任务</h3><p>{html.escape(r["original_instruction"])}</p>'
            f'<details><summary>第一步：observer 单图定位</summary>{grounding_img}{table}<pre>{html.escape(json.dumps(ground,ensure_ascii=False,indent=2))}</pre></details>'
            f'<details><summary>第二步：动作计划（{len(r["plan_attempts"])} 次尝试）</summary><pre>{html.escape(planning)}</pre></details>'
            f'<h3>本轮输出</h3><p>{html.escape(r["enhanced_instruction"] or "未生成可渲染文本")}</p>'
            f'<p class="status">{html.escape(flags)}</p>'
            f'<details><summary>上一轮单步输出</summary><p>{html.escape(prior)}</p></details></article>')
        markdown += [f'## {title}', '', f'![首帧]({r["image"]})', '',
            '原任务：' + r['original_instruction'], '', '本轮：' + r['enhanced_instruction'], '',
            '检查：' + flags, '', '上轮：' + prior, '', '```json', planning, '```', '']
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>两步 prompt 审核</title>
<style>body{font:17px/1.6 system-ui;max-width:1080px;margin:30px auto;padding:0 20px;background:#f4f5f7;color:#182332}
article{background:white;border:1px solid #dce0e6;border-radius:10px;padding:24px;margin:24px 0}img{max-width:100%;height:auto}
h2{font-size:21px}h3{font-size:17px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}details{margin:12px 0}
td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse;font-size:14px}.status{background:#fff5dc;padding:12px;white-space:pre-wrap}</style>
<h1>两步 prompt 样本审核</h1><p>Qwen2.5-VL-72B BF16 · GPU 4–7 · 同12条样本 · 100条及全量未运行</p>
<p>第一步用 observer 裁图定位、ego 补充识别，坐标只取 observer；第二步仅用原任务和固定定位/选臂参考生成结构化计划。
程序检查后，按实际右臂→实际左臂拼接句子。所有模型原始回答和重试记录均保留。坐标与动作计划也可能有误，需对图审核。</p>'''
    atomic_text(output / 'review.html', page + '\n'.join(cards) + '</html>\n')
    atomic_text(output / 'review.md', '\n'.join(markdown))
    summary = {'selected': len(results), 'caption_generated': sum(bool(r['enhanced_instruction']) for r in results),
        'automatic_checks_passed': len(screened), 'needs_review_by_checks': len(results) - len(screened),
        'human_approval': 'pending', 'mode': 'preview', 'full_generation_started': False,
        'model': config['model'], 'gpu_groups': config['gpu_groups'],
        'excluded_from_enhanced_file': [{'task': r['task'], 'episode': r['episode'], 'flags': r['review_flags']}
            for r in results if r not in screened]}
    write_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=PROJECT / 'data/robotwin_observer_ego_vpp2')
    parser.add_argument('--selection', type=Path, default=PROJECT / 'prompts/preview_selection.json')
    parser.add_argument('--model', type=Path, default=Path('/data2/models/Qwen2.5-VL-72B-Instruct'))
    parser.add_argument('--prompt-dir', type=Path, default=PROJECT / 'prompts/two_step')
    parser.add_argument('--output-root', type=Path, default=PROJECT / 'runs/prompt_two_step_v2')
    parser.add_argument('--ground-max-tokens', type=int, default=768)
    parser.add_argument('--plan-max-tokens', type=int, default=2048)
    parser.add_argument('--plan-attempts', type=int, choices=[1, 2], default=2)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    dataset, output = args.dataset_root.resolve(), args.output_root.resolve()
    if output == dataset or output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError('Preview output must be separate from dataset')
    if args.ground_max_tokens < 1 or args.plan_max_tokens < 1:
        raise ValueError('Token limits must be positive')
    rows = select_rows(dataset, args.selection, False)
    templates = {name: (args.prompt_dir / f'{name}.txt').read_text() for name in ['ground_observer', 'plan_actions']}
    for name, template in templates.items():
        if template.count('{task_instruction}') != 1:
            raise ValueError(f'Invalid task placeholder in {name}')
    if templates['plan_actions'].count('{grounding_json}') != 1:
        raise ValueError('Missing grounding placeholder')
    config = {'model': str(args.model.resolve()), 'model_config_sha256': sha256(args.model / 'config.json'),
        'model_index_sha256': sha256(args.model / 'model.safetensors.index.json'),
        'script_sha256': sha256(__file__), 'plan_validator_sha256': sha256(Path(__file__).with_name('prompt_plan.py')),
        'base_helpers_sha256': sha256(Path(__file__).with_name('enhance_prompts.py')),
        'templates_sha256': {name: hashlib.sha256(value.encode()).hexdigest() for name, value in templates.items()},
        'dataset': str(dataset), 'mode': 'preview', 'gpu_groups': ['4,5', '6,7'], 'dtype': 'bfloat16',
        'device_map': 'balanced', 'attention': 'sdpa', 'do_sample': False,
        'ground_max_tokens': args.ground_max_tokens, 'plan_max_tokens': args.plan_max_tokens,
        'plan_attempts': args.plan_attempts, 'vlm_ground_observer_size': [896, 896],
        'vlm_ground_ego_size': [448, 448], 'plan_input': 'text_only', 'original_view_size': [224, 224],
        'selection': [{'task': r['task'], 'episode': r['episode'], 'image': r['outputs']['image'],
            'image_sha256': r['output_sha256']['image'], 'instruction': r['instruction']} for r in rows]}
    config['fingerprint'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.generation.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg_path = output / 'run_config.json'
        if cfg_path.exists() and json.loads(cfg_path.read_text())['fingerprint'] != config['fingerprint']:
            raise ValueError('Different configuration in output; choose a new directory')
        write_json(cfg_path, config)
        for name, template in templates.items():
            atomic_text(output / 'prompts' / f'{name}.txt', template)
        for filename in ['enhance_prompts_two_step.py', 'prompt_plan.py', 'enhance_prompts.py']:
            atomic_text(output / 'source_snapshot' / filename, Path(__file__).with_name(filename).read_text())
        pending = []
        for row in rows:
            path = record_path(output, row)
            if path.exists():
                old = json.loads(path.read_text())
                if old['config_fingerprint'] != config['fingerprint'] or old['image_sha256'] != row['output_sha256']['image']:
                    raise ValueError('Existing result provenance mismatch')
            else:
                pending.append(row)
        print(f'Selected {len(rows)} preview episodes; pending {len(pending)}', flush=True)
        if pending:
            with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn')) as pool:
                futures = [pool.submit(worker, group, pending[i::2], config, str(dataset), str(output), templates)
                    for i, group in enumerate(config['gpu_groups']) if pending[i::2]]
                for future in as_completed(futures):
                    print(f'Replica completed {future.result()} episodes', flush=True)
        print(json.dumps(export_review(output, dataset, rows, config), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

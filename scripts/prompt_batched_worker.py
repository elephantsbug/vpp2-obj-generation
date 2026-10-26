"""Small-batch Qwen inference using the installed two-step grounding protocol.

The public worker signature and per-episode records match the serial worker.
Images are flattened in sample order; decoder outputs are trimmed after the
entire padded input, then validated for EOS followed only by generation padding.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import traceback

from enhance_prompts import check_caption, record_path, sha256, write_json
from enhance_prompts_two_step import build_plan_prompt, normalize_observer_pixels, split_views, unique
from prompt_plan import enrich_grounding, parse_json, render_caption, validate_plan


def split_image_grids(flat_grids, image_counts):
    """Assign processor image grids back to samples without reordering images."""
    if any(type(count) is not int or count < 0 for count in image_counts) or sum(image_counts) != len(flat_grids):
        raise ValueError('Image-grid count differs from the supplied per-sample images')
    result, offset = [], 0
    for count in image_counts:
        result.append(flat_grids[offset:offset + count])
        offset += count
    return result


def trim_generated_tokens(sequence, padded_input_length, eos_token_ids, pad_token_id):
    """Keep all real generated tokens, including the first EOS.

    Padding before the input is not generated output. Padding after a generated
    EOS is removed only after verifying the entire remainder is padding. A
    non-padding token after EOS is an error, never silently discarded.
    """
    if type(padded_input_length) is not int or not 0 <= padded_input_length <= len(sequence):
        raise ValueError('Invalid padded input length')
    tokens = list(sequence[padded_input_length:])
    eos = {eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids or [])
    position = next((index for index, token in enumerate(tokens) if token in eos), None)
    if position is None:
        return {'token_ids': tokens, 'eos_terminated': False, 'generated_tokens': len(tokens),
                'batch_generated_length': len(tokens), 'padding_after_eos': 0, 'eos_token_id': None}
    tail = tokens[position + 1:]
    if any(token != pad_token_id for token in tail):
        raise ValueError('Non-padding generated token after EOS; refusing to discard real output')
    return {'token_ids': tokens[:position + 1], 'eos_terminated': True, 'generated_tokens': position + 1,
            'batch_generated_length': len(tokens), 'padding_after_eos': len(tail), 'eos_token_id': tokens[position]}


def decode_batch_output(processor, input_ids, attention_masks, sequences, eos_token_ids, pad_token_id,
                        flat_grids, image_counts, elapsed_seconds, batch_key):
    """Check sequence correspondence, trim correctly, and decode in input order."""
    count = len(input_ids)
    if not count or any(len(value) != count for value in (attention_masks, sequences, image_counts)):
        raise ValueError('Batch output size differs from batch input size')
    padded_length = len(input_ids[0])
    grids = split_image_grids(flat_grids, image_counts)
    results = []
    for index, (source, mask, sequence) in enumerate(zip(input_ids, attention_masks, sequences, strict=True)):
        if len(source) != padded_length or len(mask) != padded_length or list(sequence[:padded_length]) != list(source):
            raise ValueError('Generated sequence does not preserve its padded input prefix')
        if not mask or mask[-1] != 1 or any(value not in (0, 1) for value in mask) or list(mask) != sorted(mask):
            raise ValueError('A decoder batch must use contiguous left padding and nonempty inputs')
        result = trim_generated_tokens(sequence, padded_length, eos_token_ids, pad_token_id)
        result.update(input_tokens=sum(mask), padded_input_tokens=padded_length,
                      image_grid_thw=grids[index], elapsed_seconds=round(elapsed_seconds, 3),
                      elapsed_seconds_scope='shared_batch_generation', batch_size=count,
                      batch_member_index=index, batch_key=batch_key)
        results.append(result)
    decoded = processor.batch_decode([result['token_ids'] for result in results], skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
    if len(decoded) != count:
        raise ValueError('Decoder returned the wrong number of responses')
    for result, text in zip(results, decoded, strict=True):
        result['raw_response'] = text
    return results


class BatchGenerator:
    def __init__(self, processor, model, torch_module):
        self.processor, self.model, self.torch = processor, model, torch_module
        self.processor.tokenizer.padding_side = 'left'
        if self.processor.tokenizer.pad_token_id is None:
            raise ValueError('Tokenizer must have an explicit padding token')
        self.eos_token_ids = self.model.generation_config.eos_token_id
        if self.eos_token_ids is None:
            raise ValueError('Model must have explicit EOS token IDs')
        self.pad_token_id = self.model.generation_config.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.processor.tokenizer.pad_token_id
            self.model.generation_config.pad_token_id = self.pad_token_id

    def generate(self, prompts, images_by_sample, max_tokens, batch_key):
        if not prompts or len(prompts) != len(images_by_sample):
            raise ValueError('Each prompt needs its own image list, possibly empty')
        started = time.monotonic()
        chats, flat_images = [], []
        for prompt, images in zip(prompts, images_by_sample, strict=True):
            content = [{'type': 'image'} for _ in images] + [{'type': 'text', 'text': prompt}]
            chats.append(self.processor.apply_chat_template([{'role': 'user', 'content': content}],
                         tokenize=False, add_generation_prompt=True))
            flat_images.extend(images)
        kwargs = {'images': flat_images} if flat_images else {}
        inputs = self.processor(text=chats, padding=True, return_tensors='pt', **kwargs)
        source_ids = inputs['input_ids'].tolist()
        masks = inputs['attention_mask'].tolist()
        flat_grids = inputs['image_grid_thw'].tolist() if 'image_grid_thw' in inputs else []
        image_counts = [len(images) for images in images_by_sample]
        split_image_grids(flat_grids, image_counts)  # Reject mismatched image assignment before generation.
        inputs = inputs.to(self.model.device)
        sequences = None
        try:
            with self.torch.inference_mode():
                sequences = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_tokens, use_cache=True,
                                                pad_token_id=self.pad_token_id)
            if hasattr(sequences, 'sequences'):
                sequences = sequences.sequences
            return decode_batch_output(self.processor, source_ids, masks, sequences.detach().cpu().tolist(),
                                       self.eos_token_ids, self.pad_token_id, flat_grids, image_counts,
                                       time.monotonic() - started, batch_key)
        finally:
            del inputs, sequences


def new_record(row, config, dataset, gpus, batch_index, member_index, actual_batch_size):
    return {'task': row['task'], 'episode': row['episode'], 'image': row['outputs']['image'],
            'image_sha256': sha256(dataset / row['outputs']['image']), 'original_instruction': row['instruction'],
            'instruction_source': row['inputs']['instruction'], 'instruction_key': row['instruction_key'],
            'instruction_index': row['instruction_index'], 'physical_gpus': gpus,
            'config_fingerprint': config['fingerprint'], 'grounding': None, 'plan': None,
            'grounding_generation': None, 'plan_attempts': [], 'enhanced_instruction': '',
            'review_flags': [], 'semantic_review': 'pending_human_review',
            'batch': {'configured_batch_size': config['batch_size'], 'actual_batch_size': actual_batch_size,
                      'replica_batch_index': batch_index, 'member_index': member_index, 'worker_mode': 'microbatch'}}


def pipeline_error(state, exc):
    state['failed'] = True
    record = state['record']
    record['review_flags'] = unique(record['review_flags'] + [f'pipeline_error: {type(exc).__name__}: {exc}'])


def apply_plan_generation(state, generated, attempt):
    """Apply exactly the serial worker's parse/validate/render/retry policy."""
    from prompt_plan import resolve_assignment_modes
    record, grounding = state['record'], state['record']['grounding']
    generated['attempt'] = attempt
    generated['feedback_supplied'] = state['feedback']
    plan, caption, flags = None, '', []
    try:
        plan = resolve_assignment_modes(parse_json(generated['raw_response']), grounding, state['row']['instruction'])
        flags.extend(validate_plan(plan, grounding, state['row']['instruction']))
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
    record['review_flags'] = unique(state['ground_flags'] + flags)
    repairable = [flag for flag in flags if flag not in state['ground_flags']]
    if repairable:
        state['feedback'] = ('\n\nYour previous candidate below failed checks. Correct these issues using only '
                             'the original task, fixed observer grounding, and schema. Do not invent missing evidence. '
                             'Return the entire corrected JSON. If uncertainty cannot be resolved, retain it.\n'
                             'Check findings: ' + json.dumps(unique(repairable), ensure_ascii=False) + '\n'
                             'Previous candidate: ' + generated['raw_response'])
    return bool(repairable)


def process_batch(rows, config, dataset, output, templates, gpus, batch_index, generator):
    from PIL import Image
    states = []
    for member_index, row in enumerate(rows):
        state = {'row': row, 'started': time.monotonic(), 'failed': False, 'ground_flags': [], 'feedback': '',
                 'record': new_record(row, config, dataset, gpus, batch_index, member_index, len(rows))}
        states.append(state)
        try:
            with Image.open(dataset / row['outputs']['image']) as im:
                observer, ego = split_views(im)
            view_dir = output / 'views' / row['task']
            view_dir.mkdir(parents=True, exist_ok=True)
            for name, im in [('observer', observer), ('ego', ego)]:
                path = view_dir / f'{row["episode"]}_{name}.png'
                im.save(path)
                state['record'][f'{name}_crop'] = str(path.relative_to(output))
                state['record'][f'{name}_crop_sha256'] = sha256(path)
            state['images'] = [observer.resize((896, 896), Image.Resampling.BICUBIC),
                               ego.resize((448, 448), Image.Resampling.BICUBIC)]
            state['ground_prompt'] = templates['ground_observer'].replace('{task_instruction}', row['instruction'])
        except Exception as exc:
            pipeline_error(state, exc)
    prepared = [state for state in states if not state['failed']]
    stage = 'ground'
    try:
        if prepared:
            generations = generator.generate([state['ground_prompt'] for state in prepared], [state['images'] for state in prepared],
                                             config['ground_max_tokens'], f'{gpus}:{batch_index}:ground')
            for state, generated in zip(prepared, generations, strict=True):
                state['record']['grounding_generation'] = generated
                try:
                    ground_raw = parse_json(generated['raw_response'])
                    grounding, ground_flags = enrich_grounding(normalize_observer_pixels(ground_raw))
                    state['record']['grounding'] = grounding
                    if not generated['eos_terminated']:
                        ground_flags.append('grounding_generation_truncated')
                    state['ground_flags'] = ground_flags
                    state['plan_prompt'] = build_plan_prompt(templates['plan_actions'], grounding, state['row']['instruction'])
                except Exception as exc:
                    pipeline_error(state, exc)
        active = [state for state in states if not state['failed']]
        for attempt in range(1, config['plan_attempts'] + 1):
            if not active:
                break
            stage = f'plan_attempt_{attempt}'
            generations = generator.generate([state['plan_prompt'] + state['feedback'] for state in active], [[] for _ in active],
                                             config['plan_max_tokens'], f'{gpus}:{batch_index}:{stage}')
            retry = []
            for state, generated in zip(active, generations, strict=True):
                try:
                    if apply_plan_generation(state, generated, attempt):
                        retry.append(state)
                except Exception as exc:
                    pipeline_error(state, exc)
            active = retry  # Only candidates with repairable failures enter the next batch.
    except Exception as exc:
        failure = {'physical_gpus': gpus, 'batch_index': batch_index, 'stage': stage,
                   'batch_size': len(rows), 'configured_batch_size': config['batch_size'],
                   'config_fingerprint': config['fingerprint'], 'error': f'{type(exc).__name__}: {exc}',
                   'traceback': traceback.format_exc(), 'affected': [{'task': row['task'], 'episode': row['episode']} for row in rows],
                   'partial_records': [state['record'] for state in states],
                   'note': 'Batch generation failed; this batch is not published as terminal. Use a new output directory with --batch-size 1 if memory is insufficient.'}
        write_json(output / 'batch_failures' / f'{gpus.replace(",", "_")}_{os.getpid()}_{batch_index:05d}.json', failure)
        raise
    for state in states:
        record = state['record']
        record['elapsed_seconds'] = round(time.monotonic() - state['started'], 3)
        record['elapsed_seconds_scope'] = 'microbatch_latency_for_episode'
        write_json(record_path(output, state['row']), record)
        print(json.dumps({'task': record['task'], 'episode': record['episode'], 'batch': record['batch'],
                          'seconds': record['elapsed_seconds'], 'attempts': len(record['plan_attempts']),
                          'flags': record['review_flags'], 'caption': record['enhanced_instruction']}, ensure_ascii=False), flush=True)
    return len(states)


def worker(gpus, rows, config, dataset_str, output_str, templates):
    # Check the required validator revision before loading the model.
    from prompt_plan import resolve_assignment_modes
    os.environ.update(CUDA_VISIBLE_DEVICES=gpus, CUDA_DEVICE_ORDER='PCI_BUS_ID', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    dataset, output = Path(dataset_str), Path(output_str)
    batch_size = config.get('batch_size', 4)
    if batch_size not in (2, 4):
        raise ValueError('Batched worker requires batch_size 2 or 4; select the serial worker for 1')
    with (output / f'worker_{gpus.replace(",", "_")}.log').open('a', buffering=1) as log:
        sys.stdout = log
        sys.stderr = log
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        torch.set_num_threads(4)
        torch.manual_seed(0)
        if torch.cuda.device_count() != 2:
            raise RuntimeError('Exactly two visible GPUs required per replica')
        print(f'Loading Qwen BF16 on physical GPUs {gpus}, microbatch {batch_size}', flush=True)
        started = time.monotonic()
        processor = AutoProcessor.from_pretrained(config['model'], local_files_only=True,
                                                 min_pixels=448 * 448, max_pixels=896 * 896)
        processor.tokenizer.padding_side = 'left'
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config['model'], local_files_only=True, dtype=torch.bfloat16,
                    device_map='balanced', max_memory={0: '120GiB', 1: '120GiB'}, attn_implementation='sdpa')
        model.eval()
        model.generation_config.temperature = 1.0
        if any(str(value) in {'cpu', 'disk'} for value in model.hf_device_map.values()):
            raise RuntimeError('Unexpected model CPU/disk offload')
        generator = BatchGenerator(processor, model, torch)
        write_json(output / f'runtime_{gpus.replace(",", "_")}.json', {'physical_gpus': gpus, 'torch': torch.__version__,
                   'transformers': transformers.__version__, 'device_map': {key: str(value) for key, value in model.hf_device_map.items()},
                   'load_seconds': round(time.monotonic() - started, 3), 'batch_size': batch_size, 'padding_side': 'left',
                   'generation_eos_token_ids': generator.eos_token_ids, 'generation_pad_token_id': generator.pad_token_id})
        print('Model loaded', flush=True)
        count = 0
        for batch_index, offset in enumerate(range(0, len(rows), batch_size)):
            count += process_batch(rows[offset:offset + batch_size], config, dataset, output, templates, gpus, batch_index, generator)
        return count

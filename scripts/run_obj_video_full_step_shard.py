#!/usr/bin/env python3
"""One GPU shard of a frozen full-test 6/4-step paired comparison."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import os
from pathlib import Path
import time
import traceback

from run_obj_video_low_step_sweep import (
    ROOT, MEMORY, atomic_json, digest, require, resident_model, jsonable,
    load_config, make_dataset, window_metadata, write_video,
    now, load, input_signature, verify_sample, tensor_sha, decoded_video,
)

STEPS = (6, 4)


def validate_plan(path):
    plan = load(path)
    require(plan['schema'] == 'vpp2_obj.full_sampling_step_selection.v1' and plan['status'] == 'frozen', 'Wrong plan')
    require(plan['num_inference_steps'] == list(STEPS) and plan['num_shards'] == 4, 'Wrong steps/shards')
    require(plan['seed'] == 0 and plan['sigma_shift'] == 3., 'Wrong seed/shift')
    cases = plan['cases']
    require(len(cases) == plan['case_count'] == 480 and len({c['id'] for c in cases}) == 480, 'Expected480 records')
    require(len({(c['task'], c['episode'], c['anchor']) for c in cases}) == 464, 'Expected464 unique windows')
    require(len({(c['task'], c['episode']) for c in cases}) == 240 and len({c['task'] for c in cases}) == 50, 'Wrong test scope')
    require(digest(plan['parent_plan']) == plan['parent_plan_sha256'], 'Original full selection changed')
    original = load(plan['parent_plan'])['cases']
    strip = lambda c: {k: v for k, v in c.items() if k not in ('steps', 'sampling_steps', 'variant_order', 'shard_index')}
    require([strip(c) for c in cases] == [strip(c) for c in original], 'Full window selection changed')
    groups, phases = defaultdict(Counter), defaultdict(set)
    for c in cases:
        require(Path(c['id']).name == c['id'] and c['id'] not in ('.', '..'), 'Unsafe id')
        require(c['sampling_steps'] == list(STEPS) and sorted(c['variant_order']) == [4, 6], 'Invalid variants')
        require(c['seed'] == 0 and c['inference_eligible'] is True, 'Invalid inference case')
        require(c['shard_index'] == c['episode_index'] % 4, 'Episode shard changed')
        require(type(c['phase_eligible']) is bool and (c['fallback_reason'] is None) == c['phase_eligible'], 'Invalid fallback')
        groups[(c['shard_index'], c['phase'])][tuple(c['variant_order'])] += 1
        phases[(c['task'], c['episode'])].add(c['phase'])
    require(all(p == {'pre_onset', 'mid_motion'} for p in phases.values()), 'Missing episode phase')
    require(set(groups) == {(s,p) for s in range(4) for p in ('pre_onset','mid_motion')}, 'Missing shard/phase')
    require(all(v == Counter({(6,4):30,(4,6):30}) for v in groups.values()), 'Unbalanced paired order')
    for key, sha_key in (('config','config_sha256'), ('test_manifest','test_manifest_sha256'), ('prior_24_plan','prior_24_plan_sha256')):
        require(digest(plan[key]) == plan[sha_key], f'Changed {key}')
    prior = load(plan['prior_24_plan'])
    old_run = Path(plan['prior_24_run'])
    require(digest(old_run/'run.json') == plan['prior_24_run_sha256'], 'Prior24 run changed')
    prior_run = load(old_run/'run.json')
    require(prior_run['status'] == 'complete' and prior_run['plan_sha256'] == plan['prior_24_plan_sha256'], 'Wrong prior24 run')
    require(len(prior['cases']) == 24 and {c['id'] for c in prior['cases']} <= {c['id'] for c in cases}, 'Missing prior windows')
    checkpoint = Path(plan['checkpoint']).resolve()
    stat = checkpoint.stat()
    require(checkpoint.name == 'step_010000.pt' and (stat.st_size,stat.st_mtime_ns) ==
            (plan['checkpoint_stat']['size_bytes'],plan['checkpoint_stat']['mtime_ns']), 'Checkpoint changed')
    return plan


def identity(plan, plan_path):
    checkpoint = Path(plan['checkpoint']).resolve()
    stat = checkpoint.stat()
    return {'config':str(Path(plan['config']).resolve()),'config_sha256':plan['config_sha256'],
            'manifest':str(Path(plan['test_manifest']).resolve()),'manifest_sha256':plan['test_manifest_sha256'],
            'checkpoint':str(checkpoint),'checkpoint_bytes':stat.st_size,'checkpoint_mtime_ns':stat.st_mtime_ns,
            'plan_sha256':digest(plan_path),'runner_sha256':digest(__file__),
            'runner_helper_sha256':digest(Path(__file__).with_name('run_obj_video_low_step_sweep.py')),
            'resident_loader_sha256':digest(Path(__file__).with_name('run_obj_video_eval_shard.py')),
            'seed':0,'sigma_shift':3.,'memory_strategy':MEMORY,
            'prior_24_plan_sha256':plan['prior_24_plan_sha256'],'prior_24_run_sha256':plan['prior_24_run_sha256']}


def replay_previous(plan, case, trace, video):
    import numpy as np
    directory = Path(plan['prior_24_run'])/'samples'/case['id']/f"steps_{trace['num_inference_steps']}"
    old = load(directory/'trace.json')
    fields = ('checkpoint','checkpoint_bytes','checkpoint_mtime_ns','config_sha256','manifest_sha256',
              'task','episode','anchor_index','sampling','seed','sigma_shift','num_inference_steps',
              'memory_strategy','prompt','text_embedding_sha256','context_mask','condition_shape',
              'decoded_frame_slice','prefix_clamped','input_tensor_sha256','initial_noise_sha256','initial_noise_shape')
    require(all(old[k] == trace[k] for k in fields), 'Prior24 replay settings differ')
    require(digest(directory/'future.mp4') == old['prediction_sha256'], 'Prior24 video changed')
    current, previous = decoded_video(video), decoded_video(directory/'future.mp4')
    delta = np.abs(current.astype(np.int16)-previous.astype(np.int16))
    return {'status':'pixel_identical' if np.array_equal(current,previous) else 'different',
            'previous_trace':str(directory/'trace.json'),'previous_trace_sha256':digest(directory/'trace.json'),
            'previous_prediction':str(directory/'future.mp4'),'previous_prediction_sha256':old['prediction_sha256'],
            'decoded_max_abs_difference_uint8':int(delta.max()),'decoded_mean_abs_difference_uint8':float(delta.mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--shard-index',type=int,required=True)
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args()
    plan_path, output = args.plan.resolve(), args.output.resolve()
    plan = validate_plan(plan_path)
    require(0 <= args.shard_index < 4, 'Wrong shard')
    selected = [c for c in plan['cases'] if c['shard_index'] == args.shard_index]
    require(len(selected) == 120, 'Wrong shard size')
    cfg = load_config(plan['config'])
    require(float(cfg.model.video_scheduler.infer_shift) == 3., 'Configured shift changed')
    dataset = make_dataset(cfg,Path(plan['test_manifest']))
    lookup = {(r['task'],r['episode']):i for i,r in enumerate(dataset.records)}
    require(len(lookup) == 240, 'Wrong test dataset')
    checked = {}
    for case in selected:
        index = lookup[(case['task'],case['episode'])]
        row = dataset.records[index]
        indices = jsonable(dataset.window_indices(index,case['anchor'],include_target=False))
        require(all(case[k] == v for k,v in indices.items() if k in case), 'Production condition indices differ')
        require(0 <= case['anchor'] and case['anchor']+80 < dataset.episode_length(index), 'Invalid future window')
        rgb_sha = row['rgb_sha256']['video'] if isinstance(row['rgb_sha256'],dict) else row['rgb_sha256']
        for path, expected in ((dataset._path(row['condition_video_path']),rgb_sha),(dataset._path(row['text_embedding_path']),row['text_embedding_sha256'])):
            path = str(path)
            if path not in checked: checked[path] = digest(path)
            require(checked[path] == expected, 'RGB/cache source changed')
        if args.dry_run:
            sample = dataset.get_window(index,case['anchor'],include_target=False)
            verify_sample(case,sample)
    if args.dry_run:
        print({'status':'passed','shard':args.shard_index,'cases':120,'predictions':240,'conditions_decoded':True,'gpu_used':False},flush=True)
        return
    require(digest(output/'selection.json') == digest(plan_path), 'Launcher selection mismatch')
    shard_path = output/'shards'/f'shard_{args.shard_index:03d}.json'
    require(not shard_path.exists(), 'Existing shard is immutable')
    base_identity = identity(plan,plan_path)
    state = {'schema':'vpp2_obj.full_sampling_step_shard.v1','status':'loading_model',**base_identity,
             'started_at_utc':now(),'pid':os.getpid(),'shard_index':args.shard_index,'num_shards':4,
             'physical_gpu_ids':os.environ.get('CUDA_VISIBLE_DEVICES'),'case_count':120,'prediction_count':240,
             'num_inference_steps':list(STEPS),'results':[],'warmup':[],'prepared_inputs':[],'prior_24_replays':[],
             'timing_definitions':{
                 'model_inference_seconds':'CUDA-synchronized wall time: condition VAE/CLIP, N DiT calls, full25-frame VAE decode and output conversion.',
                 'condition_load_seconds':'Read and validate conditions after an unmeasured per-case pre-read; warm storage.',
                 'export_seconds':'Write16-frame MP4, rename, SHA256.',
                 'processing_seconds':'Sum of condition_load, model_inference, export; excludes loading/warmup, input hashes, trace/run writes, GC and replay checks.',
                 'order':'Both methods on the same GPU per window;6-first/4-first balanced within shard and phase.'}}
    started = time.perf_counter()
    atomic_json(shard_path,state)
    try:
        import torch
        torch.set_num_threads(4)
        model = resident_model(cfg,Path(plan['checkpoint']))
        state.update(model_load_seconds=time.perf_counter()-started,gpu_name=torch.cuda.get_device_name(0))
        first = selected[0]
        sample = dataset.get_window(lookup[(first['task'],first['episode'])],first['anchor'],include_target=False)
        verify_sample(first,sample)
        for steps in STEPS:
            tick = time.perf_counter()
            prediction = model.infer_object_future(condition_video=sample['condition_video'],context=sample['context'],
                context_mask=sample['context_mask'],num_inference_steps=steps,sigma_shift=3.,seed=0)
            torch.cuda.synchronize()
            require(len(prediction['video']) == 16, 'Wrong warmup frame count')
            state['warmup'].append({'case_id':first['id'],'steps':steps,'seconds':time.perf_counter()-tick,'included_in_statistics':False})
            del prediction
        del sample
        gc.collect()
        noise_shape = (1,int(model.vae.model.z_dim),7,224//int(model.vae.upsampling_factor),448//int(model.vae.upsampling_factor))
        noise = torch.randn(noise_shape,generator=torch.Generator(device='cpu').manual_seed(0),dtype=torch.float32)
        noise_hash = tensor_sha(noise.to(dtype=model.torch_dtype))
        del noise
        state.update(status='running',initial_noise_shape=list(noise_shape),initial_noise_sha256=noise_hash)
        atomic_json(shard_path,state)
        prior_ids = {c['id'] for c in load(plan['prior_24_plan'])['cases']}
        for case_position,case in enumerate(selected):
            index = lookup[(case['task'],case['episode'])]
            row = dataset.records[index]
            prepared = dataset.get_window(index,case['anchor'],include_target=False)
            verify_sample(case,prepared)
            signatures = input_signature(prepared)
            state['prepared_inputs'].append({'id':case['id'],'tensor_sha256':signatures})
            del prepared
            for position,steps in enumerate(case['variant_order']):
                directory = output/'samples'/case['id']/f'steps_{steps}'
                directory.mkdir(parents=True)
                tick = time.perf_counter()
                sample = dataset.get_window(index,case['anchor'],include_target=False)
                verify_sample(case,sample)
                condition_seconds = time.perf_counter()-tick
                require(input_signature(sample) == signatures,'Input tensors changed across variants')
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                tick = time.perf_counter()
                prediction = model.infer_object_future(condition_video=sample['condition_video'],context=sample['context'],
                    context_mask=sample['context_mask'],num_inference_steps=steps,sigma_shift=3.,seed=0)
                torch.cuda.synchronize()
                model_seconds = time.perf_counter()-tick
                peak_allocated,peak_reserved = torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()
                require(len(prediction['video']) == 16,'Wrong output frame count')
                tick = time.perf_counter()
                temporary = directory/'.future.tmp.mp4'
                write_video(prediction['video'],temporary,6)
                temporary.replace(directory/'future.mp4')
                prediction_sha = digest(directory/'future.mp4')
                export_seconds = time.perf_counter()-tick
                timing = {'condition_load_seconds':condition_seconds,'model_inference_seconds':model_seconds,
                          'export_seconds':export_seconds,'processing_seconds':condition_seconds+model_seconds+export_seconds}
                trace = {'schema':'vpp2_obj.inference.v1','status':'complete','dry_run':False,**base_identity,
                         'task':case['task'],'episode':case['episode'],'anchor_index':case['anchor'],
                         'phase':case['phase'],'phase_eligible':case['phase_eligible'],'fallback_reason':case['fallback_reason'],
                         'shard_index':args.shard_index,'num_shards':4,'case_position':case_position,
                         'num_inference_steps':steps,'variant_order':case['variant_order'],'variant_position':position,
                         'sampling':window_metadata(sample),'condition_video_path':row['condition_video_path'],
                         'text_embedding_path':row['text_embedding_path'],'text_embedding_sha256':row['text_embedding_sha256'],
                         'text_embedding_cache_signature':row['text_embedding_cache_signature'],'prompt':row['prompt'],
                         'device':'cuda:0','physical_gpu_ids':state['physical_gpu_ids'],'gpu_name':state['gpu_name'],
                         'output_directory':str(directory),'video_path':str(directory/'future.mp4'),'prediction_sha256':prediction_sha,
                         'exported_frames':16,'decoded_frame_slice':[9,25],'prefix_clamped':False,
                         'target_video_read':False,'target_first_frame_read':False,'ground_truth_used_for_window_selection':True,
                         'display_fps':6,'condition_shape':list(sample['condition_video'].shape),'context_shape':list(sample['context'].shape),
                         'context_mask':sample['context_mask'].tolist(),'model_visible_text_tokens':int(sample['context_mask'].sum()),
                         'input_tensor_sha256':signatures,'initial_noise_sha256':noise_hash,'initial_noise_shape':list(noise_shape),
                         'peak_allocated_bytes':peak_allocated,'peak_reserved_bytes':peak_reserved,
                         'elapsed_seconds':timing['processing_seconds'],'timing':timing,'warmup_excluded':True}
                if case['id'] in prior_ids:
                    replay = replay_previous(plan,case,trace,directory/'future.mp4')
                    trace['prior_24_replay'] = replay
                    state['prior_24_replays'].append({'id':case['id'],'num_inference_steps':steps,**replay})
                atomic_json(directory/'trace.json',trace)
                state['results'].append({'id':case['id'],'num_inference_steps':steps,'status':'complete',
                                         'trace':str(directory/'trace.json'),'timing':timing})
                state.update(updated_at_utc=now(),completed_prediction_count=len(state['results']))
                atomic_json(shard_path,state)
                print(f"[{len(state['results'])}/240] {case['id']} steps={steps} model={model_seconds:.3f}s processing={timing['processing_seconds']:.3f}s",flush=True)
                del sample,prediction
                gc.collect()
        require(len(state['results']) == 240, 'Missing predictions')
        require(identity(plan,plan_path) == base_identity,'Implementation/config identity changed during inference')
        require(digest(plan['prior_24_run']+'/run.json') == plan['prior_24_run_sha256'],'Prior run changed')
        state.update(status='complete',finished_at_utc=now(),elapsed_seconds=time.perf_counter()-started)
        atomic_json(shard_path,state)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),finished_at_utc=now())
        atomic_json(shard_path,state)
        raise


if __name__ == '__main__':
    main()

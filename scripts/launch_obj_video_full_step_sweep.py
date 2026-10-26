#!/usr/bin/env python3
"""Supervise four paired6/4 GPU workers; preserve all previous experiments."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from run_obj_video_full_step_shard import ROOT, MEMORY, STEPS, validate_plan, identity, now, load, require, digest, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpus',default='4,5,6,7')
    args = parser.parse_args()
    plan_path,output = args.plan.resolve(),args.output.resolve()
    plan = validate_plan(plan_path)
    gpus = args.gpus.split(',')
    require(len(gpus) == len(set(gpus)) == 4 and all(g.isdigit() for g in gpus),'Expected four distinct GPU IDs')
    require(not output.exists(),'Use a new output directory; old runs are immutable')
    for gpu in gpus:
        free = int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        require(free >= MEMORY['minimum_free_before_placement_gib']*1024,f'GPU{gpu} insufficiently free')
    output.mkdir(parents=True)
    (output/'samples').mkdir()
    (output/'shards').mkdir()
    (output/'selection.json').write_bytes(plan_path.read_bytes())
    (output/'supervisor.pid').write_text(str(os.getpid())+'\n')
    control = plan_path.parent
    metadata = identity(plan,plan_path)
    state = {'schema':'vpp2_obj.full_sampling_step_run.v1','status':'preparing',**metadata,
             'launcher_sha256':digest(__file__),'started_at_utc':now(),'supervisor_pid':os.getpid(),
             'case_count':480,'prediction_count':960,'num_inference_steps':list(STEPS),'num_shards':4,
             'physical_gpu_ids':','.join(gpus),'results':[],'shards':[],'jobs':[]}
    jobs,handles = [],[]
    started = time.perf_counter()

    def publish():
        shards,results = [],{}
        for shard in range(4):
            path = output/'shards'/f'shard_{shard:03d}.json'
            if not path.exists(): continue
            # Atomic writer: hash the exact same bytes that are parsed.
            import hashlib,json
            raw = path.read_bytes()
            data = json.loads(raw)
            for key,value in metadata.items(): require(data[key] == value,f'Shard identity changed:{key}')
            require(data['shard_index'] == shard and data['physical_gpu_ids'] == gpus[shard],'Shard/GPU changed')
            require(data['case_count'] == 120 and data['prediction_count'] == 240 and data['num_shards'] == 4,'Shard counts differ')
            expected = {c['id'] for c in plan['cases'] if c['shard_index'] == shard}
            for item in data['results']:
                key = (item['id'],item['num_inference_steps'])
                require(item['id'] in expected and item['num_inference_steps'] in STEPS and item['status'] == 'complete','Unexpected result')
                require(key not in results,'Duplicate paired result')
                results[key] = item
            if data['status'] == 'complete': require(len(data['results']) == 240,'False complete shard')
            shards.append({'shard_index':shard,'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),
                           'status':data['status'],'physical_gpu_ids':gpus[shard],
                           'completed_prediction_count':len(data['results'])})
        state['shards'] = shards
        state['results'] = [results[(c['id'],n)] for c in plan['cases'] for n in c['variant_order'] if (c['id'],n) in results]
        state['completed_prediction_count'] = len(results)
        state['jobs'] = [{**job,'returncode':process.poll()} for process,job in jobs]
        state['updated_at_utc'] = now()
        atomic_json(output/'run.json',state)

    try:
        publish()
        for shard,gpu in enumerate(gpus):
            log = control/f'shard_{shard:03d}.log'
            handle = log.open('x')
            handles.append(handle)
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                       PYTORCH_ALLOC_CONF='expandable_segments:True',VPP2_TORCH_LOAD_MMAP='1',PYTHONUNBUFFERED='1')
            env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
            command = [sys.executable,str(ROOT/'scripts/run_obj_video_full_step_shard.py'),'--plan',str(plan_path),
                       '--output',str(output),'--shard-index',str(shard)]
            process = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=handle,
                                       stderr=subprocess.STDOUT,start_new_session=True)
            jobs.append((process,{'shard_index':shard,'gpu':gpu,'pid':process.pid,'log':str(log)}))
        state['status'] = 'running'
        publish()
        last_report = 0.
        while any(p.poll() is None for p,_ in jobs):
            time.sleep(5)
            publish()
            require(all(p.poll() in (None,0) for p,_ in jobs),'A worker failed; see shard log')
            if time.monotonic()-last_report >= 30:
                print({'time_utc':now(),'completed':state['completed_prediction_count'],'expected':960,
                       'shards':[s['completed_prediction_count'] for s in state['shards']]},flush=True)
                last_report = time.monotonic()
        publish()
        require(all(p.returncode == 0 for p,_ in jobs),'Worker exit failure')
        require(len(state['results']) == 960 and len(state['shards']) == 4
                and all(s['status'] == 'complete' for s in state['shards']),'Incomplete run')
        require(identity(plan,plan_path) == metadata and digest(__file__) == state['launcher_sha256'],'Implementation changed')
        state.update(status='complete',finished_at_utc=now(),elapsed_seconds=time.perf_counter()-started)
        publish()
        print({'status':'complete','predictions':960,'output':str(output)},flush=True)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),finished_at_utc=now())
        for process,_ in jobs:
            if process.poll() is None: os.killpg(process.pid,signal.SIGTERM)
        for process,_ in jobs:
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
        publish()
        raise
    finally:
        for handle in handles: handle.close()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Supervise four GPU workers and publish coherent per-checkpoint batch traces."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from run_obj_video_eval_shard import ROOT, MEMORY, digest, require, atomic_json


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    control = plan_path.parent
    plan = json.loads(plan_path.read_text())
    require(plan["case_count"] == len(plan["cases"]) == 480 and plan["episode_count"] == 240 and plan["task_count"] == 50,
            "Expected the complete clean test evaluation plan")
    pilot = ROOT / "runs/obj_video_resident_pilot_v1_control/replay_check.json"
    require(pilot.is_file() and json.loads(pilot.read_text())["status"] == "passed", "Resident replay must pass first")
    pid_path = control / "supervisor.pid"
    if pid_path.exists():
        require(args.resume, "Supervisor was already launched; explicit --resume required")
        old_pid = int(pid_path.read_text())
        try:
            os.kill(old_pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError(f"Previous supervisor PID {old_pid} is still alive")
    pid_path.write_text(str(os.getpid()) + "\n")
    checkpoint_runs, jobs, handles = {}, [], []
    state = {"status": "preparing", "started_at_utc": now(), "supervisor_pid": os.getpid(),
             "plan": str(plan_path), "plan_sha256": digest(plan_path), "case_count": 480,
             "checkpoint_steps": [8000, 10000], "predictions_expected": 960,
             "memory_strategy": MEMORY, "resident_replay": str(pilot), "jobs": []}

    def publish():
        statuses = []
        for step, output in checkpoint_runs.items():
            partial_results, shards = {}, []
            for shard in range(2):
                path = output / "shards" / f"shard_{shard:03d}.json"
                if path.is_file():
                    data = json.loads(path.read_text())
                    require(data["plan_sha256"] == state["plan_sha256"], "Shard plan mismatch")
                    expected_checkpoint = ROOT / f"runs/obj_video_train_v1/checkpoints/weights/step_{step:06d}.pt"
                    require(Path(data["checkpoint"]).resolve() == expected_checkpoint, "Shard checkpoint mismatch")
                    require(data["shard_index"] == shard and data["num_shards"] == 2 and data["case_count"] == 240,
                            "Shard identity or count mismatch")
                    require(data["memory_strategy"] == MEMORY, "Shard placement/dtypes differ")
                    expected_ids = {case["id"] for i, case in enumerate(plan["cases"]) if i % 2 == shard}
                    shards.append(data)
                    for item in data["results"]:
                        require(item["id"] in expected_ids and item["status"] == "complete", "Invalid completed shard result")
                        require(item["id"] not in partial_results, "Duplicate result across shards")
                        partial_results[item["id"]] = item
            ordered = [partial_results[c["id"]] for c in plan["cases"] if c["id"] in partial_results]
            require(len(ordered) == len(partial_results), "Unexpected shard case IDs")
            done = len(shards) == 2 and all(x["status"] == "complete" for x in shards)
            if done:
                require(len(ordered) == 480, "Complete shards have missing cases")
            status = "complete" if done else "failed" if any(x["status"] == "failed" for x in shards) else "running"
            batch = {"schema": "vpp2_obj.batch_inference.v1", "status": status,
                     "checkpoint": str(ROOT / f"runs/obj_video_train_v1/checkpoints/weights/step_{step:06d}.pt"),
                     "config": plan["config"], "plan_sha256": state["plan_sha256"],
                     "test_manifest_sha256": plan["test_manifest_sha256"], "case_count": 480,
                     "results": ordered, "ground_truth_used_for_selection": True,
                     "target_video_read_by_inference": False, "memory_strategy": MEMORY,
                     "physical_gpu_ids": "4,5" if step == 8000 else "6,7", "num_shards": 2,
                     "updated_at_utc": now(), "resident_replay_check_sha256": digest(pilot)}
            atomic_json(output / "batch_trace.json", batch)
            statuses.append({"step": step, "status": status, "complete": len(ordered), "expected": 480})
        state["checkpoints"] = statuses
        state["updated_at_utc"] = now()
        state["jobs"] = [{**metadata, "returncode": process.poll()} for process, metadata in jobs]
        atomic_json(control / "status.json", state)

    try:
        for step in (8000, 10000):
            output = ROOT / f"runs/obj_video_step{step}_full_eval_v1"
            if output.exists():
                require(args.resume and digest(output / "selection.json") == state["plan_sha256"], "Output already exists or plan changed")
            else:
                output.mkdir()
                (output / "samples").mkdir()
                (output / "shards").mkdir()
                (output / "selection.json").write_bytes(plan_path.read_bytes())
            checkpoint_runs[step] = output
        publish()
        for step, shard, gpu in ((8000, 0, 4), (8000, 1, 5), (10000, 0, 6), (10000, 1, 7)):
            free = int(subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
            require(free >= MEMORY["minimum_free_before_placement_gib"] * 1024, f"GPU{gpu} is not sufficiently free")
            log = control / f"step{step}_shard{shard}.log"
            handle = log.open("a" if args.resume else "x")
            handles.append(handle)
            command = [sys.executable, str(ROOT / "scripts/run_obj_video_eval_shard.py"),
                       "--plan", str(plan_path), "--checkpoint", str(ROOT / f"runs/obj_video_train_v1/checkpoints/weights/step_{step:06d}.pt"),
                       "--output", str(checkpoint_runs[step]), "--config", plan["config"],
                       "--shard-index", str(shard), "--num-shards", "2"]
            if args.resume:
                command.append("--resume")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                       PYTORCH_ALLOC_CONF="expandable_segments:True", VPP2_TORCH_LOAD_MMAP="1", PYTHONUNBUFFERED="1")
            env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            jobs.append((process, {"step": step, "shard": shard, "gpu": gpu, "pid": process.pid, "log": str(log)}))
        state["status"] = "running"
        publish()
        last_report = 0.0
        while any(process.poll() is None for process, _ in jobs):
            time.sleep(5)
            publish()
            if time.monotonic() - last_report >= 30:
                print(json.dumps({"time_utc": now(), "checkpoints": state["checkpoints"]}), flush=True)
                last_report = time.monotonic()
        publish()
        success = all(process.returncode == 0 for process, _ in jobs) and all(x["status"] == "complete" for x in state["checkpoints"])
        state.update(status="complete" if success else "failed", finished_at_utc=now())
        publish()
        return 0 if success else 1
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), traceback=traceback.format_exc(), finished_at_utc=now())
        for process, _ in jobs:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, _ in jobs:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        publish()
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())

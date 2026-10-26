#!/usr/bin/env python3
"""Frozen 6/4/2-step comparison on one resident GPU; no target model inputs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
import traceback

from run_obj_video_eval_shard import (
    ROOT, MEMORY, atomic_json, digest, require, resident_model,
    jsonable, load_config, make_dataset, validate_condition_sample,
    window_metadata, write_video,
)

STEPS = (6, 4, 2)
OLD_RUN = ROOT / "runs/obj_video_step10000_steps_10_8_6_v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text())


def tensor_sha(value):
    import torch
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def input_signature(sample):
    return {name: tensor_sha(sample[name]) for name in ("condition_video", "context", "context_mask")}


def verify_sample(case, sample):
    validate_condition_sample(sample)
    metadata = window_metadata(sample)
    for key, value in metadata.items():
        if key in case:
            require(value == case[key], f"Selection/condition mismatch: {case['id']} {key}")
    require(not {"video", "future_source_indices", "target_video_path"}.intersection(sample),
            "A model sample contains target-only data")
    require(sample["condition_video"].shape == (3, 9, 224, 448), "Unexpected RGB dimensions")


def decoded_video(path):
    import av
    import numpy as np
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        require((stream.width, stream.height) == (448, 224), "Replay resolution differs")
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    require(len(frames) == 16, "Replay requires exactly sixteen frames")
    return np.stack(frames)


def replay_6(case, trace, video):
    import numpy as np
    old_dir = OLD_RUN / "samples" / case["id"] / "steps_6"
    previous = load(old_dir / "trace.json")
    for key in ("checkpoint", "checkpoint_bytes", "checkpoint_mtime_ns", "config_sha256",
                "manifest_sha256", "task", "episode", "anchor_index", "sampling", "seed",
                "num_inference_steps", "memory_strategy", "prompt", "text_embedding_sha256",
                "context_mask", "condition_shape", "decoded_frame_slice", "prefix_clamped",
                "sigma_shift", "input_tensor_sha256", "initial_noise_sha256", "initial_noise_shape"):
        require(previous[key] == trace[key], f"Previous 6-step setting differs: {key}")
    old_video = old_dir / "future.mp4"
    require(digest(old_video) == previous["prediction_sha256"], "Previous prediction changed")
    current, old = decoded_video(video), decoded_video(old_video)
    difference = np.abs(current.astype(np.int16)-old.astype(np.int16))
    return {"status": "pixel_identical" if np.array_equal(current, old) else "different",
            "previous_trace": str(old_dir / "trace.json"), "previous_trace_sha256": digest(old_dir / "trace.json"),
            "previous_prediction": str(old_video), "previous_prediction_sha256": digest(old_video),
            "decoded_max_abs_difference_uint8": int(difference.max()),
            "decoded_mean_abs_difference_uint8": float(difference.mean())}


def prepare(plan_path):
    plan = load(plan_path)
    require(plan["status"] == "frozen" and plan["num_inference_steps"] == list(STEPS), "Unexpected plan")
    require(plan["sigma_shift"] == 3.0, "Expected original inference shift3")
    cases = plan["cases"]
    require(len(cases) == len({c["id"] for c in cases}) == 24, "Expected24 unique case IDs")
    require(len({(c["task"], c["episode"], c["anchor"]) for c in cases}) == 24, "Duplicate temporal windows")
    require(len({c["task"] for c in cases}) == 12, "Expected12 tasks")
    require(len({(c["task"], c["episode"]) for c in cases}) == 12, "Expected12 episodes")
    phases, orders = defaultdict(set), defaultdict(Counter)
    for case in cases:
        require(case["phase_eligible"] is True and case["seed"] == 0, "Only eligible fixed-seed cases")
        require(Path(case["id"]).name == case["id"] and case["id"] not in (".", ".."), "Unsafe case ID")
        require(sorted(case["variant_order"]) == sorted(STEPS), "Unexpected variant order")
        phases[(case["task"], case["episode"])].add(case["phase"])
        orders[case["phase"]][tuple(case["variant_order"])] += 1
    require(all(p == {"pre_onset", "mid_motion"} for p in phases.values()), "Missing phase")
    require(all(o == Counter({p: 2 for p in itertools.permutations(STEPS)}) for o in orders.values()),
            "Variant order is not balanced within phase")
    config, manifest, checkpoint = [Path(plan[k]).resolve() for k in ("config", "test_manifest", "checkpoint")]
    require(digest(config) == plan["config_sha256"], "Config fingerprint changed")
    require(digest(manifest) == plan["test_manifest_sha256"], "Test manifest fingerprint changed")
    require(checkpoint.is_file() and checkpoint.name == "step_010000.pt", "Wrong checkpoint")
    stat = checkpoint.stat()
    require((stat.st_size, stat.st_mtime_ns) == (plan["checkpoint_stat"]["size_bytes"], plan["checkpoint_stat"]["mtime_ns"]),
            "Checkpoint changed since selection")
    require(digest(plan["parent_plan"]) == plan["parent_plan_sha256"], "Parent selection changed")
    require(plan["reference_inference_steps"] == 6, "Wrong replay reference")
    require(Path(plan["previous_step_sweep_run"]).resolve() == OLD_RUN.resolve(), "Wrong previous run")
    require(digest(OLD_RUN / "run.json") == plan["previous_step_sweep_run_sha256"], "Previous run changed")
    require(digest(plan["previous_step_sweep_plan"]) == plan["previous_step_sweep_plan_sha256"], "Previous selection changed")
    previous_plan = load(plan["previous_step_sweep_plan"])
    previous_run = load(OLD_RUN / "run.json")
    require(previous_run["status"] == "complete" and previous_run["plan_sha256"] == plan["previous_step_sweep_plan_sha256"],
            "Previous run is incomplete or uses another plan")
    strip_variants = lambda case: {k: v for k, v in case.items() if k not in ("sampling_steps", "variant_order")}
    require([strip_variants(c) for c in cases] == [strip_variants(c) for c in previous_plan["cases"]],
            "Window identities differ from previous sweep")
    cfg = load_config(config)
    require(float(cfg.model.video_scheduler.infer_shift) == 3.0, "Configured shift changed")
    dataset = make_dataset(cfg, manifest)
    lookup = {(r["task"], r["episode"]): i for i, r in enumerate(dataset.records)}
    for case in cases:
        index = lookup[(case["task"], case["episode"])]
        indices = jsonable(dataset.window_indices(index, case["anchor"], include_target=False))
        for key, value in indices.items():
            if key in case:
                require(value == case[key], f"Production sampling mismatch: {case['id']} {key}")
        require(case["anchor"] + 80 < dataset.episode_length(index), "Incomplete future")
        row = dataset.records[index]
        rgb_sha = row["rgb_sha256"]
        rgb_sha = rgb_sha["video"] if isinstance(rgb_sha, dict) else rgb_sha
        require(digest(dataset._path(row["condition_video_path"])) == rgb_sha, "RGB video changed")
        require(digest(dataset._path(row["text_embedding_path"])) == row["text_embedding_sha256"], "Cached text changed")
        previous = load(OLD_RUN / "samples" / case["id"] / "steps_6" / "trace.json")
        require(previous["status"] == "complete" and previous["num_inference_steps"] == 6,
                "Missing completed previous6-step reference")
    return plan, cfg, dataset, lookup, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan_path, output = args.plan.resolve(), args.output.resolve()
    plan, cfg, dataset, lookup, checkpoint = prepare(plan_path)
    if args.dry_run:
        for case in plan["cases"]:
            sample = dataset.get_window(lookup[(case["task"], case["episode"])], case["anchor"], include_target=False)
            verify_sample(case, sample)
        print(json.dumps({"status": "passed", "case_count": 24, "predictions": 72,
                          "phase_order_balanced": True, "conditions_decoded": True, "gpu_used": False}))
        return
    require(not output.exists(), "Choose a new output directory; existing results are immutable")
    output.mkdir(parents=True)
    (output / "samples").mkdir()
    (output / "selection.json").write_bytes(plan_path.read_bytes())
    (output / "runner.pid").write_text(str(os.getpid()) + "\n")
    stat = checkpoint.stat()
    identity = {"config": str(Path(plan["config"]).resolve()), "config_sha256": plan["config_sha256"],
                "manifest": str(Path(plan["test_manifest"]).resolve()), "manifest_sha256": plan["test_manifest_sha256"],
                "checkpoint": str(checkpoint), "checkpoint_bytes": stat.st_size, "checkpoint_mtime_ns": stat.st_mtime_ns,
                "plan_sha256": digest(plan_path), "runner_sha256": digest(__file__),
                "resident_loader_sha256": digest(Path(__file__).with_name("run_obj_video_eval_shard.py")),
                "seed": 0, "sigma_shift": 3.0, "memory_strategy": MEMORY,
                "previous_step_sweep_plan_sha256": plan["previous_step_sweep_plan_sha256"],
                "previous_step_sweep_run_sha256": plan["previous_step_sweep_run_sha256"],
                "reference_inference_steps": 6}
    state = {"schema": "vpp2_obj.sampling_step_run.v1", "status": "loading_model", **identity,
             "started_at_utc": now(), "physical_gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES"),
             "case_count": 24, "prediction_count": 72, "num_inference_steps": list(STEPS),
             "results": [], "warmup": [], "prepared_inputs": [], "previous_6_step_replays": [],
             "timing_definitions": {
                 "model_inference_seconds": "Wall time bracketed by CUDA synchronization; includes RGB VAE/CLIP conditioning, N DiT calls, full25-frame VAE decode and conversion to output frames. Cached text is used.",
                 "condition_load_seconds": "get_window and input validation after an explicit unmeasured per-case pre-read; not cold storage latency.",
                 "export_seconds": "Write16-frame MP4, rename and SHA256.",
                 "processing_seconds": "Condition read+validation, synchronized inference and export; excludes model loading/warmups, input hashing, trace/shard writes, garbage collection and old6-step replay checks.",
                 "order": "All six step permutations occur twice within each phase; one GPU/model, fixed seed0 and shift3.",
             }}
    started = time.perf_counter()
    atomic_json(output / "run.json", state)
    try:
        import torch
        torch.set_num_threads(4)
        model = resident_model(cfg, checkpoint)
        state["model_load_seconds"] = time.perf_counter() - started
        state["gpu_name"] = torch.cuda.get_device_name(0)
        first = plan["cases"][0]
        warm_sample = dataset.get_window(lookup[(first["task"], first["episode"])], first["anchor"], include_target=False)
        verify_sample(first, warm_sample)
        for steps in (6, 2):
            tick = time.perf_counter()
            prediction = model.infer_object_future(condition_video=warm_sample["condition_video"], context=warm_sample["context"],
                context_mask=warm_sample["context_mask"], num_inference_steps=steps, sigma_shift=3.0, seed=0)
            torch.cuda.synchronize()
            require(len(prediction["video"]) == 16, "Wrong warmup frame count")
            state["warmup"].append({"case_id": first["id"], "steps": steps,
                                    "seconds": time.perf_counter()-tick, "included_in_statistics": False})
            del prediction
        del warm_sample
        gc.collect()
        noise_shape = (1, int(model.vae.model.z_dim), 7, 224 // int(model.vae.upsampling_factor), 448 // int(model.vae.upsampling_factor))
        noise = torch.randn(noise_shape, generator=torch.Generator(device="cpu").manual_seed(0), dtype=torch.float32)
        initial_noise_hash = tensor_sha(noise.to(dtype=model.torch_dtype))
        del noise
        state.update(status="running", initial_noise_shape=list(noise_shape), initial_noise_sha256=initial_noise_hash)
        atomic_json(output / "run.json", state)
        for case_index, case in enumerate(plan["cases"]):
            index = lookup[(case["task"], case["episode"])]
            row = dataset.records[index]
            prepared = dataset.get_window(index, case["anchor"], include_target=False)
            verify_sample(case, prepared)
            expected_inputs = input_signature(prepared)
            state["prepared_inputs"].append({"id": case["id"], "tensor_sha256": expected_inputs})
            del prepared
            for position, steps in enumerate(case["variant_order"]):
                directory = output / "samples" / case["id"] / f"steps_{steps}"
                directory.mkdir(parents=True)
                tick = time.perf_counter()
                sample = dataset.get_window(index, case["anchor"], include_target=False)
                verify_sample(case, sample)
                condition_seconds = time.perf_counter()-tick
                # Input provenance checks are intentionally outside latency metrics.
                require(input_signature(sample) == expected_inputs, "Input tensors differ across variants")
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                tick = time.perf_counter()
                prediction = model.infer_object_future(condition_video=sample["condition_video"], context=sample["context"],
                    context_mask=sample["context_mask"], num_inference_steps=steps, sigma_shift=3.0, seed=0)
                torch.cuda.synchronize()
                model_seconds = time.perf_counter()-tick
                peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
                require(len(prediction["video"]) == 16, "Wrong prediction frame count")
                tick = time.perf_counter()
                temporary = directory / ".future.tmp.mp4"
                write_video(prediction["video"], temporary, 6)
                temporary.replace(directory / "future.mp4")
                prediction_sha = digest(directory / "future.mp4")
                export_seconds = time.perf_counter()-tick
                processing_seconds = condition_seconds + model_seconds + export_seconds
                trace = {"schema": "vpp2_obj.inference.v1", "status": "complete", "dry_run": False, **identity,
                    "task": case["task"], "episode": case["episode"], "anchor_index": case["anchor"],
                    "phase": case["phase"], "phase_eligible": True,
                    "num_inference_steps": steps, "variant_order": case["variant_order"], "variant_position": position,
                    "case_position": case_index, "sampling": window_metadata(sample),
                    "condition_video_path": row["condition_video_path"], "text_embedding_path": row["text_embedding_path"],
                    "text_embedding_sha256": row["text_embedding_sha256"],
                    "text_embedding_cache_signature": row["text_embedding_cache_signature"], "prompt": row["prompt"],
                    "device": "cuda:0", "physical_gpu_ids": state["physical_gpu_ids"], "gpu_name": state["gpu_name"],
                    "output_directory": str(directory), "video_path": str(directory / "future.mp4"),
                    "prediction_sha256": prediction_sha, "exported_frames": 16, "decoded_frame_slice": [9, 25],
                    "prefix_clamped": False, "target_video_read": False, "target_first_frame_read": False,
                    "ground_truth_used_for_window_selection": True, "display_fps": 6,
                    "condition_shape": list(sample["condition_video"].shape), "context_shape": list(sample["context"].shape),
                    "context_mask": sample["context_mask"].tolist(), "model_visible_text_tokens": int(sample["context_mask"].sum()),
                    "input_tensor_sha256": expected_inputs, "initial_noise_sha256": initial_noise_hash,
                    "initial_noise_shape": list(noise_shape), "initial_noise_hash_definition": "CPU fp32 torch.randn with fresh Generator seed0, then cast to model bf16; same algorithm and shape as Wan21.infer.",
                    "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
                    "elapsed_seconds": processing_seconds, "warmup_excluded": True,
                    "timing": {"condition_load_seconds": condition_seconds, "model_inference_seconds": model_seconds,
                               "export_seconds": export_seconds, "processing_seconds": processing_seconds}}
                if steps == 6:
                    trace["previous_6_step_replay"] = replay_6(case, trace, directory / "future.mp4")
                    state["previous_6_step_replays"].append({"id": case["id"], **trace["previous_6_step_replay"]})
                atomic_json(directory / "trace.json", trace)
                state["results"].append({"id": case["id"], "num_inference_steps": steps, "status": "complete",
                                         "trace": str(directory / "trace.json"), "timing": trace["timing"]})
                state.update(updated_at_utc=now(), completed_prediction_count=len(state["results"]))
                atomic_json(output / "run.json", state)
                print(f"[{len(state['results'])}/72] {case['id']} steps={steps} model={model_seconds:.3f}s processing={processing_seconds:.3f}s", flush=True)
                del sample, prediction
                gc.collect()
        require(len(state["results"]) == 72, "Incomplete prediction set")
        require(digest(plan_path) == identity["plan_sha256"] and digest(__file__) == identity["runner_sha256"], "Plan or runner changed during run")
        replays = state["previous_6_step_replays"]
        require(len(replays) == 24, "Missing original6-step replay checks")
        state["previous_6_step_replay_summary"] = {
            "checked_case_count": len(replays), "pixel_identical_count": sum(r["status"] == "pixel_identical" for r in replays),
            "all_pixel_identical": all(r["status"] == "pixel_identical" for r in replays),
            "maximum_decoded_abs_difference_uint8": max(r["decoded_max_abs_difference_uint8"] for r in replays)}
        state.update(status="complete", finished_at_utc=now(), elapsed_seconds=time.perf_counter()-started)
        atomic_json(output / "run.json", state)
        print(json.dumps({"status": "complete", "predictions": 72, "output": str(output)}), flush=True)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), traceback=traceback.format_exc(), finished_at_utc=now())
        atomic_json(output / "run.json", state)
        raise


if __name__ == "__main__":
    main()


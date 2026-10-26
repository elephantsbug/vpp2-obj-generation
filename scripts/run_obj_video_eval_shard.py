#!/usr/bin/env python3
"""Run one deterministic shard of a frozen evaluation plan, with resumable outputs."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vpp2_obj.infer import (atomic_json, jsonable, load_config, make_dataset,
                           make_model, validate_condition_sample, window_metadata, write_video)

MEMORY = {
    "method": "all model components resident on one CUDA device",
    "initial_load_device": "cpu", "execution_device": "cuda:0",
    "pytorch_allocator_limit_gib": 96, "minimum_free_before_placement_gib": 100,
    "dit_dtype": "bfloat16", "vae_dtype": "float32", "clip_dtype": "float16",
    "quantization": False,
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def resident_model(cfg, checkpoint):
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one visible GPU required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(MEMORY["pytorch_allocator_limit_gib"] * 2**30 / total, device)
    require(torch.cuda.mem_get_info(device)[0] >= MEMORY["minimum_free_before_placement_gib"] * 2**30,
            "Insufficient free GPU memory for resident inference")
    model = make_model(cfg, checkpoint, "cpu")
    model.eval().requires_grad_(False)
    require(all(p.dtype == torch.bfloat16 for p in model.dit.parameters()), "DiT precision differs")
    require(torch.cuda.mem_get_info(device)[0] >= MEMORY["minimum_free_before_placement_gib"] * 2**30,
            "GPU free memory changed during model loading")
    model.device = device
    model.dit.to(device=device)
    model._restore_auxiliary_component_dtypes()
    require(model.text_encoder is None, "Use cached text; do not load a text encoder")
    require(all(p.device == device for p in model.parameters()), "Some model parameters are not on CUDA")
    require(all(p.dtype == torch.float32 for p in model.vae.parameters()), "VAE precision differs")
    require(all(p.dtype == torch.float16 for p in model.clip_encoder.parameters()), "CLIP precision differs")
    gc.collect()
    torch.cuda.synchronize()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/obj_video_train_v1.yaml")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    require(0 <= args.shard_index < args.num_shards, "Invalid shard")
    plan_path, checkpoint, config, output = [p.resolve() for p in (args.plan, args.checkpoint, args.config, args.output)]
    plan = json.loads(plan_path.read_text())
    manifest = Path(plan["test_manifest"]).resolve()
    manifest_sha, config_sha, plan_sha = digest(manifest), digest(config), digest(plan_path)
    require(plan["test_manifest_sha256"] == manifest_sha, "Test manifest changed")
    require(plan.get("config_sha256", config_sha) == config_sha, "Config changed")
    require(checkpoint.is_file(), "Checkpoint missing")
    cases = plan["cases"]
    require(len(cases) == plan["case_count"] and len({c["id"] for c in cases}) == len(cases), "Invalid case set")
    selected = [c for i, c in enumerate(cases) if i % args.num_shards == args.shard_index]
    cfg = load_config(config)
    dataset = make_dataset(cfg, manifest)
    lookup = {(r["task"], r["episode"]): i for i, r in enumerate(dataset.records)}
    require(len(lookup) == len(dataset.records), "Duplicate manifest identities")
    for case in selected:
        require(Path(case["id"]).name == case["id"] and case["id"] not in {".", ".."}, "Unsafe case ID")
        require(case["seed"] == 0 and case["steps"] == 10, "Seed/steps changed")
        index = lookup[(case["task"], case["episode"])]
        require(0 <= case["anchor"] and case["anchor"] + 80 < dataset.episode_length(index), "Incomplete future")
        indices = dataset.window_indices(index, case["anchor"], include_target=False)
        for field in ("condition_indices", "condition_source_indices"):
            require(jsonable(indices[field]) == case[field], f"Frozen input trajectory changed: {field}")
    if args.dry_run:
        print(json.dumps({"status": "dry_run_passed", "case_count": len(selected), "plan_sha256": plan_sha,
                          "checkpoint": str(checkpoint), "memory_strategy": MEMORY}, indent=2), flush=True)
        return
    require((output / "selection.json").is_file() and digest(output / "selection.json") == plan_sha,
            "Launcher must prepare an identical selection.json")
    shard_path = output / "shards" / f"shard_{args.shard_index:03d}.json"
    require(args.resume or not shard_path.exists(), "Shard already exists; explicit resume required")
    checkpoint_stat = checkpoint.stat()
    identity = {"config": str(config), "config_sha256": config_sha, "manifest": str(manifest),
                "manifest_sha256": manifest_sha, "checkpoint": str(checkpoint),
                "checkpoint_bytes": checkpoint_stat.st_size, "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
                "seed": 0, "num_inference_steps": 10, "memory_strategy": MEMORY}
    state = {"schema": "vpp2_obj.eval_shard.v1", "status": "preparing", "shard_index": args.shard_index,
             "num_shards": args.num_shards, "case_count": len(selected), "plan_sha256": plan_sha,
             "checkpoint": str(checkpoint), "physical_gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES"),
             "pid": os.getpid(), "results": [], "memory_strategy": MEMORY,
             "runner_sha256": digest(__file__)}
    pending = []
    for case in selected:
        directory = output / "samples" / case["id"]
        trace_path = directory / "trace.json"
        if trace_path.is_file():
            require(args.resume, "Existing sample requires --resume")
            trace = json.loads(trace_path.read_text())
            require(trace.get("status") == "complete", "Existing trace not complete")
            for key, value in identity.items():
                require(trace[key] == value, f"Resume identity differs: {key}")
            require((trace["task"], trace["episode"], trace["anchor_index"]) ==
                    (case["task"], case["episode"], case["anchor"]), "Resume case changed")
            require(trace["plan_sha256"] == plan_sha and digest(directory / "future.mp4") == trace["prediction_sha256"],
                    "Resume prediction or plan changed")
            index = lookup[(case["task"], case["episode"])]
            row = dataset.records[index]
            rgb_sha = row["rgb_sha256"]["video"] if isinstance(row["rgb_sha256"], dict) else row["rgb_sha256"]
            require(digest(dataset._path(row["condition_video_path"])) == rgb_sha, "Resume RGB source changed")
            require(digest(dataset._path(row["text_embedding_path"])) == row["text_embedding_sha256"], "Resume text cache changed")
            sample = dataset.get_window(index, case["anchor"], include_target=False)
            validate_condition_sample(sample)
            require(trace["sampling"] == window_metadata(sample), "Resume sampling changed")
            require(trace["context_mask"] == sample["context_mask"].tolist(), "Resume text mask changed")
            for key in ("prompt", "condition_video_path", "text_embedding_path", "text_embedding_sha256", "text_embedding_cache_signature"):
                require(trace[key] == row[key], f"Resume input identity differs: {key}")
            require(trace.get("phase") == case.get("phase") and trace.get("phase_eligible") == case.get("phase_eligible"), "Resume phase changed")
            del sample
            state["results"].append({"id": case["id"], "status": "complete", "trace": str(trace_path),
                                     "seconds": trace["elapsed_seconds"], "reused_on_resume": True})
        else:
            require(args.resume or not directory.exists(), "Unfinished sample exists; explicit resume required")
            pending.append(case)
    atomic_json(shard_path, state)
    started = time.perf_counter()
    try:
        if pending:
            import torch
            torch.set_num_threads(4)
            state["status"] = "loading_model"
            atomic_json(shard_path, state)
            print(f"Loading {checkpoint.name}: {len(pending)} pending / {len(selected)} shard cases", flush=True)
            model = resident_model(cfg, checkpoint)
            state.update(status="running", model_load_seconds=time.perf_counter() - started)
            atomic_json(shard_path, state)
            for number, case in enumerate(pending, 1):
                tick = time.perf_counter()
                index = lookup[(case["task"], case["episode"])]
                row = dataset.records[index]
                directory = output / "samples" / case["id"]
                directory.mkdir(exist_ok=args.resume)
                print(f"[{number}/{len(pending)}] {case['id']}", flush=True)
                trace = {"schema": "vpp2_obj.inference.v1", "dry_run": False, **identity,
                         "plan_sha256": plan_sha, "runner_sha256": state["runner_sha256"],
                         "task": case["task"], "episode": case["episode"], "anchor_index": case["anchor"],
                         "phase": case.get("phase"), "phase_eligible": case.get("phase_eligible"),
                         "condition_video_path": row["condition_video_path"],
                         "text_embedding_path": row["text_embedding_path"],
                         "text_embedding_sha256": row["text_embedding_sha256"],
                         "text_embedding_cache_signature": row["text_embedding_cache_signature"],
                         "prompt": row["prompt"], "device": "cuda:0", "output_directory": str(directory),
                         "exported_frames": 16, "decoded_frame_slice": [9, 25], "prefix_clamped": False,
                         "target_video_read": False, "target_first_frame_read": False,
                         "ground_truth_used_for_window_selection": True, "display_fps": 6,
                         "display_fps_note": "Playback only, not source simulation-time resampling."}
                try:
                    sample = dataset.get_window(index, case["anchor"], include_target=False)
                    validate_condition_sample(sample)
                    trace.update(sampling=window_metadata(sample), condition_shape=list(sample["condition_video"].shape),
                                 context_shape=list(sample["context"].shape), context_mask=sample["context_mask"].tolist(),
                                 model_visible_text_tokens=int(sample["context_mask"].sum()))
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    prediction = model.infer_object_future(condition_video=sample["condition_video"], context=sample["context"],
                                                           context_mask=sample["context_mask"], num_inference_steps=10, seed=0)
                    require(len(prediction["video"]) == 16, "Expected16 future frames")
                    torch.cuda.synchronize()
                    trace.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                 peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    temporary = directory / ".future.tmp.mp4"
                    write_video(prediction["video"], temporary, 6)
                    temporary.replace(directory / "future.mp4")
                    trace.update(status="complete", video_path=str(directory / "future.mp4"),
                                 prediction_sha256=digest(directory / "future.mp4"), elapsed_seconds=time.perf_counter() - tick)
                    atomic_json(directory / "trace.json", trace)
                    del sample, prediction
                    gc.collect()
                except BaseException as exc:
                    trace.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
                    atomic_json(directory / "failure.json", trace)
                    raise
                state["results"].append({"id": case["id"], "status": "complete", "trace": str(directory / "trace.json"),
                                         "seconds": trace["elapsed_seconds"]})
                atomic_json(shard_path, state)
                print(f"  complete {trace['elapsed_seconds']:.2f}s", flush=True)
        state.update(status="complete", elapsed_seconds=time.perf_counter() - started)
        atomic_json(shard_path, state)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - started)
        atomic_json(shard_path, state)
        raise


if __name__ == "__main__":
    main()

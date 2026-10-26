#!/usr/bin/env python3
"""CPU comparison of old10/8 and new6/4/2 steps, with a real static O_s baseline.

All quality scores are recomputed on GT|static|old10|old8|new6|new4|new2.
Latency cohorts stay separate; current6 is the current-round speed reference.
Frozen source scripts/runs are read-only. No GPU/model operation is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import html
import importlib.util
import io
import itertools
import math
import os
from pathlib import Path
import shutil
import statistics
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HELPER = Path(__file__).with_name("evaluate_obj_video_step_sweep.py")
spec = importlib.util.spec_from_file_location("_low_step_previous_helper", HELPER)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
full, shared, base = previous.full, previous.shared, previous.base
require, digest, load_json = base.require, base.digest, base.load_json
STEPS = (6, 4, 2)
ALL_STEPS = (10, 8, 6, 4, 2)
METHODS = ("static", *[f"steps{n}" for n in ALL_STEPS])
PHASES, SCALARS, TIMINGS = previous.PHASES, previous.SCALARS, previous.TIMINGS
EPS = 1e-12
SCHEMA = "vpp2_obj.low_inference_step_sweep.v1"


def prepare_plans(previous_run, run, files):
    old_plan, old_sha, dataset, index = previous.prepare_plan(previous_run, files)
    plan = load_json(run / "selection.json")
    plan_sha = files.sha(run / "selection.json")
    require(plan["status"] == "frozen" and plan["num_inference_steps"] == list(STEPS), "Expected frozen6/4/2 plan")
    require(plan["reference_inference_steps"] == 6
            and Path(plan["previous_step_sweep_run"]).resolve() == previous_run
            and plan["previous_step_sweep_plan_sha256"] == old_sha, "Previous round binding differs")
    files.sha(plan["previous_step_sweep_plan"], old_sha)
    files.sha(previous_run / "run.json", plan["previous_step_sweep_run_sha256"])
    for field in ("config", "config_sha256", "test_manifest", "test_manifest_sha256", "checkpoint", "sigma_shift", "seed"):
        require(plan[field] == old_plan[field], f"Selection changed across rounds:{field}")
    require(len(plan["cases"]) == 24 and plan.get("case_count", 24) == 24, "Expected24 cases")
    old_cases = {case["id"]: case for case in old_plan["cases"]}
    require([case["id"] for case in plan["cases"]] == [case["id"] for case in old_plan["cases"]], "Case identities/order changed")
    orders = defaultdict(Counter)
    for case in plan["cases"]:
        old = old_cases[case["id"]]
        for key, value in old.items():
            if key not in ("sampling_steps", "variant_order", "steps"):
                require(case.get(key) == value, f"Frozen case changed:{case['id']}/{key}")
        require(case["sampling_steps"] == list(STEPS) and sorted(case["variant_order"]) == sorted(STEPS), "Wrong step variants")
        orders[case["phase"]][tuple(case["variant_order"])] += 1
    require(all(counts == Counter({order: 2 for order in itertools.permutations(STEPS)}) for counts in orders.values()), "Variant order not balanced within phase")
    old_state, old_state_sha = previous.read_run(previous_run / "run.json")
    previous.validate_run(old_state, old_plan, old_sha)
    require(old_state["status"] == "complete", "Previous72 predictions must be complete")
    files.sha(previous_run / "run.json", old_state_sha)
    return plan, plan_sha, old_plan, old_sha, old_state, dataset, index


def validate_run(state, plan, plan_sha):
    require(state["schema"] == "vpp2_obj.sampling_step_run.v1" and state["plan_sha256"] == plan_sha, "Run/plan identity differs")
    for key in ("config_sha256", "checkpoint", "sigma_shift", "seed"):
        require(state[key] == plan[key], f"Run/plan {key} differs")
    for key in ("previous_step_sweep_plan_sha256", "previous_step_sweep_run_sha256", "reference_inference_steps"):
        require(state[key] == plan[key], f"Previous round/run binding differs:{key}")
    require(state["manifest_sha256"] == plan["test_manifest_sha256"] and state["num_inference_steps"] == list(STEPS), "Run manifest/steps differ")
    require(state["case_count"] == 24 and state["prediction_count"] == 72, "Run count differs")
    results = {(row["id"], row["num_inference_steps"]): row for row in state["results"]}
    wanted = {(case["id"], n) for case in plan["cases"] for n in STEPS}
    require(len(results) == len(state["results"]) and set(results) <= wanted, "Duplicate/unexpected scored call")
    if state["status"] == "complete":
        require(set(results) == wanted and all(row["status"] == "complete" for row in results.values()), "Missing completed variants")
    if state["status"] in ("running", "complete"):
        warmups = state["warmup"]
        require(len(warmups) == 2 and {row["steps"] for row in warmups} == {6, 2}
                and all(row["included_in_statistics"] is False for row in warmups), "Expected two unscored6/2 warmups")
    return results


def prepare_case(previous_run, run, plan, plan_sha, old_plan, old_sha, old_state, state, case, dataset, index, files, caches):
    old_case = next(row for row in old_plan["cases"] if row["id"] == case["id"])
    # Reuse frozen validation of actual production RGB/text tensors and fresh
    # CPU noise, all old10/8/6 traces and hashes, and the old10 source binding.
    old = previous.prepare_case(previous_run, old_plan, old_sha, old_state, old_case, dataset, index, files, caches)
    old_results = previous.validate_run(old_state, old_plan, old_sha)
    for item in old["items"]:
        row = old_results[(case["id"], item["steps"])]
        require(Path(row["trace"]).resolve() == Path(item["trace_path"]) and row["timing"] == item["trace"]["timing"], "Previous result/trace timing differs")
    reference = old["items"][2]["trace"]
    prepared = {row["id"]: row["tensor_sha256"] for row in state["prepared_inputs"]}
    require(prepared.get(case["id"]) == reference["input_tensor_sha256"], "Prepared input tensors changed across rounds")
    for key in ("initial_noise_sha256", "initial_noise_shape", "memory_strategy", "physical_gpu_ids"):
        require(state[key] == reference[key], f"Current run changed {key}")
    current = []
    results = validate_run(state, plan, plan_sha)
    invariant = [key for key in shared.TRACE_FIELDS if key != "num_inference_steps"]
    invariant += list(shared.OPTIONAL_MATH_FIELDS) + ["checkpoint", "checkpoint_bytes", "checkpoint_mtime_ns", "phase", "phase_eligible",
                    "input_tensor_sha256", "initial_noise_sha256", "initial_noise_shape", "physical_gpu_ids"]
    for steps in STEPS:
        directory = run / "samples" / case["id"] / f"steps_{steps}"
        trace_path, prediction = directory / "trace.json", directory / "future.mp4"
        trace = load_json(trace_path)
        for key, value in {"schema": "vpp2_obj.inference.v1", "status": "complete", "dry_run": False,
                           "num_inference_steps": steps, "plan_sha256": plan_sha, "sigma_shift": 3., "warmup_excluded": True}.items():
            require(trace.get(key) == value, f"Invalid current trace:{case['id']}/{steps}/{key}")
        for key in invariant:
            require((key in trace, trace.get(key)) == (key in reference, reference.get(key)), f"Input/execution changed across rounds:{key}")
        require(Path(trace["output_directory"]).resolve() == directory and Path(trace["video_path"]).resolve() == prediction, "Trace output path differs")
        require(trace["variant_order"] == case["variant_order"] and trace["variant_position"] == case["variant_order"].index(steps), "Variant order/position differs")
        for key in ("checkpoint", "checkpoint_bytes", "checkpoint_mtime_ns", "config_sha256", "manifest_sha256", "memory_strategy", "initial_noise_sha256",
                    "previous_step_sweep_plan_sha256", "previous_step_sweep_run_sha256", "reference_inference_steps"):
            require(trace[key] == state[key], f"Current trace/run identity differs:{key}")
        timing = trace["timing"]
        require(all(isinstance(timing.get(key), (int, float)) and math.isfinite(timing[key]) and timing[key] >= 0 for key in TIMINGS), "Invalid measured time")
        require(timing["model_inference_seconds"] > 0 and math.isclose(timing["processing_seconds"], sum(timing[key] for key in TIMINGS if key != "processing_seconds"), rel_tol=1e-12, abs_tol=1e-9), "Measured processing sum differs")
        result = results[(case["id"], steps)]
        require(result["status"] == "complete" and Path(result["trace"]).resolve() == trace_path and result["timing"] == timing, "Run result differs from trace")
        current.append({"steps": steps, "trace": trace, "trace_path": str(trace_path), "trace_sha256": files.sha(trace_path),
                        "prediction_path": str(prediction), "prediction_sha256": files.sha(prediction, trace["prediction_sha256"])})
    replay = current[0]["trace"]["previous_6_step_replay"]
    require(replay["status"] in ("pixel_identical", "different"), "Malformed6-step replay")
    for key in ("trace", "prediction"):
        require(Path(replay[f"previous_{key}"]).resolve() == Path(old["items"][2][f"{key}_path"]), "Replay must target this case's previous6-step variant")
        files.sha(replay[f"previous_{key}"], replay[f"previous_{key}_sha256"])
    return {**old, "case": case, "old_items": old["items"], "items": [*old["items"][:2], *current], "current_items": current, "replay6": replay}


def measure(static, predictions, target):
    import numpy as np
    require(len(predictions) == 5 and all(array.shape == target.shape for array in (static, *predictions)), "Expected five prediction variants")
    common = np.logical_or.reduce([array.max(axis=-1) > 16 for array in (target, static, *predictions)])
    scores = {name: {**shared.metric_one(array, target, common),
                     "per_future_frame": [shared.metric_one(array[i], target[i], common[i]) for i in range(16)]}
              for name, array in zip(METHODS, (static, *predictions))}
    frozen = scores["static"]["common_foreground_mae"]
    for name in METHODS[1:]:
        values = scores[name]
        gain = frozen-values["common_foreground_mae"]
        values.update(mae_improvement_vs_static=gain, relative_mae_improvement_vs_static=gain/frozen if frozen > EPS else None,
                      mae_beats_static=gain > EPS, mae_ties_static=abs(gain) <= EPS)
        for baseline in ("steps10", "steps6"):
            denominator = scores[baseline]["common_foreground_mae"]
            delta = values["common_foreground_mae"]-denominator
            values[f"mae_degradation_vs_{baseline}"] = delta
            values[f"relative_mae_degradation_vs_{baseline}"] = delta/denominator if denominator > EPS else None
            values[f"iou_degradation_vs_{baseline}"] = scores[baseline]["silhouette_iou"]-values["silhouette_iou"]
    return {"common_foreground_pixels": int(common.sum()), "common_foreground_pixels_per_frame": common.sum(axis=(1, 2)).tolist(), "methods": scores}


def render(directory, inputs, rgb, static, predictions, target):
    import av
    import numpy as np
    from PIL import Image, ImageDraw
    width, height, fps = inputs["width"], inputs["height"], inputs["fps"]
    case = inputs["case"]
    primary_labels = ["Full-scene RGB GT", "Static O_s", "6 steps (current)", "4 steps (current)", "2 steps (current)"]
    all_labels = ["Full-scene RGB GT", "Static O_s", "10 steps (previous)", "8 steps (previous)", "6 steps (current)", "4 steps (current)", "2 steps (current)"]
    def caption(i):
        us = inputs["indices"]["future_timestamps_us"][i]
        start = inputs["indices"]["condition_timestamps_us"][-1]
        return f"F{i+1}/16 | sim {us/1e6:.3f}s ({(us-start)/1e6:+.3f}s)"
    decodes = {}
    for name, labels, arrays in (("comparison.mp4", primary_labels, [rgb, static, *predictions[2:]]),
                                 ("full_comparison.mp4", all_labels, [rgb, static, *predictions])):
        columns = len(labels)
        with av.open(str(directory/name), "w") as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.width, stream.height, stream.pix_fmt = columns*width, height+76, "yuv420p"
            stream.options = {"crf": "18", "preset": "fast", "threads": "1"}
            for i in range(16):
                canvas = Image.new("RGB", (columns*width, height+76), "#202936")
                draw = ImageDraw.Draw(canvas)
                for col, (label, frames) in enumerate(zip(labels, arrays)):
                    draw.text((col*width+8, 5), label, font=base.font(18), fill="white")
                    canvas.paste(Image.fromarray(frames[i]), (col*width, 76))
                draw.text((10, 31), f"{case['task']} / {case['episode']} / {case['phase']} | s={case['anchor']} | {caption(i)}", font=base.font(17), fill="white")
                draw.text((10, 54), "Checkpoint10000 / seed0 / shift3 | observer left, ego right in each panel | 6fps display", font=base.font(14), fill="#d5dfef")
                frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")
                frame.pts, frame.time_base = i, 1/fps
                for packet in stream.encode(frame): container.mux(packet)
            for packet in stream.encode(): container.mux(packet)
        _, decodes[name] = base.read_prediction(directory/name, columns*width, height+76, fps)
    for name, labels, arrays in (("contact.png", primary_labels, [rgb, static, *predictions[2:]]),
                                 ("object_contact.png", ["Object-only GT", *all_labels[1:]], [target, static, *predictions])):
        canvas = Image.new("RGB", (len(labels)*width, 76+4*(height+32)), "#202936")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 6), f"{case['task']} / {case['episode']} / {case['phase']} | anchor {case['anchor']}", font=base.font(22), fill="white")
        for col, label in enumerate(labels): draw.text((col*width+8, 43), label, font=base.font(18), fill="white")
        for row, i in enumerate((0, 4, 9, 15)):
            y = 76+row*(height+32)
            draw.text((10, y+4), caption(i), font=base.font(17), fill="white")
            for col, frames in enumerate(arrays): canvas.paste(Image.fromarray(frames[i]), (col*width, y+32))
        canvas.save(directory/name)
    return decodes


def process_case(output, inputs, implementation_sha, plan_sha, previous_plan_sha):
    import numpy as np
    case = inputs["case"]
    fingerprint = {"implementation_sha256": implementation_sha, "plan_sha256": plan_sha, "previous_plan_sha256": previous_plan_sha,
                   "case": case, "source_paths": inputs["paths"], "source_sha256": inputs["hashes"], "indices": inputs["indices"],
                   "anchor_source_index": inputs["anchor_source_index"], "previous_6_step_replay": inputs["replay6"],
                   "predictions": [{key: item[key] for key in ("steps", "trace_path", "trace_sha256", "prediction_path", "prediction_sha256")} for item in inputs["items"]],
                   "previous6_bridge": {key: inputs["old_items"][2][key] for key in ("trace_path", "trace_sha256", "prediction_path", "prediction_sha256")}}
    signature = full.canonical_hash(fingerprint)
    target_dir = output/"samples"/case["id"]
    if target_dir.exists():
        report = load_json(target_dir/"verification.json")
        require(report.get("status") == "passed" and report.get("input_signature") == signature, "Cached case identity changed")
        for name, artifact in report["artifacts"].items(): require(digest(target_dir/name) == artifact["sha256"], "Cached output changed")
        metrics = load_json(target_dir/"metrics.json")
        require(metrics["input_signature"] == signature, "Cached metrics identity differs")
        return metrics, True
    staging = Path(tempfile.mkdtemp(prefix=f".{case['id']}-", dir=output/"samples"))
    try:
        w, h = inputs["width"], inputs["height"]
        ordinals = inputs["indices"]["future_source_indices"]
        rgb = base.read_source(inputs["paths"]["rgb"], ordinals, w, h)
        objects = base.read_source(inputs["paths"]["object"], [inputs["anchor_source_index"], *ordinals], w, h)
        target, static = objects[1:], np.broadcast_to(objects[0], objects[1:].shape)
        predictions, decodes = [], []
        for item in inputs["items"]:
            frames, metadata = base.read_prediction(item["prediction_path"], w, h, inputs["fps"])
            predictions.append(frames); decodes.append(metadata)
        old6, old6_decode = base.read_prediction(inputs["old_items"][2]["prediction_path"], w, h, inputs["fps"])
        delta = np.abs(old6.astype(np.int16)-predictions[2].astype(np.int16))
        replay = {"status": "pixel_identical" if np.array_equal(old6, predictions[2]) else "different",
                  "max_absolute_pixel_difference": int(delta.max()), "mean_absolute_pixel_difference": float(delta.mean()),
                  "scope": "This case's16decoded MP4frames of previous6 and current6 only;not a numerical-equivalence claim beyond the selected windows."}
        declared = inputs["replay6"]
        require(replay["status"] == declared["status"] and replay["max_absolute_pixel_difference"] == declared["decoded_max_abs_difference_uint8"]
                and math.isclose(replay["mean_absolute_pixel_difference"], declared["decoded_mean_abs_difference_uint8"], abs_tol=1e-12), "Independent6-step replay differs from runner declaration")
        metrics = {"schema": SCHEMA+".case_metrics", "status": "passed", "input_signature": signature,
                   **{key: case[key] for key in ("id", "task", "episode", "phase", "phase_eligible", "anchor")},
                   **measure(static, predictions, target), "previous_6_step_replay": replay,
                   "timing": {"current": {f"steps{item['steps']}": item["trace"]["timing"] for item in inputs["current_items"]},
                              "previous": {f"steps{item['steps']}": item["trace"]["timing"] for item in inputs["old_items"]}}}
        base.write_json(staging/"metrics.json", metrics)
        comparison_decode = render(staging, inputs, rgb, static, predictions, target)
        report = {"schema": SCHEMA+".case_verification", "status": "passed", "input_signature": signature, "input": fingerprint,
                  "matched_input": {key: inputs["items"][0]["trace"][key] for key in shared.TRACE_FIELDS if key != "num_inference_steps"},
                  "checkpoint": inputs["items"][0]["trace"]["checkpoint"], "sigma_shift": 3,
                  "input_tensor_sha256": inputs["items"][0]["trace"]["input_tensor_sha256"],
                  "initial_noise_sha256": inputs["items"][0]["trace"]["initial_noise_sha256"],
                  "static_anchor_raw_rgb_sha256": hashlib.sha256(objects[0].tobytes()).hexdigest(),
                  "prediction_decode": decodes, "previous6_decode": old6_decode, "independent_6_step_replay": replay,
                  "comparison_decode": comparison_decode, "target_supplied_to_model": False, "gpu_used": False,
                  "artifacts": {name: {"sha256": digest(staging/name), "bytes": (staging/name).stat().st_size}
                                for name in ("metrics.json", "comparison.mp4", "full_comparison.mp4", "contact.png", "object_contact.png")}}
        base.write_json(staging/"verification.json", report)
        require(not target_dir.exists(), "Case output appeared during processing")
        os.rename(staging, target_dir)
        return metrics, False
    finally:
        if staging.exists(): shutil.rmtree(staging)


def aggregate_group(cases, tasks):
    grouped = defaultdict(lambda: defaultdict(list))
    for case in cases: grouped[case["task"]][case["episode"]].append(case)
    per_task = {}
    for task, episodes in grouped.items():
        methods = {}
        for method in METHODS:
            methods[method] = {key: statistics.fmean(statistics.fmean(case["methods"][method][key] for case in rows) for rows in episodes.values()) for key in SCALARS}
            if method != "static":
                for out, source in (("mae_win_rate_vs_static", "mae_beats_static"), ("mae_tie_rate_vs_static", "mae_ties_static")):
                    methods[method][out] = statistics.fmean(statistics.fmean(float(case["methods"][method][source]) for case in rows) for rows in episodes.values())
        per_task[task] = {"case_count": sum(map(len, episodes.values())), "episode_count": len(episodes), "methods": methods}
    macro = None
    if per_task:
        macro = {method: {key: statistics.fmean(task["methods"][method][key] for task in per_task.values()) for key in next(iter(per_task.values()))["methods"][method]} for method in METHODS}
        frozen = macro["static"]["common_foreground_mae"]
        for method in METHODS[1:]:
            values = macro[method]
            gain = frozen-values["common_foreground_mae"]
            values.update(mae_improvement_vs_static=gain, relative_mae_improvement_vs_static=gain/frozen if frozen > EPS else None)
            for baseline in ("steps10", "steps6"):
                denominator = macro[baseline]["common_foreground_mae"]
                delta = values["common_foreground_mae"]-denominator
                values[f"mae_degradation_vs_{baseline}"] = delta
                values[f"relative_mae_degradation_vs_{baseline}"] = delta/denominator if denominator > EPS else None
                values[f"iou_degradation_vs_{baseline}"] = macro[baseline]["silhouette_iou"]-values["silhouette_iou"]
    timings = {cohort: {f"steps{n}": {key: previous.distribution([case["timing"][cohort][f"steps{n}"][key] for case in cases]) for key in TIMINGS}
                        for n in numbers} for cohort, numbers in (("current", STEPS), ("previous", (10, 8, 6)))}
    for method, values in timings["current"].items():
        for key, stat in values.items():
            reference, actual = timings["current"]["steps6"][key]["mean"], stat["mean"]
            stat["speedup_vs_current6_ratio_of_means"] = reference/actual if actual is not None and actual > EPS else None
            stat["relative_latency_reduction_vs_current6"] = 1-actual/reference if reference is not None and reference > EPS else None
    bridge = {}
    for key in TIMINGS:
        old, new = timings["previous"]["steps6"][key]["mean"], timings["current"]["steps6"][key]["mean"]
        bridge[key] = {"previous6_mean": old, "current6_mean": new,
                       "current_minus_previous_seconds": new-old if old is not None else None,
                       "current_vs_previous_ratio": new/old if old is not None and old > EPS else None}
    return {"case_count": len(cases), "covered_task_count": len(per_task), "expected_task_count": len(tasks),
            "missing_tasks": sorted(set(tasks)-set(per_task)), "task_macro": macro, "per_task": per_task,
            "timing": timings, "six_step_timing_bridge": bridge}


def aggregate(cases, plan):
    tasks = sorted({case["task"] for case in plan["cases"]})
    return {"overall": aggregate_group(cases, tasks), **{phase: aggregate_group([case for case in cases if case["phase"] == phase], tasks) for phase in PHASES}}


def definitions():
    return {"metric_target": "object_only_gt", "foreground_threshold_uint8": 16,
            "common_mask_rule": "Per pixel/frame maxRGB>16 union GT|static_O_s|previous10|previous8|current6|current4|current2. All six methods use the same MAE denominator. Previous6 is only bridge/replay and is not an extra mask member.",
            "mae_rule": "Pool16frames absolute RGB error on shared mask/(mask pixels*3*255);0..1,lower better. All old10/8 scores are recomputed,never copied from the old mask report.",
            "iou_rule": "Per-method foreground intersection/union with GT,pooled16frames;higher better.",
            "static_rule": "Raw original object video at condition_source_indices[-1],the current O_s,repeated16;not a prediction and not episode-first unless s=0.",
            "aggregation_rule": "Phase:equal episode means within task,then equal tasks. Overall:average two phases within episode,then episodes within task,then tasks. Partial missing-task coverage is explicit.",
            "win_and_degradation_rule": "MAE gain=static-model;win iff gain>1e-12,tie iff abs(gain)<=1e-12. Rates use same episode/task weights. Degradation vs10 or6=modelMAE-referenceMAE,positive=worse;relative divides by reference aggregate MAE,null if<=1e-12. IoU degradation=referenceIoU-modelIoU. Not action success.",
            "timing_rule": "Primary comparisons are current-round6/4/2 with current6 baseline. Previous10/8/6 distributions are a separate cohort,never represented as same-round timing. Report same-case previous6/current6 timing bridge separately. Mean/median/p95 pool matched cases;full24 coverage has equal task/phase counts. p95 uses NumPy linear interpolation.",
            "timing_scope": "Model time is synchronized infer_object_future including condition VAE/CLIP,N DiT calls,full25-frameVAE decode. Processing=sum pre-read-warmed condition load+model+export. Both exclude model loading,two separatewarmups,input hashchecks,trace writes,GC,and replay checks;not cold-start end-to-end latency.",
            "background_rule": "GT-background predicted foreground/intensity also include displaced foreground object errors;not pure background retention.",
            "empty_conventions": "Empty common maskMAE0;empty pairunionIoU1;zeroGTbackground rate/intensity0;near-zero reference relative ratiosnull.",
            "cross_report_warning": "New shared mask includes2/4-step outputs and can be larger. MAE values are not directly comparable to older reports;use this report's recomputed10/8/6/4/2 scores.",
            "scope": "Exploratory on the same24 previously examined test windows from12tasks;not independent validation or the complete50task test distribution. No task success or statistical significance claim."}


def csv_document(cases, groups):
    stream = io.StringIO(newline="")
    fields = ["row_type", "group", "task", "episode", "id", "phase", "anchor", "method", "cohort", *SCALARS,
              "mae_win_rate_vs_static", "mae_degradation_vs_steps10", "relative_mae_degradation_vs_steps10",
              "mae_degradation_vs_steps6", "relative_mae_degradation_vs_steps6", "timing_field", "timing_n", "mean", "median", "p95", "speedup_vs_current6"]
    writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
    for case in cases:
        for method in METHODS:
            writer.writerow({"row_type": "case", **{key: case[key] for key in ("task", "episode", "id", "phase", "anchor")}, "method": method,
                             **{key: value for key, value in case["methods"][method].items() if key in fields}})
    for name, stats in groups.items():
        for method, values in (stats["task_macro"] or {}).items():
            writer.writerow({"row_type": "task_macro", "group": name, "method": method, **{key: value for key, value in values.items() if key in fields}})
        for cohort, methods in stats["timing"].items():
            for method, values in methods.items():
                for field, numbers in values.items():
                    writer.writerow({"row_type": "timing_distribution", "group": name, "cohort": cohort, "method": method, "timing_field": field,
                                     "timing_n": numbers["n"], "mean": numbers["mean"], "median": numbers["median"], "p95": numbers["p95"],
                                     "speedup_vs_current6": numbers.get("speedup_vs_current6_ratio_of_means")})
    return stream.getvalue()


def page(cases, plan, status, groups):
    tables = []
    for name, stats in groups.items():
        rows = []
        for method, score in (stats["task_macro"] or {}).items():
            delta, wins = score.get("relative_mae_degradation_vs_steps6"), score.get("mae_win_rate_vs_static")
            rows.append(f"<tr><td>{method}</td><td>{score['common_foreground_mae']:.6f}</td><td>{score['silhouette_iou']:.6f}</td><td>{'—' if wins is None else f'{wins:.1%}'}</td><td>{'—' if delta is None else f'{delta:+.2%}'}</td></tr>")
        tables.append(f"<section><h2>{name}：{stats['case_count']} cases / {stats['covered_task_count']}/12 tasks</h2><table><tr><th>方法</th><th>共同前景MAE↓</th><th>IoU↑</th><th>MAE胜静止比例</th><th>相对本轮6步MAE变化↑为退化</th></tr>{''.join(rows)}</table></section>")
    timing_tables = []
    for cohort, methods in groups["overall"]["timing"].items():
        rows = []
        for method, fields in methods.items():
            for label, key in (("模型调用", "model_inference_seconds"), ("处理计时", "processing_seconds")):
                value = fields[key]
                if value["n"]:
                    speed = value.get("speedup_vs_current6_ratio_of_means")
                    rows.append(f"<tr><td>{method}</td><td>{label}</td><td>{value['n']}</td><td>{value['mean']:.3f}</td><td>{value['median']:.3f}</td><td>{value['p95']:.3f}</td><td>{'—' if speed is None else f'{speed:.3f}×'}</td></tr>")
        title = "本轮6/4/2：主速度比较，以本轮6步为基准" if cohort == "current" else "上一轮10/8/6：历史计时，非本轮同次实验"
        timing_tables.append(f"<section><h2>{title}</h2><table><tr><th>步数</th><th>范围</th><th>n</th><th>均值秒</th><th>中位秒</th><th>p95秒</th><th>本轮6步/当前耗时</th></tr>{''.join(rows)}</table></section>")
    bridge = groups["overall"]["six_step_timing_bridge"]
    bridge_rows = []
    for key in ("model_inference_seconds", "processing_seconds"):
        row = bridge[key]
        if row["previous6_mean"] is not None:
            bridge_rows.append(f"<tr><td>{key}</td><td>{row['previous6_mean']:.3f}</td><td>{row['current6_mean']:.3f}</td><td>{row['current_minus_previous_seconds']:+.4f}</td><td>{row['current_vs_previous_ratio']:.4f}</td></tr>")
    identical = sum(case["previous_6_step_replay"]["status"] == "pixel_identical" for case in cases)
    cards = []
    for case in cases:
        folder = f"samples/{case['id']}"
        cards.append(f'''<article data-task="{html.escape(case['task'])}" data-phase="{case['phase']}"><h3>{html.escape(case['task'])} / {html.escape(case['episode'])} / {case['phase']} / s={case['anchor']}</h3>
<video controls loop playsinline preload="none" src="{folder}/comparison.mp4"></video><p>主视频：全景RGB GT | static O_s | 本轮6 | 本轮4 | 本轮2；每格observer左、ego右。</p>
<details><summary>全部步数七列视频：RGB GT | static | 旧10 | 旧8 | 新6 | 新4 | 新2</summary><video controls loop playsinline preload="none" src="{folder}/full_comparison.mp4"></video></details>
<details><summary>物体GT七列对照：未来1/5/10/16帧</summary><a href="{folder}/object_contact.png"><img loading="lazy" src="{folder}/object_contact.png"></a></details>
<p><a href="{folder}/contact.png">本轮五列关键帧</a> · <a href="{folder}/metrics.json">逐帧指标/两轮计时</a> · <a href="{folder}/verification.json">来源与桥接校验</a></p></article>''')
    options = ''.join(f'<option>{html.escape(task)}</option>' for task in sorted({case['task'] for case in plan['cases']}))
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Checkpoint10000:10/8/6/4/2步对照</title>
<style>body{{margin:24px auto;padding:0 18px;max-width:2240px;background:#151b25;color:#eef3fc;font:16px/1.6 system-ui}}a{{color:#a9ceff}}section,article{{padding:18px;margin:24px 0;background:#253144}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border:1px solid #526174;text-align:left}}video,img{{display:block;width:100%;height:auto}}select{{padding:8px;margin:6px}}[hidden]{{display:none}}</style>
<h1>Checkpoint10000：本轮6/4/2步与此前10/8步</h1><p>状态：{status}；完成{len(cases)}/24窗口。{'部分结果，覆盖未齐，不作最终结论。' if status!='passed' else '12任务各一episode、两个eligible阶段全部齐全。'}</p>
<p>所有质量指标重新计算，共同mask=物体GT | static O_s | 旧10 | 旧8 | 新6 | 新4 | 新2。六种方法共用MAE分母；不能直接拼接或比较旧报告中不同mask的MAE。全景RGB GT只展示；静止基线取真实原始物体O_s重复16次，GT不输入模型。</p>
<p>速度主比较只使用本轮6/4/2，以本轮6步为参照；旧10/8/6另列历史计时。模型计时含条件编码、DiT、完整25帧VAE解码；处理计时为预读后的条件读取+模型+导出，排除加载、2warmups、检查哈希/trace写入/GC/重放。</p>
<p>新6↔旧6逐例独立解码重放：{identical}/{len(cases)}完全像素相同；{len(cases)-identical}存在差异。若有差异，质量仍用本轮6作配对基线，passed不等于重放全相同。输入张量、初始噪声、权重、seed0、shift3和dtype/内存策略跨轮核验相同。</p>
<p>这是已查看过test子集上的探索实验，不是独立验证集或全50任务结论；静止胜率不是动作成功率。</p><p><a href="metrics.json">完整指标JSON</a> · <a href="metrics.csv">CSV</a> · <a href="summary.json">来源摘要</a></p>
{''.join(tables)}{''.join(timing_tables)}<h2>6步跨轮计时桥接</h2><table><tr><th>范围</th><th>旧6均值秒</th><th>新6均值秒</th><th>新−旧秒</th><th>新/旧</th></tr>{''.join(bridge_rows)}</table>
<h2>逐例</h2><select id="task"><option value="">全部任务</option>{options}</select><select id="phase"><option value="">全部阶段</option><option>pre_onset</option><option>mid_motion</option></select>{''.join(cards)}
<script>function filter(){{document.querySelectorAll('article[data-task]').forEach(e=>{{e.hidden=['task','phase'].some(k=>document.getElementById(k).value&&e.dataset[k]!==document.getElementById(k).value)}})}}['task','phase'].forEach(k=>document.getElementById(k).addEventListener('change',filter));</script></html>'''


def evaluate(previous_run, run, output, allow_partial):
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/".evaluate.lock").open("a")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise ValueError("Another evaluator owns output")
    files = full.Files()
    plan, plan_sha, old_plan, old_sha, old_state, dataset, index = prepare_plans(previous_run, run, files)
    state, state_sha = previous.read_run(run/"run.json")
    validate_run(state, plan, plan_sha)
    if not allow_partial: require(state["status"] == "complete", "Run incomplete;use --allow-partial")
    implementation = {str(path.resolve()): digest(path) for path in (Path(__file__), HELPER, previous.HELPER, full.HELPER, shared.HELPER)}
    implementation_sha = full.canonical_hash(implementation)
    contract = {"schema": SCHEMA+".contract", "previous_run": str(previous_run), "run": str(run), "previous_plan_sha256": old_sha,
                "plan_sha256": plan_sha, "implementation": implementation, "definitions": definitions()}
    if (output/"evaluation_contract.json").exists(): require(load_json(output/"evaluation_contract.json") == contract, "Evaluation contract changed;use a new output")
    else: full.atomic_json(output/"evaluation_contract.json", contract)
    (output/"samples").mkdir(exist_ok=True)
    completed, pending, reused, caches = [], [], 0, set()
    for case in plan["cases"]:
        available = []
        for steps in STEPS:
            directory = run/"samples"/case["id"]/f"steps_{steps}"
            available.append((directory/"trace.json").is_file() and (directory/"future.mp4").is_file()
                             and load_json(directory/"trace.json").get("status") == "complete")
        if not all(available): pending.append({"id": case["id"], "complete": dict(zip(map(str, STEPS), available))}); continue
        state, state_sha = previous.read_run(run/"run.json")
        results = validate_run(state, plan, plan_sha)
        if not all((case["id"], steps) in results for steps in STEPS): pending.append({"id": case["id"], "reason": "awaiting run result publication"}); continue
        inputs = prepare_case(previous_run, run, plan, plan_sha, old_plan, old_sha, old_state, state, case, dataset, index, files, caches)
        metrics, used = process_case(output, inputs, implementation_sha, plan_sha, old_sha)
        completed.append(metrics); reused += used
        full.atomic_json(output/"progress.json", {"status": "partial", "completed_case_count": len(completed), "planned_case_count": 24,
                                                  "last_case": case["id"], "reused_case_count": reused})
        print(f"{'Reused' if used else 'Evaluated'} {len(completed)}/24:{case['id']}", flush=True)
    state, state_sha = previous.read_run(run/"run.json")
    validate_run(state, plan, plan_sha)
    final = len(completed) == 24 and state["status"] == "complete"
    require(allow_partial or final, "Missing matching predictions")
    status = "passed" if final else "partial"
    groups = aggregate(completed, plan)
    compact = [{**{key: value for key, value in case.items() if key != "methods"},
                "methods": {method: {key: value for key, value in values.items() if key != "per_future_frame"} for method, values in case["methods"].items()}} for case in completed]
    metrics = {"schema": SCHEMA+".metrics", "status": status, "case_count": len(completed), "planned_case_count": 24,
               **definitions(), "aggregates": groups, "cases": compact}
    html = page(completed, plan, status, groups)
    links = full.check_links(html, output)
    for path, expected in implementation.items(): require(digest(path) == expected, "Implementation changed during evaluation")
    for path, (_, expected) in list(files.entries.items()): files.sha(path, expected)
    contents = {"metrics.json": full.json_text(metrics), "metrics.csv": csv_document(completed, groups), "index.html": html}
    for name, value in contents.items(): full.atomic_text(output/name, value)
    identical = sum(case["previous_6_step_replay"]["status"] == "pixel_identical" for case in completed)
    summary = {"schema": SCHEMA, "status": status, "created_at_utc": datetime.now(timezone.utc).isoformat(), "case_count": len(completed),
               "planned_case_count": 24, "previous_run": str(previous_run), "run": str(run), "current_run_metadata": state,
               "current_run_json_sha256": state_sha, "previous_run_json_sha256": files.sha(previous_run/"run.json"),
               "plan_sha256": plan_sha, "previous_plan_sha256": old_sha, "pending": pending, "reused_case_count": reused,
               "independent_previous6_replay": {"checked_case_count": len(completed), "pixel_identical_count": identical,
                    "all_processed_cases_pixel_identical": identical == len(completed),
                    "max_absolute_pixel_difference": max((case["previous_6_step_replay"]["max_absolute_pixel_difference"] for case in completed), default=None),
                    "different_case_ids": [case["id"] for case in completed if case["previous_6_step_replay"]["status"] != "pixel_identical"],
                    "scope": "Only these selected cases'16decoded frames;passed evaluation status does not assert all replay outputs are identical."},
               "aggregates": groups, "scope": definitions()["scope"], "input_content_sha256": {path: value for path, (_, value) in files.entries.items()},
               "case_verifications": [{"id": case["id"], "path": f"samples/{case['id']}/verification.json",
                                        "sha256": digest(output/"samples"/case["id"]/"verification.json")} for case in completed],
               "implementation": implementation, "html_local_links_checked": links, "gpu_used": False, "model_loaded": False,
               "original_predictions_modified": False, "artifacts": {name: {"sha256": digest(output/name), "bytes": (output/name).stat().st_size} for name in contents}}
    full.atomic_json(output/"summary.json", summary)
    full.atomic_json(output/"progress.json", {"status": status, "completed_case_count": len(completed), "planned_case_count": 24, "reused_case_count": reused})
    print(full.json_text({"status": status, "case_count": len(completed), "output": str(output), "overall": groups["overall"]["task_macro"],
                          "current_timing": groups["overall"]["timing"]["current"], "six_step_bridge": groups["overall"]["six_step_timing_bridge"]}))


def self_test():
    import numpy as np
    gt = np.zeros((16, 8, 8, 3), dtype=np.uint8); gt[:, 1:3, 1:3] = 255
    static = np.zeros_like(gt); extra = gt.copy(); extra[:, 5:7, 5:7] = 255
    measured = measure(static, [gt, gt, gt, gt, extra], gt)
    require(measured["methods"]["static"]["common_foreground_mae"] == .5
            and measured["methods"]["steps2"]["common_foreground_mae"] == .5, "2step foreground must expand the shared denominator")
    require(measured["methods"]["steps2"]["silhouette_iou"] == .5 and measured["methods"]["steps6"]["mae_beats_static"] is True, "IoU/static win failed")
    require(measured["methods"]["steps2"]["relative_mae_degradation_vs_steps6"] is None, "Near-zero reference must yield null ratio")
    cases = []
    for task, count, value in (("A", 1, .2), ("B", 3, .8)):
        for episode in range(count):
            import copy
            methods = copy.deepcopy(measured["methods"])
            for method in METHODS: methods[method]["common_foreground_mae"] = value
            cases.append({"task": task, "episode": str(episode), "methods": methods,
                          "timing": {cohort: {f"steps{n}": {key: float(n)+(1 if cohort == "previous" else 0) for key in TIMINGS} for n in numbers}
                                     for cohort, numbers in (("current", STEPS), ("previous", (10, 8, 6)))}})
    aggregate = aggregate_group(cases, ["A", "B", "missing"])
    require(math.isclose(aggregate["task_macro"]["steps2"]["common_foreground_mae"], .5), "Task/episode weighting failed")
    require(aggregate["timing"]["current"]["steps2"]["model_inference_seconds"]["speedup_vs_current6_ratio_of_means"] == 3, "Current6 speed baseline failed")
    require(aggregate["six_step_timing_bridge"]["model_inference_seconds"]["current_minus_previous_seconds"] == -1, "Cohort bridge failed")
    require("speedup_vs_current6_ratio_of_means" not in aggregate["timing"]["previous"]["steps10"]["model_inference_seconds"], "Previous/current cohorts mixed")
    print(full.json_text({"status": "self_test_passed", "checks": ["seven-image union/six methods", "pairIoU", "staticwin", "zero relative denominator", "task-balanced aggregation", "current6 speed baseline", "separate previous cohort and6bridge"], "gpu_used": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", type=Path); parser.add_argument("--run", type=Path); parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-partial", action="store_true"); parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    if args.self_test: self_test(); return
    require(args.previous_run and args.run and args.output, "Provide --previous-run --run --output")
    require(args.previous_run.resolve() != args.run.resolve(), "Previous/current runs must differ")
    evaluate(args.previous_run.resolve(), args.run.resolve(), args.output.resolve(), args.allow_partial)


if __name__ == "__main__": main()

#!/usr/bin/env python3
"""Resumable CPU evaluation: object O_s freeze baseline versus step8000/10000.

Reads immutable paired inference runs. Processes one matched case at a time;
--allow-partial publishes explicitly partial summaries while inference runs.
No model, text encoder, VAE or GPU is used. Existing predictions are untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import csv
import fcntl
import hashlib
import html
from html.parser import HTMLParser
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import tempfile
from urllib.parse import unquote, urlsplit

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HELPER = Path(__file__).with_name("compare_obj_video_checkpoints.py")
spec = importlib.util.spec_from_file_location("_full_eval_metrics", HELPER)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
base = shared.base
require, load_json, digest = base.require, base.load_json, base.digest
METHODS = ("static", "step8000", "step10000")
PHASES = ("pre_onset", "mid_motion")
SCALARS = shared.SCALAR_METRICS
TIE_EPSILON = 1e-12
VERSION = "vpp2_obj.full_clean_evaluation.v1"
LABELS = ("Full-scene RGB GT", "Static: repeat object O_s", "Step 8000: object prediction", "Step 10000: object prediction")


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_text(path, content):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_text(path, json_text(value))


class Files:
    """Hash immutable sources once per invocation, checking stat before reuse."""
    def __init__(self):
        self.entries = {}

    def sha(self, path, expected=None):
        path = str(Path(path).resolve())
        stat = Path(path).stat()
        identity = (stat.st_size, stat.st_mtime_ns)
        if path in self.entries:
            old_identity, value = self.entries[path]
            require(identity == old_identity, f"Input changed during evaluation: {path}")
        else:
            value = digest(path)
            self.entries[path] = (identity, value)
        require(expected is None or value == expected, f"SHA256 differs: {path}")
        return value


def plan_and_dataset(runs, files):
    from vpp2_obj.infer import jsonable, load_config, make_dataset
    plans = [load_json(run / "selection.json") for run in runs]
    require(plans[0] == plans[1], "Two runs must use identical frozen selection plans")
    hashes = [files.sha(run / "selection.json") for run in runs]
    require(hashes[0] == hashes[1], "Selection file bytes differ")
    plan = plans[0]
    require(plan.get("checkpoint_steps") == [8000, 10000], "Unexpected checkpoint selection")
    cases = plan["cases"]
    require(plan["case_count"] == len(cases) == 480 and plan["episode_count"] == 240
            and plan["task_count"] == 50, "Expected all240 held-out episodes,50tasks,two phases")
    config, manifest = Path(plan["config"]).resolve(), Path(plan["test_manifest"]).resolve()
    files.sha(config, plan["config_sha256"])
    files.sha(manifest, plan["test_manifest_sha256"])
    dataset = make_dataset(load_config(config), manifest)
    require((dataset.history_frames, dataset.history_stride, dataset.prediction_frames, dataset.prediction_stride)
            == (8, 30, 16, 5), "Expected first-frame+8history(stride30),16future(stride5)")
    record_index = {(row["task"], row["episode"]): i for i, row in enumerate(dataset.records)}
    require(len(record_index) == len(dataset.records) == 240, "Expected unique full test manifest")
    require(len({task for task, _ in record_index}) == 50, "Expected50 test tasks")
    identities, ids, phases = set(), set(), defaultdict(set)
    for case in cases:
        identity = (case["task"], case["episode"], case["phase"])
        require(re.fullmatch(r"[A-Za-z0-9_-]+", case["id"]) and case["id"] not in ids, "Unsafe/duplicate case id")
        require(identity not in identities and identity[:2] in record_index, "Duplicate or non-test case")
        require(case["phase"] in PHASES and type(case["phase_eligible"]) is bool, "Invalid phase eligibility")
        require(case.get("inference_eligible") is True, "Every planned case must be inferable, including fallbacks")
        require(case.get("anchor_index", case["anchor"]) == case["anchor"] and type(case["anchor"]) is int,
                "Anchor fields disagree")
        require(case["seed"] == 0 and case["steps"] == 10, "Expected fixed seed0 and10 inference steps")
        require((case.get("fallback_reason") is None) == case["phase_eligible"], "Fallback reason/eligibility conflict")
        production = jsonable(dataset.window_indices(record_index[identity[:2]], case["anchor"], include_target=True))
        for key, value in production.items():
            if key in case:
                require(case[key] == value, f"Frozen plan differs from production sampling: {case['id']} {key}")
        phases[identity[:2]].add(case["phase"])
        ids.add(case["id"])
        identities.add(identity)
    require(set(phases) == set(record_index) and all(value == set(PHASES) for value in phases.values()),
            "Plan must include exactly both phases for every test episode")
    return plan, hashes[0], dataset, record_index


def batch_snapshot(run, plan, plan_sha, step):
    path = run / "batch_trace.json"
    if not path.is_file():
        return {"status": "not_started", "complete_ids": [], "path": str(path)}
    raw = path.read_bytes()
    batch = json.loads(raw)
    require(batch.get("plan_sha256") == plan_sha and batch.get("test_manifest_sha256") == plan["test_manifest_sha256"],
            "Batch selection/manifest fingerprint differs")
    require(shared.checkpoint_step(batch["checkpoint"]) == step, "Batch checkpoint differs")
    results = batch.get("results", [])
    require(len({item["id"] for item in results}) == len(results), "Duplicate batch result")
    planned = {case["id"] for case in plan["cases"]}
    require({item["id"] for item in results} <= planned, "Batch contains unexpected cases")
    complete = {item["id"] for item in results if item.get("status") == "complete"}
    if batch.get("status") == "complete":
        require(complete == planned and batch.get("case_count") == len(planned), "Batch claims complete with missing cases")
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "status": batch.get("status"),
            "checkpoint": batch["checkpoint"], "complete_ids": sorted(complete),
            "resident_replay_check_sha256": batch.get("resident_replay_check_sha256")}


def case_inputs(case, runs, plan, plan_sha, snapshots, dataset, record_index, files, cache_checked):
    from vpp2_obj.infer import jsonable
    index = record_index[(case["task"], case["episode"])]
    row = dataset.records[index]
    condition_indices = jsonable(dataset.window_indices(index, case["anchor"], include_target=False))
    indices = jsonable(dataset.window_indices(index, case["anchor"], include_target=True))
    require(indices["condition_indices"][-1] == case["anchor"], "Condition does not end at anchor")
    anchor_source = indices["condition_source_indices"][-1]
    if "anchor_source_index" in case:
        require(case["anchor_source_index"] == anchor_source, "Plan anchor raw index differs")
    for key, value in indices.items():
        if key in case:
            require(case[key] == value, f"Plan production mapping differs: {case['id']} {key}")
    require(len(indices["condition_indices"]) == 9 and len(indices["future_indices"]) == 16,
            "Expected9 RGB conditions and16 future frames")
    source = {"rgb": str(dataset._path(row["condition_video_path"]).resolve()),
              "object": str(dataset._path(row["target_video_path"]).resolve()),
              "dedup": str(dataset._path(row["dedup_index_path"]).resolve())}
    source_shas = {"rgb": files.sha(source["rgb"], base.video_digest(row, "rgb_sha256")),
                   "object": files.sha(source["object"], base.video_digest(row, "obj_sha256")),
                   "dedup": files.sha(source["dedup"], row["dedup_index_sha256"])}
    items = []
    for run, step, snapshot in zip(runs, (8000, 10000), snapshots):
        directory = run / "samples" / case["id"]
        trace_path, prediction = directory / "trace.json", directory / "future.mp4"
        trace = load_json(trace_path)
        expected = {"schema": "vpp2_obj.inference.v1", "status": "complete", "dry_run": False,
                    "exported_frames": 16, "decoded_frame_slice": [9, 25], "prefix_clamped": False,
                    "target_video_read": False, "target_first_frame_read": False,
                    "task": case["task"], "episode": case["episode"], "anchor_index": case["anchor"],
                    "phase": case["phase"], "phase_eligible": case["phase_eligible"], "plan_sha256": plan_sha,
                    "seed": case["seed"], "num_inference_steps": case["steps"]}
        for key, value in expected.items():
            require(trace.get(key) == value, f"{case['id']}: invalid trace {key}")
        require(trace["sampling"] == condition_indices, "Inference sampling differs from production condition-only mapping")
        require(not {"future_source_indices", "future_timestamps_us", "target_video_path", "video"}.intersection(trace["sampling"]),
                "Inference sampling includes target-only data")
        require(Path(trace["output_directory"]).resolve() == directory
                and Path(trace["video_path"]).resolve() == prediction, "Trace output path differs")
        require(Path(trace["config"]).resolve() == Path(plan["config"]).resolve()
                and trace["config_sha256"] == plan["config_sha256"]
                and Path(trace["manifest"]).resolve() == Path(plan["test_manifest"]).resolve()
                and trace["manifest_sha256"] == plan["test_manifest_sha256"], "Trace config/test split differs")
        require(Path(trace["condition_video_path"]).resolve() == Path(source["rgb"]), "Condition source differs")
        for key in ("prompt", "text_embedding_sha256", "text_embedding_cache_signature"):
            require(trace[key] == row[key], f"Manifest text binding differs: {key}")
        cache = dataset._path(row["text_embedding_path"]).resolve()
        require(Path(trace["text_embedding_path"]).resolve() == cache, "Text cache path differs")
        files.sha(cache, trace["text_embedding_sha256"])
        cache_key = (str(cache), tuple(trace["context_mask"]), trace["model_visible_text_tokens"])
        if cache_key not in cache_checked:
            shared.validate_cached_text({"trace": trace})
            cache_checked.add(cache_key)
        checkpoint = Path(trace["checkpoint"])
        require(checkpoint.is_file() and shared.checkpoint_step(checkpoint) == step, "Checkpoint step/file differs")
        require(snapshot.get("checkpoint") is not None
                and checkpoint.resolve() == Path(snapshot["checkpoint"]).resolve(), "Trace checkpoint differs from run batch checkpoint")
        stat = checkpoint.stat()
        require((stat.st_size, stat.st_mtime_ns) == (trace["checkpoint_bytes"], trace["checkpoint_mtime_ns"]),
                "Checkpoint stat identity changed")
        memory = trace["memory_strategy"]
        require(all(key in memory for key in ("method", "dit_dtype", "vae_dtype", "clip_dtype", "quantization")),
                "Trace must describe placement/dtypes/quantization")
        require(trace["condition_shape"] == [3, 9, *dataset.video_size]
                and trace["context_shape"] == [128, 4096] and Fraction(str(trace["display_fps"])) == 6,
                "Input shape/FPS contract differs")
        items.append({"step": step, "trace": trace, "trace_path": str(trace_path),
                      "trace_sha256": files.sha(trace_path), "prediction_path": str(prediction),
                      "prediction_sha256": files.sha(prediction, trace["prediction_sha256"])})
    for field in (*shared.TRACE_FIELDS, *shared.OPTIONAL_MATH_FIELDS, "phase"):
        require((field in items[0]["trace"], items[0]["trace"].get(field)) ==
                (field in items[1]["trace"], items[1]["trace"].get(field)), f"Models differ in {field}")
    return {"case": case, "source": source, "source_sha256": source_shas, "indices": indices,
            "anchor_source_index": anchor_source, "items": items, "width": dataset.video_size[1],
            "height": dataset.video_size[0], "fps": Fraction(6)}


def measure(static, predictions, gt):
    import numpy as np
    methods = [static, *predictions]
    require(all(array.shape == gt.shape == (16, *gt.shape[1:]) for array in methods), "Metric shape mismatch")
    common = np.logical_or.reduce([array.max(axis=-1) > 16 for array in (gt, *methods)])
    scores = {name: {**shared.metric_one(array, gt, common),
                     "per_future_frame": [shared.metric_one(array[i], gt[i], common[i]) for i in range(16)]}
              for name, array in zip(METHODS, methods)}
    frozen = scores["static"]["common_foreground_mae"]
    for name in METHODS[1:]:
        current = scores[name]
        gain = frozen - current["common_foreground_mae"]
        current["mae_improvement_vs_static"] = gain
        current["relative_mae_improvement_vs_static"] = gain / frozen if frozen > TIE_EPSILON else None
        current["mae_beats_static"] = gain > TIE_EPSILON
        current["mae_ties_static"] = abs(gain) <= TIE_EPSILON
        current["iou_improvement_vs_static"] = current["silhouette_iou"] - scores["static"]["silhouette_iou"]
    return {"common_foreground_pixels": int(common.sum()),
            "common_foreground_pixels_per_frame": common.sum(axis=(1, 2)).tolist(), "methods": scores}


def render_case(directory, inputs, rgb, static, predictions, gt):
    import av
    import numpy as np
    from PIL import Image, ImageDraw
    w, h, fps = inputs["width"], inputs["height"], inputs["fps"]
    case = inputs["case"]
    arrays = [rgb, static, *predictions]
    def timestamp(i):
        us = inputs["indices"]["future_timestamps_us"][i]
        anchor_us = inputs["indices"]["condition_timestamps_us"][-1]
        return f"F{i+1}/16 | sim {us/1e6:.3f}s ({(us-anchor_us)/1e6:+.3f}s)"
    with av.open(str(directory / "comparison.mp4"), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 4*w, h+76, "yuv420p"
        stream.options = {"crf": "18", "preset": "fast", "threads": "1"}
        for i in range(16):
            canvas = Image.new("RGB", (4*w, h+76), "#202936")
            draw = ImageDraw.Draw(canvas)
            for col, (label, frames) in enumerate(zip(LABELS, arrays)):
                draw.text((col*w+8, 5), label, font=base.font(17), fill="white")
                canvas.paste(Image.fromarray(frames[i]), (col*w, 76))
            draw.text((10, 31), f"{case['task']} / {case['episode']} / {case['phase']} | anchor {case['anchor']} | {timestamp(i)}", font=base.font(16), fill="white")
            draw.text((10, 54), "Observer left / ego right in each panel;6fps playback is not simulation time", font=base.font(14), fill="#d5dfef")
            frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")
            frame.pts, frame.time_base = i, 1/fps
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    for object_reference, name in ((False, "contact.png"), (True, "object_contact.png")):
        canvas = Image.new("RGB", (4*w, 76 + 4*(h+32)), "#202936")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 6), f"{case['task']} / {case['episode']} / {case['phase']} | eligible={case['phase_eligible']} | anchor {case['anchor']}", font=base.font(20), fill="white")
        labels = (("Object-only GT", *LABELS[1:]) if object_reference else LABELS)
        for col, label in enumerate(labels):
            draw.text((col*w+8, 43), label, font=base.font(17), fill="white")
        for row, i in enumerate((0, 4, 9, 15)):
            y = 76 + row*(h+32)
            draw.text((10, y+4), timestamp(i), font=base.font(17), fill="white")
            for col, frames in enumerate([gt if object_reference else rgb, static, *predictions]):
                canvas.paste(Image.fromarray(frames[i]), (col*w, y+32))
        canvas.save(directory / name)
    _, decoded = base.read_prediction(directory / "comparison.mp4", 4*w, h+76, fps)
    return decoded


def process_case(output, inputs, implementation_sha, plan_sha):
    import numpy as np
    case = inputs["case"]
    fingerprint = {"implementation": implementation_sha, "plan": plan_sha, "case": case,
                   "source": inputs["source"], "source_sha256": inputs["source_sha256"],
                   "anchor_source_index": inputs["anchor_source_index"], "indices": inputs["indices"],
                   "predictions": [{key: item[key] for key in ("step", "trace_path", "trace_sha256", "prediction_path", "prediction_sha256")} for item in inputs["items"]]}
    signature = canonical_hash(fingerprint)
    target = output / "samples" / case["id"]
    if target.exists():
        verification = load_json(target / "verification.json")
        require(verification.get("status") == "passed" and verification.get("input_signature") == signature,
                f"Cannot reuse changed case: {case['id']}; preserve prior report and choose a new output directory")
        for name, artifact in verification["artifacts"].items():
            require(digest(target/name) == artifact["sha256"], f"Cached report artifact changed: {target/name}")
        metrics = load_json(target / "metrics.json")
        require(metrics["input_signature"] == signature, "Cached metric identity differs")
        return metrics, True
    staging = Path(tempfile.mkdtemp(prefix=f".{case['id']}-", dir=output / "samples"))
    try:
        w, h = inputs["width"], inputs["height"]
        ordinals = inputs["indices"]["future_source_indices"]
        rgb = base.read_source(inputs["source"]["rgb"], ordinals, w, h)
        object_frames = base.read_source(inputs["source"]["object"], [inputs["anchor_source_index"], *ordinals], w, h)
        gt = object_frames[1:]
        static = np.broadcast_to(object_frames[0], gt.shape)
        predictions, decoded = [], []
        for item in inputs["items"]:
            array, metadata = base.read_prediction(item["prediction_path"], w, h, inputs["fps"])
            predictions.append(array)
            decoded.append(metadata)
        metrics = {"schema": VERSION + ".case_metrics", "status": "passed", "input_signature": signature,
                   "id": case["id"], "task": case["task"], "episode": case["episode"],
                   "anchor": case["anchor"], "phase": case["phase"], "phase_eligible": case["phase_eligible"],
                   "fallback_reason": case.get("fallback_reason"), **measure(static, predictions, gt)}
        base.write_json(staging / "metrics.json", metrics)
        comparison_decode = render_case(staging, inputs, rgb, static, predictions, gt)
        verification = {"schema": VERSION + ".case_verification", "status": "passed", "input_signature": signature,
                        "input": fingerprint, "matched_input": {key: inputs["items"][0]["trace"][key] for key in shared.TRACE_FIELDS},
                        "baseline_definition": "Decode original object source at condition_source_indices[-1] (current anchor O_s), then repeat this same decoded RGB frame16 times. Never use a prediction frame as baseline.",
                        "static_anchor_raw_rgb_sha256": hashlib.sha256(object_frames[0].tobytes()).hexdigest(),
                        "prediction_decode": decoded, "comparison_decode": comparison_decode,
                        "checkpoint_identity": [{key: item["trace"][key] for key in ("checkpoint", "checkpoint_bytes", "checkpoint_mtime_ns")} for item in inputs["items"]],
                        "target_supplied_to_inference": False, "gpu_used": False,
                        "artifacts": {name: {"sha256": digest(staging/name), "bytes": (staging/name).stat().st_size}
                                      for name in ("metrics.json", "comparison.mp4", "contact.png", "object_contact.png")}}
        base.write_json(staging / "verification.json", verification)
        require(not target.exists(), "Case output appeared during processing")
        os.rename(staging, target)
        return metrics, False
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def task_aggregate(cases, expected_tasks):
    """Within task: case means per episode, then equal episode means; then equal tasks."""
    grouped = defaultdict(lambda: defaultdict(list))
    for case in cases:
        grouped[case["task"]][case["episode"]].append(case)
    per_task = {}
    for task, episodes in sorted(grouped.items()):
        scores = {}
        for method in METHODS:
            scores[method] = {key: statistics.fmean(statistics.fmean(case["methods"][method][key] for case in rows)
                                                  for rows in episodes.values()) for key in SCALARS}
            if method != "static":
                scores[method]["mae_win_rate_vs_static"] = statistics.fmean(statistics.fmean(float(case["methods"][method]["mae_beats_static"]) for case in rows) for rows in episodes.values())
                scores[method]["mae_tie_rate_vs_static"] = statistics.fmean(statistics.fmean(float(case["methods"][method]["mae_ties_static"]) for case in rows) for rows in episodes.values())
        per_task[task] = {"case_count": sum(map(len, episodes.values())), "episode_count": len(episodes), "methods": scores}
    aggregate = None
    if per_task:
        aggregate = {method: {key: statistics.fmean(value["methods"][method][key] for value in per_task.values())
                              for key in next(iter(per_task.values()))["methods"][method]} for method in METHODS}
        for method in METHODS[1:]:
            frozen = aggregate["static"]["common_foreground_mae"]
            gain = frozen - aggregate[method]["common_foreground_mae"]
            aggregate[method]["mae_improvement_vs_static"] = gain
            aggregate[method]["relative_mae_improvement_vs_static"] = gain/frozen if frozen > TIE_EPSILON else None
            aggregate[method]["iou_improvement_vs_static"] = aggregate[method]["silhouette_iou"] - aggregate["static"]["silhouette_iou"]
    missing = sorted(set(expected_tasks)-set(per_task))
    return {"case_count": len(cases), "episode_count": len({(c["task"], c["episode"]) for c in cases}),
            "covered_task_count": len(per_task), "expected_task_count": len(expected_tasks), "missing_tasks": missing,
            "task_macro_over_covered_tasks": aggregate, "full_50_task_macro": aggregate if not missing and len(expected_tasks) == 50 else None,
            "per_task": per_task}


def aggregate(cases, plan):
    tasks = sorted({case["task"] for case in plan["cases"]})
    groups = {}
    for phase in PHASES:
        for subset in ("eligible", "fallback", "all"):
            selected = [case for case in cases if case["phase"] == phase
                        and (subset == "all" or case["phase_eligible"] == (subset == "eligible"))]
            groups[f"{phase}_{subset}"] = task_aggregate(selected, tasks)
    # Overall descriptive coverage score averages the two phase cases within each
    # episode, then episodes within task, then tasks. It includes fallback cases.
    groups["overall_all_cases_including_fallback"] = task_aggregate(cases, tasks)
    return groups


def definitions():
    return {"metric_target": "object_only_gt", "foreground_threshold_uint8": 16,
            "common_mask_rule": "Per pixel/frame: union of maxRGB>16 for objectGT, frozen object O_s, prediction8000 and prediction10000. All three methods use the same MAE denominator.",
            "mae_rule": "Pool absolute uint8 RGB differences over16 frames on the common mask, divide by(common pixels*3*255). Lower is better; range0..1.",
            "iou_rule": "Each method uses its own conventional foreground intersection/union with objectGT, pooled over16 frames. Higher is better.",
            "baseline_rule": "Original object video raw frame at current anchor_source_index=condition_source_indices[-1], repeated16 times. Not a model prediction, not the episode first frame unless anchor=0.",
            "aggregation_rule": "For each phase/subset: equal case means within episode, equal episode means within each task, then equal task means. With one case per episode/phase this is simply episode-equal within task. full_50_task_macro exists only if all50 tasks have samples; otherwise covered-task macro and missing task list are explicit.",
            "overall_rule": "overall_all_cases_including_fallback averages the two phase cases within each episode, then episodes within task, then50tasks. Includes fallbacks, so it is a coverage score, not an eligible-motion-stage score. During partial inference the observed case set may be unbalanced and is not a final estimate.",
            "eligibility_rule": "Frozen selection phase_eligible defines eligible pre_onset/mid_motion subsets; all ineligible fallback cases remain in separate phase_fallback and phase_all reports. No fallback is silently counted as eligible onset/motion.",
            "relative_improvement_rule": "At aggregate level, (macro MAE_static-macro MAE_model)/macro MAE_static. Null if denominator<=1e-12; not the mean of individual relative ratios. Case-level ratios are also stored, null for near-zero static error.",
            "win_rate_rule": "Case win if staticMAE-modelMAE>1e-12; tie if abs(delta)<=1e-12. Episode/task-balanced mean of these indicators, not action success. Positive MAE improvement is better.",
            "background_rule": "GT-background predicted-foreground rate=count(pred maxRGB>16 and GT maxRGB<=16)/GT-background pixels; average intensity=sum RGB there/(pixels*3*255). Both can increase from object position/timing error and are not pure background-retention measures.",
            "empty_conventions": "Empty common mask MAE0; empty pair union IoU1; zero GT-background denominator gives rate/intensity0; zero static error relative improvement null.",
            "cross_report_warning": "This mask includes static O_s and two predictions; MAE is not directly comparable with previous4000/8000/10000 three-prediction reports or2000/4000 reports.",
            "limitations": "All held-out episodes are covered with two fixed phase records, not all possible windows or necessarily two distinct temporal windows. Fallbacks may reuse the same episode/anchor across phases. No detected pixel-displacement onset does not establish absolute stillness. Pixel metrics depend on timing, visibility, geometry and lossy encoding; they do not measure task success or establish significance."}


def compact_case(case):
    return {**{key: value for key, value in case.items() if key != "methods"},
            "methods": {name: {key: value for key, value in values.items() if key != "per_future_frame"}
                        for name, values in case["methods"].items()}}


def csv_document(cases, aggregates):
    buffer = io.StringIO(newline="")
    fields = ["row_type", "group", "task", "episode", "case_id", "phase", "phase_eligible", "anchor", "method",
              *SCALARS, "mae_improvement_vs_static", "relative_mae_improvement_vs_static", "mae_win_rate_vs_static", "mae_tie_rate_vs_static"]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for case in cases:
        for method in METHODS:
            values = case["methods"][method]
            row = {"row_type": "case", "task": case["task"], "episode": case["episode"], "case_id": case["id"],
                   "phase": case["phase"], "phase_eligible": case["phase_eligible"], "anchor": case["anchor"], "method": method,
                   **{key: values.get(key) for key in fields if key in values}}
            writer.writerow(row)
    for group, stats in aggregates.items():
        for method, values in (stats["task_macro_over_covered_tasks"] or {}).items():
            writer.writerow({"row_type": "covered_task_macro", "group": group, "method": method,
                             **{key: value for key, value in values.items() if key in fields}})
        for task, stats_task in stats["per_task"].items():
            for method, values in stats_task["methods"].items():
                writer.writerow({"row_type": "task_episode_mean", "group": group, "task": task, "method": method,
                                 **{key: value for key, value in values.items() if key in fields}})
    return buffer.getvalue()


def page(cases, plan, status, aggregates):
    esc = html.escape
    tasks = sorted({case["task"] for case in plan["cases"]})
    tables = []
    order = ["overall_all_cases_including_fallback", *[name for name in aggregates if name != "overall_all_cases_including_fallback"]]
    for name in order:
        group = aggregates[name]
        rows = []
        for method, scores in (group["task_macro_over_covered_tasks"] or {}).items():
            gain = scores.get("relative_mae_improvement_vs_static")
            wins = scores.get("mae_win_rate_vs_static")
            rows.append(f"<tr><td>{method}</td><td>{scores['common_foreground_mae']:.6f}</td><td>{scores['silhouette_iou']:.6f}</td><td>{'-' if gain is None else f'{gain:.2%}'}</td><td>{'-' if wins is None else f'{wins:.2%}'}</td></tr>")
        tables.append(f"<details open><summary>{name} — {group['case_count']} cases / {group['covered_task_count']}/50 tasks</summary><p>缺少任务：{esc(', '.join(group['missing_tasks'])) or '无'}</p><table><tr><th>方法</th><th>共同前景MAE↓</th><th>IoU↑</th><th>相对静止MAE改进↑</th><th>MAE胜过静止的比例</th></tr>{''.join(rows)}</table></details>")
    cards = []
    for case in cases:
        folder = f"samples/{case['id']}"
        cards.append(f'''<article data-task="{esc(case['task'])}" data-phase="{case['phase']}" data-eligible="{str(case['phase_eligible']).lower()}"><h3>{esc(case['task'])} / {esc(case['episode'])} / {case['phase']} / s={case['anchor']}</h3>
<p>phase eligible={case['phase_eligible']}；fallback：{esc(case.get('fallback_reason') or '无')}</p>
<details><summary>四列视频：全景 RGB GT | 静止物体 O_s | 8000 | 10000</summary><video controls loop playsinline preload="none" src="{folder}/comparison.mp4"></video></details>
<details><summary>物体 GT 与三种方法：未来1/5/10/16帧</summary><a href="{folder}/object_contact.png"><img loading="lazy" src="{folder}/object_contact.png"></a></details>
<p><a href="{folder}/contact.png">全景 GT 四时刻图</a> · <a href="{folder}/metrics.json">逐帧指标</a> · <a href="{folder}/verification.json">来源校验</a></p></article>''')
    options = ''.join(f'<option value="{esc(task)}">{esc(task)}</option>' for task in tasks)
    temporal = temporal_coverage(plan["cases"])
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Full clean evaluation:8000/10000/static</title>
<style>body{{margin:24px auto;padding:0 18px;max-width:1800px;background:#151b25;color:#eef3fc;font:16px/1.6 system-ui}}a{{color:#a9ceff}}article,details{{margin:16px 0;padding:16px;background:#253144}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border:1px solid #526174;text-align:left}}video,img{{display:block;width:100%;height:auto}}select{{padding:8px;margin:5px}}.warning{{background:#593d1f;padding:16px}}[hidden]{{display:none}}</style>
<h1>完整 clean 测试：静止基线 / checkpoint8000 /10000</h1><p class="warning">状态：{status}；已处理{len(cases)}/{len(plan['cases'])}窗口。{'这是部分结果，覆盖未齐，不能当作最终全测试指标。' if status != 'passed' else '240 episodes ×2窗口已齐。'}</p>
<p>每格observer左、ego右。全景RGB GT仅展示；计分目标为物体GT。静止基线为当前真实物体帧O_s重复16次，GT不传入模型。按phase筛选eligible，fallback单列；先task内episode等权再task等权。若未覆盖50task，只报告已覆盖task宏平均并列出缺失任务。</p>
<p>计划有{temporal['phase_record_count']}条阶段记录、{temporal['unique_temporal_windows']}个不同(task,episode,anchor)时间窗口；{temporal['duplicated_temporal_window_count']}组窗口由跨阶段fallback重复保留，因此不能称480条独立轨迹。“未检出onset”仅指像素位移阈值未触发，不代表物体绝对静止。</p>
<p>本报告共同mask为GT|static|8k|10k，不能直接与旧三checkpoint或2000/4000报告比较MAE。MAE胜率不是动作成功率；GT背景上的预测前景率也受对象位移误差影响。整体分数含fallback，仅用于全样本覆盖比较。</p>
<p><a href="metrics.json">全部指标/定义JSON</a> · <a href="metrics.csv">CSV</a> · <a href="summary.json">进度与来源摘要</a></p>{''.join(tables)}
<h2>按任务与阶段浏览</h2><label>任务<select id="task"><option value="">全部</option>{options}</select></label><label>阶段<select id="phase"><option value="">全部</option><option>pre_onset</option><option>mid_motion</option></select></label><label>资格<select id="eligible"><option value="">全部</option><option value="true">eligible</option><option value="false">fallback</option></select></label>{''.join(cards)}
<script>function filter(){{document.querySelectorAll('article[data-task]').forEach(e=>{{e.hidden=['task','phase','eligible'].some(k=>document.getElementById(k).value && e.dataset[k]!==document.getElementById(k).value)}})}}['task','phase','eligible'].forEach(k=>document.getElementById(k).addEventListener('change',filter));</script></html>'''


def check_links(document, output):
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.values = []
        def handle_starttag(self, tag, attrs):
            self.values.extend(value for key, value in attrs if key in ("href", "src"))
    parser = Links()
    parser.feed(document)
    for value in parser.values:
        uri = urlsplit(value)
        require(not uri.scheme, "Only local links expected")
        path = unquote(uri.path)
        require(path in ("metrics.json", "metrics.csv", "summary.json") or (output/path).is_file(), f"Missing report link:{value}")
    return len(parser.values)


def coverage(plan, completed):
    done = {case["id"] for case in completed}
    result = {}
    for phase in PHASES:
        expected = [case for case in plan["cases"] if case["phase"] == phase]
        result[phase] = {"planned": len(expected), "completed": sum(case["id"] in done for case in expected),
                         "eligible_planned": sum(case["phase_eligible"] for case in expected),
                         "eligible_completed": sum(case["phase_eligible"] and case["id"] in done for case in expected),
                         "fallback_planned": sum(not case["phase_eligible"] for case in expected),
                         "fallback_reason_counts": dict(Counter(case.get("fallback_reason") for case in expected if not case["phase_eligible"])),
                         "per_task": {task: {"planned": sum(case["task"] == task for case in expected),
                                             "eligible_planned": sum(case["task"] == task and case["phase_eligible"] for case in expected),
                                             "completed": sum(case["task"] == task and case["id"] in done for case in expected)}
                                      for task in sorted({case["task"] for case in expected})}}
    return result


def temporal_coverage(cases):
    grouped = defaultdict(list)
    for case in cases:
        grouped[(case["task"], case["episode"], case["anchor"])].append(case)
    duplicates = [{"task": task, "episode": episode, "anchor": anchor,
                   "record_ids": [case["id"] for case in rows], "phases": [case["phase"] for case in rows],
                   "all_records_fallback": all(not case["phase_eligible"] for case in rows)}
                  for (task, episode, anchor), rows in grouped.items() if len(rows) > 1]
    return {"phase_record_count": len(cases), "unique_temporal_windows": len(grouped),
            "duplicated_temporal_window_count": len(duplicates),
            "extra_phase_records_at_repeated_windows": len(cases)-len(grouped), "duplicate_windows": duplicates}


def evaluate(runs, output, allow_partial, execution_check=None):
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".evaluate.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError("Another evaluator owns this output directory")
    files = Files()
    plan, plan_sha, dataset, record_index = plan_and_dataset(runs, files)
    implementation = {str(Path(__file__).resolve()): digest(__file__), str(HELPER): digest(HELPER),
                      str(shared.HELPER): digest(shared.HELPER)}
    implementation_sha = canonical_hash(implementation)
    contract = {"schema": VERSION + ".contract", "plan_sha256": plan_sha, "runs": [str(run) for run in runs],
                "implementation_sha256": implementation_sha, "implementation_files": implementation,
                "config_sha256": plan["config_sha256"], "test_manifest_sha256": plan["test_manifest_sha256"],
                "case_count": len(plan["cases"]), "metric_definitions": definitions()}
    contract_path = output / "evaluation_contract.json"
    if contract_path.exists():
        require(load_json(contract_path) == contract, "Evaluation contract changed; use a new output directory")
    else:
        atomic_json(contract_path, contract)
    (output / "samples").mkdir(exist_ok=True)
    snapshots = [batch_snapshot(run, plan, plan_sha, step) for run, step in zip(runs, (8000, 10000))]
    if not allow_partial:
        require(all(snapshot["status"] == "complete" for snapshot in snapshots), "Batches incomplete; use --allow-partial for explicitly partial progress")
    completed, pending, reused, cache_checked = [], [], 0, set()
    replay = None
    if execution_check is not None:
        replay_report = load_json(execution_check)
        require(replay_report.get("status") == "passed", "Execution-strategy replay check has not passed")
        replay_sha = files.sha(execution_check)
        for snapshot in snapshots:
            if snapshot.get("resident_replay_check_sha256") is not None:
                require(snapshot["resident_replay_check_sha256"] == replay_sha, "Batch replay report binding differs")
        replay = {"path": str(execution_check), "sha256": replay_sha, "report": replay_report,
                  "scope_note": "Only the explicitly replayed windows were checked for decoded-pixel identity; this is not proof of equivalence for all evaluation windows."}
    matched_strategy = None
    for case in plan["cases"]:
        available = []
        for run in runs:
            directory = run / "samples" / case["id"]
            trace = directory / "trace.json"
            available.append(trace.is_file() and (directory/"future.mp4").is_file()
                             and load_json(trace).get("status") == "complete")
        if not all(available):
            pending.append({"id": case["id"], "step8000_complete": available[0], "step10000_complete": available[1]})
            continue
        inputs = case_inputs(case, runs, plan, plan_sha, snapshots, dataset, record_index, files, cache_checked)
        strategy = inputs["items"][0]["trace"]["memory_strategy"]
        if matched_strategy is None:
            matched_strategy = strategy
        require(strategy == matched_strategy, "Inference strategy changed across full-eval cases")
        metrics, was_reused = process_case(output, inputs, implementation_sha, plan_sha)
        completed.append(metrics)
        reused += was_reused
        atomic_json(output / "progress.json", {"schema": VERSION + ".progress", "status": "partial",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(), "completed_case_count": len(completed),
                    "planned_case_count": len(plan["cases"]), "last_case": case["id"], "reused_case_count": reused})
        print(f"{'Reused' if was_reused else 'Evaluated'} {len(completed)}/{len(plan['cases'])}: {case['id']}", flush=True)
    snapshots = [batch_snapshot(run, plan, plan_sha, step) for run, step in zip(runs, (8000, 10000))]
    if replay is not None:
        for snapshot in snapshots:
            require(snapshot.get("resident_replay_check_sha256") in (None, replay["sha256"]), "Replay report binding changed")
    final = len(completed) == len(plan["cases"]) and all(snapshot["status"] == "complete" for snapshot in snapshots)
    status = "passed" if final else "partial"
    require(allow_partial or final, "Missing matched predictions; use --allow-partial")
    aggregates = aggregate(completed, plan)
    metrics_document = {"schema": VERSION + ".metrics", "status": status, "case_count": len(completed),
                        "planned_case_count": len(plan["cases"]), **definitions(), "aggregates": aggregates,
                        "coverage": coverage(plan, completed), "planned_temporal_coverage": temporal_coverage(plan["cases"]),
                        "completed_temporal_coverage": temporal_coverage(completed), "cases": [compact_case(case) for case in completed]}
    document = page(completed, plan, status, aggregates)
    links = check_links(document, output)
    for path, expected in implementation.items():
        require(digest(path) == expected, "Evaluation implementation changed while running")
    for path, (_, expected) in list(files.entries.items()):
        files.sha(path, expected)
    contents = {"metrics.json": json_text(metrics_document), "metrics.csv": csv_document(completed, aggregates), "index.html": document}
    for name, content in contents.items():
        atomic_text(output/name, content)
    summary = {"schema": VERSION, "status": status, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "completed_case_count": len(completed), "planned_case_count": len(plan["cases"]),
               "case_count": len(completed), "episode_count": len({(case['task'],case['episode']) for case in completed}),
               "plan_sha256": plan_sha, "runs": [str(run) for run in runs], "batches": snapshots,
               "coverage": metrics_document["coverage"], "pending": pending, "reused_case_count": reused,
               "planned_temporal_coverage": metrics_document["planned_temporal_coverage"],
               "completed_temporal_coverage": metrics_document["completed_temporal_coverage"], "execution_replay_check": replay,
               "memory_strategy": matched_strategy, "aggregates": aggregates,
               "input_content_sha256": {path: value for path, (_, value) in files.entries.items()},
               "case_verifications": [{"id": case["id"], "path": f"samples/{case['id']}/verification.json",
                                        "sha256": digest(output/"samples"/case["id"]/"verification.json")} for case in completed],
               "implementation": implementation, "html_local_links_checked": links,
               "ground_truth_usage": "Used in frozen selection and post-inference baseline/metrics/display; checked inference traces say no target video/first target frame read.",
               "original_predictions_modified": False, "model_loaded": False, "gpu_used": False,
               "artifacts": {name: {"sha256": digest(output/name), "bytes": (output/name).stat().st_size} for name in contents}}
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "progress.json", {"status": status, "completed_case_count": len(completed),
                                          "planned_case_count": len(plan["cases"]), "reused_case_count": reused})
    print(json_text({"status": status, "completed_case_count": len(completed), "planned_case_count": len(plan["cases"]),
                     "output": str(output), "eligible_phase_scores": {phase: aggregates[f"{phase}_eligible"]["task_macro_over_covered_tasks"] for phase in PHASES}}))


def self_test():
    import numpy as np
    gt = np.zeros((16, 8, 8, 3), dtype=np.uint8)
    gt[:, 1:3, 1:3] = 255
    static = np.zeros_like(gt)
    extra = gt.copy()
    extra[:, 5:7, 5:7] = 255
    scores = measure(static, [gt, extra], gt)
    require(scores["methods"]["static"]["common_foreground_mae"] == .5, "Static common denominator wrong")
    require(scores["methods"]["step8000"]["common_foreground_mae"] == 0
            and scores["methods"]["step10000"]["common_foreground_mae"] == .5, "Three-way shared MAE denominator wrong")
    require(scores["methods"]["step8000"]["mae_beats_static"] is True
            and scores["methods"]["step10000"]["mae_ties_static"] is True, "Win/tie indicators wrong")
    empty = measure(static, [static, static], static)
    require(empty["methods"]["step8000"]["relative_mae_improvement_vs_static"] is None, "Zero denominator must be null")
    cases = []
    for task, count, value in (("taskA", 1, .2), ("taskB", 3, .8)):
        for episode in range(count):
            methods = json.loads(json.dumps(scores["methods"]))
            for method in METHODS:
                methods[method]["common_foreground_mae"] = value
            cases.append({"task": task, "episode": str(episode), "methods": methods})
    agg = task_aggregate(cases, ["taskA", "taskB", "missing"])
    require(math.isclose(agg["task_macro_over_covered_tasks"]["static"]["common_foreground_mae"], .5), "Task weights became episode weights")
    require(agg["full_50_task_macro"] is None and agg["missing_tasks"] == ["missing"], "Missing task coverage hidden")
    duplicate_phase = {**cases[0], "methods": json.loads(json.dumps(cases[0]["methods"]))}
    for method in METHODS:
        duplicate_phase["methods"][method]["common_foreground_mae"] = .6
    agg2 = task_aggregate([*cases, duplicate_phase], ["taskA", "taskB"])
    require(math.isclose(agg2["task_macro_over_covered_tasks"]["static"]["common_foreground_mae"], .6), "Phase/episode nesting incorrect")
    print(json_text({"status": "self_test_passed", "checks": ["common mask includes static and both predictions", "pair IoU", "wins/ties", "zero static denominator", "task-balanced mean", "missing tasks explicit", "within-episode phase mean"], "gpu_used": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run8", type=Path)
    parser.add_argument("--run10", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--execution-check", type=Path, help="Optional scoped execution-strategy replay report to link and hash")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    if args.self_test:
        self_test()
        return
    require(args.run8 and args.run10 and args.output, "Provide --run8 --run10 --output")
    runs = [args.run8.resolve(), args.run10.resolve()]
    require(runs[0] != runs[1], "Prediction runs must be distinct")
    evaluate(runs, args.output.resolve(), args.allow_partial, args.execution_check.resolve() if args.execution_check else None)


if __name__ == "__main__":
    main()

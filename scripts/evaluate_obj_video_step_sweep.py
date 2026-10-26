#!/usr/bin/env python3
"""CPU review of matched checkpoint10000 inference at10/8/6 solver steps.

Frozen helpers and original predictions remain read-only. Static baseline is
decoded object O_s, never a generated frame. Results are exploratory on a
previously examined test subset, not an independent validation-set selection.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import csv
import fcntl
import hashlib
import importlib.util
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HELPER = Path(__file__).with_name("evaluate_obj_video_full.py")
spec = importlib.util.spec_from_file_location("_step_sweep_cpu_helper", HELPER)
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)
shared, base = full.shared, full.base
require, digest, load_json = base.require, base.digest, base.load_json
STEPS = (10, 8, 6)
METHODS = ("static", "steps10", "steps8", "steps6")
PHASES = ("pre_onset", "mid_motion")
SCALARS = shared.SCALAR_METRICS
TIMINGS = ("model_inference_seconds", "condition_load_seconds", "export_seconds", "processing_seconds")
EPS = 1e-12
SCHEMA = "vpp2_obj.inference_step_sweep.v1"
LABELS = ("Full-scene RGB GT", "Static: repeat object O_s", "10 steps", "8 steps", "6 steps")


def read_run(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def tensor_sha(value):
    import torch
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def validate_run(state,plan,plan_sha):
    require(state["schema"]=="vpp2_obj.sampling_step_run.v1" and state["plan_sha256"]==plan_sha,"Run identity differs")
    for field in ("config_sha256","checkpoint","sigma_shift","seed"):
        require(state[field]==plan[field],f"Run differs from selection:{field}")
    require(state["manifest_sha256"]==plan["test_manifest_sha256"] and state["num_inference_steps"]==list(STEPS),"Run manifest/steps differ")
    require(state["case_count"]==24 and state["prediction_count"]==72,"Run count contract differs")
    planned={(case["id"],steps) for case in plan["cases"] for steps in STEPS}
    results={(item["id"],item["num_inference_steps"]):item for item in state["results"]}
    require(len(results)==len(state["results"]) and set(results)<=planned,"Duplicate/unexpected scored calls")
    if state["status"]=="complete":
        require(set(results)==planned and all(item["status"]=="complete" for item in results.values()),"Run claims complete with missing variants")
    if state["status"] in ("running","complete"):
        warmup=state["warmup"]
        require(len(warmup)==2 and {item["steps"] for item in warmup}=={10,6}
                and all(item["included_in_statistics"] is False for item in warmup),"Expected two unscored warm-up calls")
    return results


def prepare_plan(run, files):
    from vpp2_obj.infer import jsonable, load_config, make_dataset
    path = run / "selection.json"
    plan = load_json(path)
    plan_sha = files.sha(path)
    require(plan["status"]=="frozen" and plan["num_inference_steps"]==list(STEPS) and plan["sigma_shift"]==3,"Unexpected step-sweep plan")
    config, manifest = Path(plan["config"]).resolve(), Path(plan["test_manifest"]).resolve()
    files.sha(config, plan["config_sha256"])
    files.sha(manifest, plan["test_manifest_sha256"])
    dataset = make_dataset(load_config(config), manifest)
    require((dataset.history_frames, dataset.history_stride, dataset.prediction_frames, dataset.prediction_stride)
            == (8, 30, 16, 5), "Expected9RGB conditions and16 future frames with history/future stride30/5")
    index = {(row["task"], row["episode"]): i for i, row in enumerate(dataset.records)}
    cases = plan["cases"]
    require(len(cases) == 24 and plan.get("case_count", 24) == 24, "Expected24 planned cases")
    seen, identities, tasks = set(), set(), defaultdict(set)
    episodes_by_task = defaultdict(set)
    order_counts=defaultdict(Counter)
    for case in cases:
        require(re.fullmatch(r"[A-Za-z0-9_-]+", case["id"]) and case["id"] not in seen, "Unsafe/duplicate id")
        key = (case["task"], case["episode"], case["phase"])
        require(key not in identities and key[:2] in index and case["phase"] in PHASES, "Wrong/duplicate test case")
        require(case["phase_eligible"] is True and case.get("inference_eligible", True) is True,
                "This sweep requires eligible pre-onset and mid-motion windows")
        require(case.get("seed", 0) == 0 and case.get("anchor_index", case["anchor"]) == case["anchor"], "Seed/anchor mismatch")
        require(sorted(case["variant_order"])==sorted(STEPS),"Invalid variant order")
        order_counts[case["phase"]][tuple(case["variant_order"])]+=1
        metadata = jsonable(dataset.window_indices(index[key[:2]], case["anchor"], include_target=True))
        for field, value in metadata.items():
            if field in case:
                require(case[field] == value, f"Plan sampling differs: {case['id']} {field}")
        seen.add(case["id"])
        identities.add(key)
        tasks[case["task"]].add(case["phase"])
        episodes_by_task[case["task"]].add(case["episode"])
    require(len(tasks) == 12 and all(value == set(PHASES) for value in tasks.values())
            and all(len(value) == 1 for value in episodes_by_task.values()), "Expected12tasks,one episode each,both eligible phases")
    require(all(counts==Counter({order:2 for order in itertools.permutations(STEPS)}) for counts in order_counts.values()),"Timing order is not balanced per phase")
    return plan, plan_sha, dataset, index


def prepare_case(run, plan, plan_sha, state, case, dataset, index, files, cache_checked):
    from vpp2_obj.infer import jsonable
    import torch
    episode_index = index[(case["task"], case["episode"])]
    row = dataset.records[episode_index]
    conditioning = jsonable(dataset.window_indices(episode_index, case["anchor"], include_target=False))
    indices = jsonable(dataset.window_indices(episode_index, case["anchor"], include_target=True))
    anchor_source = indices["condition_source_indices"][-1]
    require(indices["condition_indices"][-1] == case["anchor"], "Condition tail must be current anchor")
    paths = {"rgb": str(dataset._path(row["condition_video_path"]).resolve()),
             "object": str(dataset._path(row["target_video_path"]).resolve()),
             "dedup": str(dataset._path(row["dedup_index_path"]).resolve())}
    hashes = {"rgb": files.sha(paths["rgb"], base.video_digest(row, "rgb_sha256")),
              "object": files.sha(paths["object"], base.video_digest(row, "obj_sha256")),
              "dedup": files.sha(paths["dedup"], row["dedup_index_sha256"])}
    actual_sample=dataset.get_window(episode_index,case["anchor"],include_target=False)
    require(not {"video","future_source_indices","target_video_path"}.intersection(actual_sample),"CPU input audit unexpectedly loaded targets")
    actual_input_hash={key:tensor_sha(actual_sample[key]) for key in ("condition_video","context","context_mask")}
    del actual_sample
    prepared={item["id"]:item["tensor_sha256"] for item in state["prepared_inputs"]}
    require(prepared.get(case["id"])==actual_input_hash,"Run prepared input tensors differ from actual production input")
    noise_shape=[1,16,7,dataset.video_size[0]//8,dataset.video_size[1]//8]
    noise=torch.randn(noise_shape,generator=torch.Generator(device="cpu").manual_seed(0),dtype=torch.float32).to(torch.bfloat16)
    noise_hash=tensor_sha(noise);del noise
    require(state["initial_noise_sha256"]==noise_hash and state["initial_noise_shape"]==noise_shape,"Run initial noise differs")
    items = []
    for steps in STEPS:
        directory = run / "samples" / case["id"] / f"steps_{steps}"
        trace_path, prediction = directory / "trace.json", directory / "future.mp4"
        trace = load_json(trace_path)
        expected = {"schema": "vpp2_obj.inference.v1", "status": "complete", "dry_run": False,
                    "phase": case["phase"], "phase_eligible": True, "plan_sha256": plan_sha,
                    "task": case["task"], "episode": case["episode"], "anchor_index": case["anchor"],
                    "seed": case.get("seed", 0), "num_inference_steps": steps, "sigma_shift": 3,
                    "exported_frames": 16, "decoded_frame_slice": [9, 25], "prefix_clamped": False,
                    "target_video_read": False, "target_first_frame_read": False}
        for key, value in expected.items():
            require(trace.get(key) == value, f"Invalid trace field:{case['id']}/steps{steps}/{key}")
        require(trace["input_tensor_sha256"]==actual_input_hash and trace["initial_noise_sha256"]==noise_hash
                and trace["initial_noise_shape"]==noise_shape,"Trace input/noise tensor hashes differ")
        require(trace["variant_order"]==case["variant_order"] and trace["variant_position"]==case["variant_order"].index(steps),"Variant timing order/position differs")
        require(trace["physical_gpu_ids"]==state["physical_gpu_ids"],"GPU assignment differs")
        require(trace["sampling"] == conditioning, "Production condition-only sampling differs")
        require(not {"future_source_indices", "future_timestamps_us", "target_video_path", "video"}.intersection(trace["sampling"]),
                "Target-only sampling supplied to inference")
        require(Path(trace["output_directory"]).resolve() == directory
                and Path(trace["video_path"]).resolve() == prediction, "Output paths differ")
        require(Path(trace["config"]).resolve() == Path(plan["config"]).resolve()
                and trace["config_sha256"] == plan["config_sha256"]
                and Path(trace["manifest"]).resolve() == Path(plan["test_manifest"]).resolve()
                and trace["manifest_sha256"] == plan["test_manifest_sha256"], "Config/test manifest differs")
        require(Path(trace["condition_video_path"]).resolve() == Path(paths["rgb"]), "RGB input differs")
        for field in ("prompt", "text_embedding_sha256", "text_embedding_cache_signature"):
            require(trace[field] == row[field], f"Manifest text binding differs:{field}")
        cache = dataset._path(row["text_embedding_path"]).resolve()
        require(Path(trace["text_embedding_path"]).resolve() == cache, "Cache path differs")
        files.sha(cache, trace["text_embedding_sha256"])
        key = (str(cache), tuple(trace["context_mask"]), trace["model_visible_text_tokens"])
        if key not in cache_checked:
            shared.validate_cached_text({"trace": trace})
            cache_checked.add(key)
        checkpoint = Path(trace["checkpoint"]).resolve()
        require(shared.checkpoint_step(checkpoint) == 10000 and checkpoint.is_file(), "Expected checkpoint10000")
        for origin in (plan, state):
            if "checkpoint" in origin:
                require(checkpoint == Path(origin["checkpoint"]).resolve(), "Run/selection checkpoint differs")
        stat = checkpoint.stat()
        require((stat.st_size, stat.st_mtime_ns) == (trace["checkpoint_bytes"], trace["checkpoint_mtime_ns"]), "Checkpoint identity changed")
        memory = trace["memory_strategy"]
        require(all(key in memory for key in ("method", "dit_dtype", "vae_dtype", "clip_dtype", "quantization")), "Missing memory/dtype fields")
        require(trace["condition_shape"] == [3, 9, *dataset.video_size]
                and trace["context_shape"] == [128, 4096] and Fraction(str(trace["display_fps"])) == 6, "Input/output shape contract differs")
        timing = trace["timing"]
        require(all(isinstance(timing.get(key), (int, float)) and math.isfinite(timing[key]) and timing[key] >= 0 for key in TIMINGS), "Invalid timing measurements")
        require(timing["model_inference_seconds"] > 0 and timing["processing_seconds"] > 0, "Zero measured latency")
        require(timing["processing_seconds"] + .005 >= timing["model_inference_seconds"], "Processing time excludes its model call")
        require(math.isclose(timing["processing_seconds"],sum(timing[key] for key in TIMINGS if key!="processing_seconds"),rel_tol=1e-12,abs_tol=1e-9),"Processing is not the sum of its measured parts")
        if "warmup_excluded" in trace:
            require(trace["warmup_excluded"] is True, "Measured sample must exclude warm-up")
        items.append({"steps": steps, "trace": trace, "trace_path": str(trace_path), "trace_sha256": files.sha(trace_path),
                      "prediction_path": str(prediction), "prediction_sha256": files.sha(prediction, trace["prediction_sha256"])})
    invariant_fields = [key for key in shared.TRACE_FIELDS if key != "num_inference_steps"]
    invariant_fields += list(shared.OPTIONAL_MATH_FIELDS) + ["phase", "phase_eligible", "plan_sha256", "checkpoint", "checkpoint_bytes", "checkpoint_mtime_ns","input_tensor_sha256","initial_noise_sha256","initial_noise_shape","variant_order","physical_gpu_ids"]
    for item in items[1:]:
        for key in invariant_fields:
            a, b = items[0]["trace"], item["trace"]
            require((key in a, a.get(key)) == (key in b, b.get(key)), f"Variants differ in {key}")
    replay=items[0]["trace"]["previous_10_step_replay"]
    require(replay["status"] in ("pixel_identical","different") and replay["decoded_max_abs_difference_uint8"]>=0
            and replay["decoded_mean_abs_difference_uint8"]>=0,"Malformed10-step replay result")
    files.sha(replay["previous_trace"],replay["previous_trace_sha256"])
    files.sha(replay["previous_prediction"],replay["previous_prediction_sha256"])
    previous=load_json(replay["previous_trace"])
    require(previous["status"]=="complete" and previous["prediction_sha256"]==replay["previous_prediction_sha256"],"Old10-step provenance differs")
    for key in (*shared.TRACE_FIELDS,"checkpoint","checkpoint_bytes","checkpoint_mtime_ns"):
        require(previous[key]==items[0]["trace"][key],f"Old10-step input differs:{key}")
    return {"case": case, "paths": paths, "hashes": hashes, "indices": indices, "anchor_source_index": anchor_source,"replay":replay,
            "items": items, "width": dataset.video_size[1], "height": dataset.video_size[0], "fps": Fraction(6)}


def measure(static, predictions, gt):
    import numpy as np
    require(len(predictions) == 3 and all(x.shape == gt.shape for x in (static, *predictions)), "Metric shapes differ")
    common = np.logical_or.reduce([x.max(axis=-1) > 16 for x in (gt, static, *predictions)])
    methods = {name: {**shared.metric_one(array, gt, common),
                      "per_future_frame": [shared.metric_one(array[i], gt[i], common[i]) for i in range(16)]}
               for name, array in zip(METHODS, (static, *predictions))}
    frozen, baseline = methods["static"]["common_foreground_mae"], methods["steps10"]["common_foreground_mae"]
    for name in METHODS[1:]:
        current = methods[name]
        gain, degradation = frozen-current["common_foreground_mae"], current["common_foreground_mae"]-baseline
        current.update(mae_improvement_vs_static=gain, relative_mae_improvement_vs_static=gain/frozen if frozen > EPS else None,
                       mae_beats_static=gain > EPS, mae_ties_static=abs(gain) <= EPS,
                       mae_degradation_vs_steps10=degradation, relative_mae_degradation_vs_steps10=degradation/baseline if baseline > EPS else None,
                       iou_degradation_vs_steps10=methods["steps10"]["silhouette_iou"]-current["silhouette_iou"])
    return {"common_foreground_pixels": int(common.sum()), "common_foreground_pixels_per_frame": common.sum(axis=(1,2)).tolist(), "methods": methods}


def render(directory, inputs, rgb, static, predictions, gt):
    import av
    import numpy as np
    from PIL import Image, ImageDraw
    w, h, fps = inputs["width"], inputs["height"], inputs["fps"]
    case = inputs["case"]
    arrays = [rgb, static, *predictions]
    def time_label(i):
        t = inputs["indices"]["future_timestamps_us"][i]
        start = inputs["indices"]["condition_timestamps_us"][-1]
        return f"F{i+1}/16 | sim {t/1e6:.3f}s ({(t-start)/1e6:+.3f}s)"
    with av.open(str(directory/"comparison.mp4"), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 5*w, h+76, "yuv420p"
        stream.options = {"crf": "18", "preset": "fast", "threads": "1"}
        for i in range(16):
            canvas = Image.new("RGB", (5*w, h+76), "#202936")
            draw = ImageDraw.Draw(canvas)
            for col, (label, frames) in enumerate(zip(LABELS, arrays)):
                draw.text((col*w+8, 5), label, font=base.font(18), fill="white")
                canvas.paste(Image.fromarray(frames[i]), (col*w, 76))
            draw.text((10,31), f"{case['task']} / {case['episode']} / {case['phase']} | anchor {case['anchor']} | {time_label(i)}", font=base.font(17), fill="white")
            draw.text((10,54), "Checkpoint10000 | seed0 | sigma shift3 | observer left/ego right in every panel | 6fps display", font=base.font(14), fill="#d5dfef")
            frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")
            frame.pts, frame.time_base = i, 1/fps
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    for object_gt, name in ((False,"contact.png"),(True,"object_contact.png")):
        canvas = Image.new("RGB", (5*w,76+4*(h+32)), "#202936")
        draw = ImageDraw.Draw(canvas)
        draw.text((10,6), f"{case['task']} / {case['episode']} / {case['phase']} | anchor {case['anchor']} | checkpoint10000", font=base.font(22), fill="white")
        labels = ("Object-only GT",*LABELS[1:]) if object_gt else LABELS
        for col,label in enumerate(labels):
            draw.text((col*w+8,43),label,font=base.font(18),fill="white")
        for row,i in enumerate((0,4,9,15)):
            y=76+row*(h+32)
            draw.text((10,y+4),time_label(i),font=base.font(17),fill="white")
            for col,frames in enumerate([gt if object_gt else rgb,static,*predictions]):
                canvas.paste(Image.fromarray(frames[i]),(col*w,y+32))
        canvas.save(directory/name)
    _,decoded=base.read_prediction(directory/"comparison.mp4",5*w,h+76,fps)
    return decoded


def process_case(output, inputs, implementation_sha, plan_sha):
    import numpy as np
    case=inputs["case"]
    fingerprint={"implementation_sha256":implementation_sha,"plan_sha256":plan_sha,"case":case,
                 "paths":inputs["paths"],"source_sha256":inputs["hashes"],"indices":inputs["indices"],
                 "anchor_source_index":inputs["anchor_source_index"],
                 "previous_10_step_replay":inputs["replay"],
                 "predictions":[{key:item[key] for key in ("steps","trace_path","trace_sha256","prediction_path","prediction_sha256")} for item in inputs["items"]]}
    signature=full.canonical_hash(fingerprint)
    target=output/"samples"/case["id"]
    if target.exists():
        verification=load_json(target/"verification.json")
        require(verification.get("status")=="passed" and verification.get("input_signature")==signature,"Cached case input changed; use a new output")
        for name,artifact in verification["artifacts"].items():
            require(digest(target/name)==artifact["sha256"],"Cached artifact changed")
        metrics=load_json(target/"metrics.json")
        require(metrics["input_signature"]==signature,"Cached metric signature changed")
        return metrics,True
    staging=Path(tempfile.mkdtemp(prefix=f".{case['id']}-",dir=output/"samples"))
    try:
        w,h=inputs["width"],inputs["height"]
        future=inputs["indices"]["future_source_indices"]
        rgb=base.read_source(inputs["paths"]["rgb"],future,w,h)
        objects=base.read_source(inputs["paths"]["object"],[inputs["anchor_source_index"],*future],w,h)
        gt=objects[1:];static=np.broadcast_to(objects[0],gt.shape)
        predictions,decodes=[],[]
        for item in inputs["items"]:
            video,decode=base.read_prediction(item["prediction_path"],w,h,inputs["fps"])
            predictions.append(video);decodes.append(decode)
        previous,previous_decode=base.read_prediction(inputs["replay"]["previous_prediction"],w,h,inputs["fps"])
        difference=np.abs(previous.astype(np.int16)-predictions[0].astype(np.int16))
        replay_status="pixel_identical" if np.array_equal(previous,predictions[0]) else "different"
        replay_result={"status":replay_status,"previous_decode":previous_decode,"max_absolute_pixel_difference":int(difference.max()),
                       "mean_absolute_pixel_difference":float(difference.mean()),
                       "scope":"Only this fixed case's sixteen decoded MP4 frames are compared;comparison metrics use this experiment's new10-step output even if replay differs."}
        require(inputs["replay"]["status"]==replay_status
                and inputs["replay"]["decoded_max_abs_difference_uint8"]==replay_result["max_absolute_pixel_difference"]
                and math.isclose(inputs["replay"]["decoded_mean_abs_difference_uint8"],replay_result["mean_absolute_pixel_difference"],abs_tol=1e-12),"Independent replay does not match runner's reported result")
        metrics={"schema":SCHEMA+".case_metrics","status":"passed","input_signature":signature,
                 **{key:case[key] for key in ("id","task","episode","phase","phase_eligible","anchor")},
                 **measure(static,predictions,gt),
                 "previous_10_step_replay":replay_result,
                 "timing":{f"steps{item['steps']}":{key:item["trace"]["timing"][key] for key in TIMINGS} for item in inputs["items"]}}
        base.write_json(staging/"metrics.json",metrics)
        comparison_decode=render(staging,inputs,rgb,static,predictions,gt)
        verification={"schema":SCHEMA+".case_verification","status":"passed","input_signature":signature,"input":fingerprint,
                      "matched_input":{key:inputs["items"][0]["trace"][key] for key in shared.TRACE_FIELDS if key!="num_inference_steps"},
                      "sigma_shift":3,"checkpoint_identity":{key:inputs["items"][0]["trace"][key] for key in ("checkpoint","checkpoint_bytes","checkpoint_mtime_ns")},
                      "static_anchor_raw_rgb_sha256":hashlib.sha256(objects[0].tobytes()).hexdigest(),
                      "static_source":"Original object video at current condition_source_indices[-1], repeated16; not first prediction frame.",
                      "prediction_decode":decodes,"comparison_decode":comparison_decode,"target_supplied_to_model":False,"gpu_used":False,
                      "independent_10_step_replay":replay_result,
                      "artifacts":{name:{"sha256":digest(staging/name),"bytes":(staging/name).stat().st_size} for name in ("metrics.json","comparison.mp4","contact.png","object_contact.png")}}
        base.write_json(staging/"verification.json",verification)
        require(not target.exists(),"Output appeared during processing")
        os.rename(staging,target)
        return metrics,False
    finally:
        if staging.exists():shutil.rmtree(staging)


def distribution(values):
    import numpy as np
    return {"n":len(values),"mean":statistics.fmean(values),"median":float(np.median(values)),"p95":float(np.percentile(values,95))} if values else {"n":0,"mean":None,"median":None,"p95":None}


def aggregate_group(cases,expected_tasks):
    groups=defaultdict(lambda:defaultdict(list))
    for case in cases:groups[case["task"]][case["episode"]].append(case)
    per_task={}
    for task,episodes in sorted(groups.items()):
        methods={}
        for method in METHODS:
            methods[method]={key:statistics.fmean(statistics.fmean(case["methods"][method][key] for case in rows) for rows in episodes.values()) for key in SCALARS}
            if method!="static":
                for dest,src in (("mae_win_rate_vs_static","mae_beats_static"),("mae_tie_rate_vs_static","mae_ties_static")):
                    methods[method][dest]=statistics.fmean(statistics.fmean(float(case["methods"][method][src]) for case in rows) for rows in episodes.values())
        per_task[task]={"case_count":sum(map(len,episodes.values())),"episode_count":len(episodes),"methods":methods}
    macro=None
    if per_task:
        macro={method:{key:statistics.fmean(task["methods"][method][key] for task in per_task.values()) for key in next(iter(per_task.values()))["methods"][method]} for method in METHODS}
        static,ten=macro["static"]["common_foreground_mae"],macro["steps10"]["common_foreground_mae"]
        for method in METHODS[1:]:
            gain=static-macro[method]["common_foreground_mae"];degradation=macro[method]["common_foreground_mae"]-ten
            macro[method].update(mae_improvement_vs_static=gain,relative_mae_improvement_vs_static=gain/static if static>EPS else None,
                                 mae_degradation_vs_steps10=degradation,relative_mae_degradation_vs_steps10=degradation/ten if ten>EPS else None,
                                 iou_degradation_vs_steps10=macro["steps10"]["silhouette_iou"]-macro[method]["silhouette_iou"])
    times={method:{key:distribution([case["timing"][method][key] for case in cases]) for key in TIMINGS} for method in METHODS[1:]}
    for method in METHODS[1:]:
        for key in TIMINGS:
            current,ten=times[method][key]["mean"],times["steps10"][key]["mean"]
            times[method][key]["speedup_vs_steps10_ratio_of_means"]=ten/current if current is not None and current>EPS else None
            times[method][key]["relative_latency_reduction_vs_steps10"]=1-current/ten if current is not None and ten>EPS else None
    return {"case_count":len(cases),"covered_task_count":len(per_task),"expected_task_count":len(expected_tasks),
            "missing_tasks":sorted(set(expected_tasks)-set(per_task)),"task_macro":macro,"per_task":per_task,"timing":times}


def aggregate(cases,plan):
    tasks=sorted({case["task"] for case in plan["cases"]})
    return {"overall":aggregate_group(cases,tasks),**{phase:aggregate_group([case for case in cases if case["phase"]==phase],tasks) for phase in PHASES}}


def definitions():
    return {"metric_target":"object_only_gt","foreground_threshold_uint8":16,
            "common_mask_rule":"Per pixel/frame maxRGB>16 union of GT|static_O_s|steps10|steps8|steps6. All four methods use the same RGB MAE denominator.",
            "mae_rule":"Pool16frames absolute RGB uint8 error on common mask / (common pixels*3*255);0..1,lower better.",
            "iou_rule":"Each method uses conventional foreground intersection/union with objectGT,pooled16frames;higher better.",
            "static_rule":"Decode actual raw object frame at current anchor source index,repeat16. Never use first predicted frame.",
            "aggregation_rule":"Within each phase: episode-equal within task,then task-equal. Overall first averages two phase records within episode,then episodes within task,then tasks. Partial reports have explicit missing tasks and are not final results.",
            "static_win_rule":"Per-case staticMAE-modelMAE>1e-12 is win;absolute delta<=1e-12 is tie. Report episode/task-balanced rate,not action success.",
            "steps10_degradation_rule":"MAE_model-MAE_steps10;positive=worse. Relative value divides aggregate delta by aggregate10-step MAE (null if<=1e-12). IoU degradation=IoU_steps10-IoU_model;positive=worse. Same formulas also stored per case.",
            "timing_rule":"Separate recorded model_inference_seconds,condition_load_seconds,export_seconds,processing_seconds. Means/median/p95 pool matched cases in each phase/overall;at full coverage each task contributes equally. p95 uses NumPy linear percentile. Speedup is mean10/meanN,not a mean of pairwise speedup ratios. Model timing covers synchronized infer_object_future including condition VAE/CLIP and output VAE,not DiT-only time. Processing is condition+model+export;input reads follow an unmeasured pre-read and reflect warm inputs,not cold-storage latency. Model load,two separate warmups,input hashing,trace writes,GC,and old-output replay checks are excluded.",
            "background_rule":"GT-background predicted-foreground rate and mean predicted intensity include displaced task-object errors;they are not pure background retention metrics.",
            "empty_conventions":"Empty common mask MAE0;empty pair union IoU1;zeroGTbackground rate/intensity0;near-zero reference relative ratios null.",
            "cross_report_warning":"The shared mask now includes static and all three step-count predictions. Do not directly compare these MAEs to earlier full10-step or three-checkpoint reports with different common masks.",
            "scope":"Exploratory comparison on24 held-out-test windows selected from12tasks using ground-truth phase eligibility,not predictions. This is a previously examined test subset,not independent validation data or a representative estimate of all50tasks;no task success rate is claimed."}


def write_csv(cases,aggregates):
    output=io.StringIO(newline="")
    fields=["row_type","group","task","episode","id","phase","anchor","method",*SCALARS,"mae_win_rate_vs_static",
            "mae_degradation_vs_steps10","relative_mae_degradation_vs_steps10","iou_degradation_vs_steps10",*TIMINGS,
            "timing_field","timing_n","timing_mean","timing_median","timing_p95","speedup_vs_steps10"]
    writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader()
    for case in cases:
        for method in METHODS:
            row={"row_type":"case",**{key:case[key] for key in ("task","episode","id","phase","anchor")},"method":method,
                 **{key:value for key,value in case["methods"][method].items() if key in fields}}
            if method!="static":row.update(case["timing"][method])
            writer.writerow(row)
    for group,stats in aggregates.items():
        for method,values in (stats["task_macro"] or {}).items():
            row={"row_type":"task_macro","group":group,"method":method,**{key:value for key,value in values.items() if key in fields}}
            if method!="static":row.update({key:stats["timing"][method][key]["mean"] for key in TIMINGS})
            writer.writerow(row)
        for method,timings in stats["timing"].items():
            for field,values in timings.items():
                writer.writerow({"row_type":"timing_distribution","group":group,"method":method,"timing_field":field,
                                 "timing_n":values["n"],"timing_mean":values["mean"],"timing_median":values["median"],"timing_p95":values["p95"],
                                 "speedup_vs_steps10":values["speedup_vs_steps10_ratio_of_means"]})
    return output.getvalue()


def page(cases,plan,status,aggregates):
    import html
    tables=[]
    for group,stats in aggregates.items():
        rows=[]
        for method,score in (stats["task_macro"] or {}).items():
            delta=score.get("relative_mae_degradation_vs_steps10");wins=score.get("mae_win_rate_vs_static")
            timing=stats["timing"].get(method)
            time_text="—" if timing is None else f"{timing['model_inference_seconds']['mean']:.3f} / {timing['processing_seconds']['mean']:.3f}"
            rows.append(f"<tr><td>{method}</td><td>{score['common_foreground_mae']:.6f}</td><td>{score['silhouette_iou']:.6f}</td><td>{'—' if wins is None else f'{wins:.1%}'}</td><td>{'—' if delta is None else f'{delta:+.2%}'}</td><td>{time_text}</td></tr>")
        tables.append(f"<section><h2>{group}：{stats['case_count']} cases / {stats['covered_task_count']}/12 tasks</h2><table><tr><th>方法</th><th>共同前景MAE↓</th><th>IoU↑</th><th>MAE胜过静止比例</th><th>相对10步MAE变化↑为退化</th><th>模型/完整处理均值秒↓</th></tr>{''.join(rows)}</table></section>")
    timing_rows=[]
    for method,stats in aggregates["overall"]["timing"].items():
        for label,key in (("模型调用","model_inference_seconds"),("完整处理","processing_seconds")):
            d=stats[key]
            if d["n"]:
                timing_rows.append(f"<tr><td>{method}</td><td>{label}</td><td>{d['n']}</td><td>{d['mean']:.3f}</td><td>{d['median']:.3f}</td><td>{d['p95']:.3f}</td><td>{d['speedup_vs_steps10_ratio_of_means']:.3f}×</td></tr>")
    cards=[]
    for case in cases:
        folder=f"samples/{case['id']}"
        cards.append(f'''<article data-task="{html.escape(case['task'])}" data-phase="{case['phase']}"><h3>{html.escape(case['task'])} / {html.escape(case['episode'])} / {case['phase']} / s={case['anchor']}</h3>
<video controls loop playsinline preload="none" src="{folder}/comparison.mp4"></video><p>五列：全景RGB GT | 静止物体O_s | 10步 | 8步 | 6步；每格observer左、ego右。</p>
<details><summary>物体GT与四种方法：未来1/5/10/16帧</summary><a href="{folder}/object_contact.png"><img loading="lazy" src="{folder}/object_contact.png"></a></details>
<p><a href="{folder}/contact.png">全景GT四时刻图</a> · <a href="{folder}/metrics.json">逐帧指标</a> · <a href="{folder}/verification.json">来源校验</a></p></article>''')
    options=''.join(f'<option>{html.escape(task)}</option>' for task in sorted({case['task'] for case in plan['cases']}))
    identical=sum(case['previous_10_step_replay']['status']=='pixel_identical' for case in cases)
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Checkpoint10000:10/8/6步对照</title>
<style>body{{margin:24px auto;padding:0 18px;max-width:2240px;background:#151b25;color:#eef3fc;font:16px/1.6 system-ui}}a{{color:#a9ceff}}article,section{{margin:24px 0;padding:18px;background:#253144}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #526174;padding:8px;text-align:left}}video,img{{display:block;width:100%;height:auto}}select{{padding:8px;margin:6px}}[hidden]{{display:none}}</style>
<h1>Checkpoint10000：采样10/8/6步与静止基线</h1><p>状态：{status}；完成{len(cases)}/24个窗口。{'当前为不完整结果，不能作最终比较。' if status!='passed' else '12任务各一episode，合格运动启动前/运动中各一窗。'}</p>
<p>固定相同权重、输入、seed0、sigma_shift3和内存/dtype策略，仅采样步数变化。全景RGB GT只展示，指标对物体GT计算。静止基线是原始物体O_s重复16次；GT不输入模型。</p>
<p>这是已经查看过的test子集上的探索实验，不是独立验证集，也不代表50任务总体。共同mask为GT|static|10|8|6，不能与此前不同mask报告直接比较MAE。MAE胜率不是动作成功率。模型时间包括条件编码/DiT/VAE解码；处理计时为预读之后的条件读取+模型+导出，不含模型加载、2次warm-up、校验哈希/trace写入/GC/旧输出重放。</p>
<p>旧full10重放独立检查：{identical}/{len(cases)}个已处理窗口的全部16帧解码像素相同；{len(cases)-identical}个有差异。比较指标使用本次新生成10步输出作为配对基线，报告passed仅表示来源/计算/覆盖核验通过，不等同于所有旧输出都逐像素重现。</p>
<p><a href="metrics.json">全部指标/定义JSON</a> · <a href="metrics.csv">CSV</a> · <a href="summary.json">审计摘要</a></p>{''.join(tables)}
<h2>延迟：均值 / 中位数 / p95</h2><table><tr><th>步数</th><th>范围</th><th>n</th><th>均值秒</th><th>中位秒</th><th>p95秒</th><th>相对10步加速</th></tr>{''.join(timing_rows)}</table>
<h2>逐例对照</h2><select id="task"><option value="">全部任务</option>{options}</select><select id="phase"><option value="">两个阶段</option><option>pre_onset</option><option>mid_motion</option></select>{''.join(cards)}
<script>function filter(){{document.querySelectorAll('article[data-task]').forEach(e=>{{e.hidden=['task','phase'].some(k=>document.getElementById(k).value&&e.dataset[k]!==document.getElementById(k).value)}})}}['task','phase'].forEach(k=>document.getElementById(k).addEventListener('change',filter));</script></html>'''


def evaluate(run,output,allow_partial):
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/".evaluate.lock").open("a")
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise ValueError("Another evaluator owns output")
    files=full.Files()
    plan,plan_sha,dataset,index=prepare_plan(run,files)
    state,state_sha=read_run(run/"run.json")
    validate_run(state,plan,plan_sha)
    if not allow_partial:require(state["status"]=="complete","Run incomplete;use --allow-partial")
    implementation={str(path.resolve()):digest(path) for path in (Path(__file__),HELPER,full.HELPER,shared.HELPER)}
    implementation_sha=full.canonical_hash(implementation)
    contract={"schema":SCHEMA+".contract","run":str(run),"plan_sha256":plan_sha,"implementation":implementation,"definitions":definitions()}
    if (output/"evaluation_contract.json").exists():require(load_json(output/"evaluation_contract.json")==contract,"Evaluation contract changed;use new output")
    else:full.atomic_json(output/"evaluation_contract.json",contract)
    (output/"samples").mkdir(exist_ok=True)
    completed,pending,reused,cache_checked=[],[],0,set()
    execution=None
    for case in plan["cases"]:
        available=[]
        for steps in STEPS:
            directory=run/"samples"/case["id"]/f"steps_{steps}"
            available.append((directory/"trace.json").is_file() and (directory/"future.mp4").is_file()
                             and load_json(directory/"trace.json").get("status")=="complete")
        if not all(available):pending.append({"id":case["id"],"complete":dict(zip(map(str,STEPS),available))});continue
        # Refresh growing run metadata so a newly completed case has its
        # prepared-input binding even when it finished after our first snapshot.
        state,state_sha=read_run(run/"run.json")
        results=validate_run(state,plan,plan_sha)
        if not all((case["id"],steps) in results for steps in STEPS):
            pending.append({"id":case["id"],"reason":"run_result_publication_pending"});continue
        inputs=prepare_case(run,plan,plan_sha,state,case,dataset,index,files,cache_checked)
        for item in inputs["items"]:
            result=results[(case["id"],item["steps"])]
            require(Path(result["trace"]).resolve()==Path(item["trace_path"]) and result["timing"]==item["trace"]["timing"],"Run result trace/timing differs")
        trace=inputs["items"][0]["trace"]
        current={key:trace[key] for key in ("checkpoint","checkpoint_bytes","checkpoint_mtime_ns","memory_strategy","sigma_shift","seed","config_sha256","manifest_sha256")}
        if execution is None:execution=current
        require(current==execution,"Execution settings changed across cases")
        metrics,used=process_case(output,inputs,implementation_sha,plan_sha)
        completed.append(metrics);reused+=used
        full.atomic_json(output/"progress.json",{"status":"partial","completed_case_count":len(completed),"planned_case_count":24,"last_case":case["id"],"reused_case_count":reused})
        print(f"{'Reused' if used else 'Evaluated'} {len(completed)}/24:{case['id']}",flush=True)
    state,state_sha=read_run(run/"run.json")
    validate_run(state,plan,plan_sha)
    final=len(completed)==24 and state["status"]=="complete"
    require(allow_partial or final,"Missing matching predictions")
    status="passed" if final else "partial"
    aggregates=aggregate(completed,plan)
    compact=[{**{key:value for key,value in case.items() if key!="methods"},"methods":{name:{key:value for key,value in scores.items() if key!="per_future_frame"} for name,scores in case["methods"].items()}} for case in completed]
    document={"schema":SCHEMA+".metrics","status":status,"case_count":len(completed),"planned_case_count":24,**definitions(),"aggregates":aggregates,"cases":compact}
    webpage=page(completed,plan,status,aggregates)
    links=full.check_links(webpage,output)
    for path,expected in implementation.items():require(digest(path)==expected,"Implementation changed while evaluating")
    for path,(_,expected) in list(files.entries.items()):files.sha(path,expected)
    contents={"metrics.json":full.json_text(document),"metrics.csv":write_csv(completed,aggregates),"index.html":webpage}
    for name,content in contents.items():full.atomic_text(output/name,content)
    summary={"schema":SCHEMA,"status":status,"created_at_utc":datetime.now(timezone.utc).isoformat(),"case_count":len(completed),"planned_case_count":24,
             "run":str(run),"run_json_sha256":state_sha,"run_metadata":state,"plan_sha256":plan_sha,"pending":pending,"reused_case_count":reused,
             "execution":execution,"aggregates":aggregates,"scope":definitions()["scope"],"input_content_sha256":{path:value for path,(_,value) in files.entries.items()},
             "independent_previous_10_step_replay":{"checked_case_count":len(completed),
                 "pixel_identical_case_count":sum(case["previous_10_step_replay"]["status"]=="pixel_identical" for case in completed),
                 "all_processed_cases_pixel_identical":all(case["previous_10_step_replay"]["status"]=="pixel_identical" for case in completed),
                 "max_absolute_pixel_difference":max((case["previous_10_step_replay"]["max_absolute_pixel_difference"] for case in completed),default=None),
                 "different_case_ids":[case["id"] for case in completed if case["previous_10_step_replay"]["status"]!="pixel_identical"],
                 "scope":"Each processed case compared all16 decoded MP4frames;not a claim beyond these windows. passed summary status does not imply replay pixel identity."},
             "case_verifications":[{"id":case["id"],"path":f"samples/{case['id']}/verification.json","sha256":digest(output/"samples"/case["id"]/"verification.json")} for case in completed],
             "implementation":implementation,"html_local_links_checked":links,"gpu_used":False,"model_loaded":False,"original_predictions_modified":False,
             "artifacts":{name:{"sha256":digest(output/name),"bytes":(output/name).stat().st_size} for name in contents}}
    full.atomic_json(output/"summary.json",summary)
    full.atomic_json(output/"progress.json",{"status":status,"completed_case_count":len(completed),"planned_case_count":24,"reused_case_count":reused})
    print(full.json_text({"status":status,"case_count":len(completed),"output":str(output),"overall":aggregates["overall"]["task_macro"],"timing":aggregates["overall"]["timing"]}))


def self_test():
    import numpy as np
    gt=np.zeros((16,8,8,3),dtype=np.uint8);gt[:,1:3,1:3]=255
    static=np.zeros_like(gt);extra=gt.copy();extra[:,5:7,5:7]=255
    score=measure(static,[gt,extra,static],gt)
    require(score["methods"]["static"]["common_foreground_mae"]==.5 and score["methods"]["steps10"]["common_foreground_mae"]==0,"Common mask normalization failed")
    require(score["methods"]["steps8"]["common_foreground_mae"]==.5 and score["methods"]["steps8"]["silhouette_iou"]==.5,"MAE or pairIoU failed")
    require(score["methods"]["steps8"]["relative_mae_degradation_vs_steps10"] is None,"Zero10-step denominator must be null")
    require(score["methods"]["steps10"]["mae_beats_static"] is True and score["methods"]["steps8"]["mae_ties_static"] is True,"Win/tie failed")
    cases=[]
    for task,count,value in (("A",1,.2),("B",3,.8)):
        for i in range(count):
            methods=json.loads(json.dumps(score["methods"]))
            for method in METHODS:methods[method]["common_foreground_mae"]=value
            cases.append({"task":task,"episode":str(i),"methods":methods,"timing":{f"steps{n}":{key:float(n) for key in TIMINGS} for n in STEPS}})
    agg=aggregate_group(cases,["A","B","missing"])
    require(math.isclose(agg["task_macro"]["steps10"]["common_foreground_mae"],.5),"Task weighting failed")
    require(agg["missing_tasks"]==["missing"],"Missing task hidden")
    require(agg["timing"]["steps8"]["model_inference_seconds"]["speedup_vs_steps10_ratio_of_means"]==1.25,"Timing speedup wrong")
    require(distribution([1.,2.,3.,4.])["p95"]==3.8499999999999996,"Percentile convention changed")
    print(full.json_text({"status":"self_test_passed","checks":["five-way shared mask","pairIoU","wins/ties","zero baseline ratios","task-balanced aggregation","missing task coverage","timing mean/median/p95/speedup"],"gpu_used":False}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path);parser.add_argument("--output",type=Path)
    parser.add_argument("--allow-partial",action="store_true");parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    import torch
    torch.set_num_threads(1)
    if args.self_test:self_test();return
    require(args.run and args.output,"Provide --run --output")
    evaluate(args.run.resolve(),args.output.resolve(),args.allow_partial)


if __name__=="__main__":main()

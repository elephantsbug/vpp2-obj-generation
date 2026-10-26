#!/usr/bin/env python3
"""CPU full-clean comparison of paired6/4 solver steps and real static O_s.

Read-only sources; resumable per-case artifact checks; processes at most --workers
cases simultaneously. Phase-eligible/fallback and prior24/new456 are separate.
Overall quality deduplicates physical windows; timing retains all actual calls.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from fractions import Fraction
import csv
import fcntl
import hashlib
import html
import importlib.util
import io
import math
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import statistics
import tempfile
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
HELPER=Path(__file__).with_name("evaluate_obj_video_step_sweep.py")
spec=importlib.util.spec_from_file_location("_full_steps_cpu",HELPER)
previous=importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
full,shared,base=previous.full,previous.shared,previous.base
require,digest,load_json=base.require,base.digest,base.load_json
tensor_sha=previous.tensor_sha
STEPS=(6,4)
METHODS=("static","steps6","steps4")
PHASES,SCALARS,TIMINGS=previous.PHASES,previous.SCALARS,previous.TIMINGS
EPS=1e-12
SCHEMA="vpp2_obj.full_inference_step_sweep.v1"
IDENTITY_FIELDS=("config","config_sha256","manifest","manifest_sha256","checkpoint","checkpoint_bytes",
                 "checkpoint_mtime_ns","plan_sha256","runner_sha256","runner_helper_sha256","resident_loader_sha256",
                 "seed","sigma_shift","memory_strategy","prior_24_plan_sha256","prior_24_run_sha256")


def prepare_plan(run,files):
    from vpp2_obj.infer import jsonable,load_config,make_dataset
    plan=load_json(run/"selection.json"); sha=files.sha(run/"selection.json")
    require(plan["schema"]=="vpp2_obj.full_sampling_step_selection.v1" and plan["status"]=="frozen", "Wrong frozen selection")
    require(plan["num_inference_steps"]==list(STEPS) and plan["sigma_shift"]==3 and plan["seed"]==0 and plan["num_shards"]==4,"Wrong sampling contract")
    for key in ("config","test_manifest","parent_plan","prior_24_plan"):
        files.sha(plan[key],plan[key+"_sha256"])
    files.sha(Path(plan["prior_24_run"])/"run.json",plan["prior_24_run_sha256"])
    prior=load_json(plan["prior_24_plan"])
    prior_state=load_json(Path(plan["prior_24_run"])/"run.json")
    require(prior_state["status"]=="complete" and prior_state["plan_sha256"]==plan["prior_24_plan_sha256"],"Prior run/plan differs")
    strip=lambda c:{k:v for k,v in c.items() if k not in ("steps","sampling_steps","variant_order","shard_index")}
    require([strip(c) for c in plan["cases"]]==[strip(c) for c in load_json(plan["parent_plan"])["cases"]],"Original full selection changed")
    checkpoint=Path(plan["checkpoint"]).resolve(); stat=checkpoint.stat()
    require(shared.checkpoint_step(checkpoint)==10000 and (stat.st_size,stat.st_mtime_ns)==(plan["checkpoint_stat"]["size_bytes"],plan["checkpoint_stat"]["mtime_ns"]),"Checkpoint stat changed")
    dataset=make_dataset(load_config(plan["config"]),Path(plan["test_manifest"]))
    require((dataset.history_frames,dataset.history_stride,dataset.prediction_frames,dataset.prediction_stride)==(8,30,16,5),"9RGB/16object layout differs")
    index={(r["task"],r["episode"]):i for i,r in enumerate(dataset.records)}
    require(len(index)==240 and len(plan["cases"])==plan["case_count"]==480,"Full test count differs")
    identities=set(); ids=set(); orders=defaultdict(Counter); counts=Counter(); episode_phases=defaultdict(set)
    for c in plan["cases"]:
        require(re.fullmatch(r"[A-Za-z0-9_-]+",c["id"]) and c["id"] not in ids,"Unsafe/duplicate id")
        key=(c["task"],c["episode"],c["phase"])
        require(key not in identities and key[:2] in index and c["phase"] in PHASES,"Duplicate/unknown case")
        require(c["inference_eligible"] is True and type(c["phase_eligible"]) is bool and (c["fallback_reason"] is None)==c["phase_eligible"],"Invalid phase/fallback")
        require(c["seed"]==0 and c["anchor"]==c["anchor_index"] and c["episode_index"]==index[key[:2]],"Episode/anchor differs")
        require(c["shard_index"]==index[key[:2]]%4 and c["sampling_steps"]==list(STEPS) and sorted(c["variant_order"])==[4,6],"Shard/order differs")
        metadata=jsonable(dataset.window_indices(index[key[:2]],c["anchor"],include_target=True))
        for field,value in metadata.items():
            if field in c: require(c[field]==value,f"Production plan sampling differs:{c['id']}/{field}")
        require(len(metadata["future_indices"])==16 and c["anchor"]+80<dataset.episode_length(index[key[:2]]),"Incomplete future")
        ids.add(c["id"]);identities.add(key);episode_phases[key[:2]].add(c["phase"])
        orders[(c["shard_index"],c["phase"])][tuple(c["variant_order"])]+=1
        counts[(c["phase"],c["phase_eligible"])]+=1
    require(set(episode_phases)==set(index) and all(p==set(PHASES) for p in episode_phases.values()),"Missing test episode/phase")
    require(len({c["task"] for c in plan["cases"]})==50 and full.temporal_coverage(plan["cases"])["unique_temporal_windows"]==464,"Task/window coverage differs")
    require(counts==Counter({("pre_onset",True):215,("pre_onset",False):25,("mid_motion",True):210,("mid_motion",False):30}),"Phase eligibility changed")
    require(len(orders)==8 and all(v==Counter({(6,4):30,(4,6):30}) for v in orders.values()),"Unbalanced variant order by shard/phase")
    prior_ids={c["id"] for c in prior["cases"]}
    require(len(prior_ids)==24 and prior_ids<=ids,"Prior24 overlap differs")
    for old in prior["cases"]:
        new=next(c for c in plan["cases"] if c["id"]==old["id"])
        require(strip(old)==strip(new),"Prior24 physical window changed")
    return plan,sha,dataset,index,prior_ids


def snapshot(run,plan,sha):
    # Shards may advance after the aggregate manifest was published. A changed
    # running shard is accepted as a newer atomic snapshot, with both SHAs logged;
    # immutable complete shards must exactly match the aggregate's declared SHA.
    state,state_sha=previous.read_run(run/"run.json")
    require(state["schema"]=="vpp2_obj.full_sampling_step_run.v1" and state["plan_sha256"]==sha,"Run identity differs")
    require(state["status"] in ("preparing","running","complete"),"Run failed/invalid")
    require(state["case_count"]==480 and state["prediction_count"]==960 and state["num_shards"]==4 and state["num_inference_steps"]==list(STEPS),"Run counts differ")
    for field in ("config","config_sha256","checkpoint","sigma_shift","seed","prior_24_plan_sha256","prior_24_run_sha256"):
        require(state[field]==plan[field],f"Run/plan identity differs:{field}")
    require(state["manifest"]==plan["test_manifest"] and state["manifest_sha256"]==plan["test_manifest_sha256"],"Run test manifest differs")
    gpus=state["physical_gpu_ids"].split(",")
    require(len(gpus)==len(set(gpus))==4,"Expected four GPU shards")
    aggregate={(r["id"],r["num_inference_steps"]):r for r in state["results"]}
    require(len(aggregate)==len(state["results"]),"Duplicate aggregate result")
    wanted={(c["id"],n) for c in plan["cases"] for n in STEPS}
    require(set(aggregate)<=wanted,"Unexpected aggregate result")
    shards={}; hashes={}; all_results={}; descriptions={d["shard_index"]:d for d in state["shards"]}
    require(len(descriptions)==len(state["shards"]) and set(descriptions)<=set(range(4)),"Invalid shard list")
    for s,desc in descriptions.items():
        path=run/"shards"/f"shard_{s:03d}.json"
        require(Path(desc["path"]).resolve()==path and desc["physical_gpu_ids"]==gpus[s],"Shard path/GPU differs")
        shard,current_sha=previous.read_run(path)
        if desc["status"]=="complete": require(desc["sha256"]==current_sha,"Complete shard hash changed")
        require(shard["schema"]=="vpp2_obj.full_sampling_step_shard.v1" and shard["status"] in ("loading_model","running","complete"),"Invalid/failed shard")
        for field in IDENTITY_FIELDS: require(shard[field]==state[field],f"Shard/run differs:{field}")
        require(shard["shard_index"]==s and shard["num_shards"]==4 and shard["physical_gpu_ids"]==gpus[s]
                and shard["case_count"]==120 and shard["prediction_count"]==240 and shard["num_inference_steps"]==list(STEPS),"Shard contract differs")
        rows={(r["id"],r["num_inference_steps"]):r for r in shard["results"]}
        expected={(c["id"],n) for c in plan["cases"] if c["shard_index"]==s for n in STEPS}
        require(len(rows)==len(shard["results"]) and set(rows)<=expected and all(r["status"]=="complete" for r in rows.values()),"Unexpected shard result")
        if shard["status"] in ("running","complete"):
            warm=shard["warmup"]
            require(len(warm)==2 and {w["steps"] for w in warm}==set(STEPS) and all(w["included_in_statistics"] is False for w in warm),"Warmups must be excluded6/4")
        prepared=shard["prepared_inputs"]
        require(len({r["id"] for r in prepared})==len(prepared) and {r["id"] for r in prepared}<={key[0] for key in expected},"Invalid prepared-input identities")
        if shard["status"]=="complete": require(set(rows)==expected,"Incomplete complete shard")
        for key,row in rows.items():
            require(key not in all_results,"Case appears on multiple shards")
            all_results[key]=row
        shards[s]=shard;hashes[str(path)]={"observed_sha256":current_sha,"aggregate_sha256":desc["sha256"],"observed_status":shard["status"]}
    for key,row in aggregate.items(): require(all_results.get(key)==row,"Aggregate result differs from atomic shard")
    if state["status"]=="complete":
        require(set(aggregate)==wanted and len(shards)==4 and all(s["status"]=="complete" for s in shards.values())
                and all(d["status"]=="complete" for d in descriptions.values()),"False complete run")
    return state,state_sha,shards,hashes,all_results


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
                    "phase": case["phase"], "phase_eligible": case["phase_eligible"], "plan_sha256": plan_sha,
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
    results={(r["id"],r["num_inference_steps"]):r for r in state["results"]}
    prior_ids={c["id"] for c in load_json(plan["prior_24_plan"])["cases"]}
    for item in items:
        trace=item["trace"]
        for key in IDENTITY_FIELDS:
            require(trace[key] == state[key], f"Trace/shard identity differs:{key}")
        require(trace["shard_index"] == state["shard_index"] == case["shard_index"] and trace["num_shards"] == 4,
                "Shard assignment differs")
        require(trace["fallback_reason"] == case["fallback_reason"] and trace["warmup_excluded"] is True, "Fallback/warmup differs")
        result=results[(case["id"],item["steps"])]
        require(result["status"] == "complete" and Path(result["trace"]).resolve() == Path(item["trace_path"])
                and result["timing"] == trace["timing"], "Shard result/trace differs")
        if case["id"] in prior_ids:
            replay=trace["prior_24_replay"]
            directory=Path(plan["prior_24_run"])/"samples"/case["id"]/f"steps_{item['steps']}"
            for name in ("trace","prediction"):
                path=directory/("trace.json" if name=="trace" else "future.mp4")
                require(Path(replay[f"previous_{name}"]).resolve()==path.resolve(), "Prior replay path differs")
                files.sha(path,replay[f"previous_{name}_sha256"])
            old=load_json(directory/"trace.json")
            files.sha(directory/"future.mp4",old["prediction_sha256"])
            fields=[k for k in shared.TRACE_FIELDS if k != "device"]
            fields+=list(shared.OPTIONAL_MATH_FIELDS)+["checkpoint","checkpoint_bytes","checkpoint_mtime_ns",
                    "input_tensor_sha256","initial_noise_sha256","initial_noise_shape"]
            for key in fields:
                require((key in old,old.get(key))==(key in trace,trace.get(key)), f"Prior replay math/input differs:{key}")
            expected={"id":case["id"],"num_inference_steps":item["steps"],**replay}
            require(expected in state["prior_24_replays"], "Missing shard prior replay")
        else:
            require("prior_24_replay" not in trace, "Unexpected prior replay")
    return {"case":case,"paths":paths,"hashes":hashes,"indices":indices,"anchor_source_index":anchor_source,
            "items":items,"width":dataset.video_size[1],"height":dataset.video_size[0],"fps":Fraction(6),
            "prior_overlap":case["id"] in prior_ids}


def measure(static,predictions,target):
    import numpy as np
    require(len(predictions)==2 and all(a.shape==target.shape for a in (static,*predictions)),"Metric shapes differ")
    common=np.logical_or.reduce([a.max(axis=-1)>16 for a in (target,static,*predictions)])
    methods={name:{**shared.metric_one(a,target,common),
                   "per_future_frame":[shared.metric_one(a[i],target[i],common[i]) for i in range(16)]}
             for name,a in zip(METHODS,(static,*predictions))}
    frozen,reference=methods["static"]["common_foreground_mae"],methods["steps6"]["common_foreground_mae"]
    for name in METHODS[1:]:
        score=methods[name];gain=frozen-score["common_foreground_mae"];delta=score["common_foreground_mae"]-reference
        score.update(mae_improvement_vs_static=gain,relative_mae_improvement_vs_static=gain/frozen if frozen>EPS else None,
                     mae_beats_static=gain>EPS,mae_ties_static=abs(gain)<=EPS,
                     mae_degradation_vs_steps6=delta,relative_mae_degradation_vs_steps6=delta/reference if reference>EPS else None,
                     iou_degradation_vs_steps6=methods["steps6"]["silhouette_iou"]-score["silhouette_iou"],
                     mae_beats_steps6=delta < -EPS,mae_ties_steps6=abs(delta)<=EPS)
    return {"common_foreground_pixels":int(common.sum()),"common_foreground_pixels_per_frame":common.sum(axis=(1,2)).tolist(),"methods":methods}


def render(directory,inputs,rgb,static,predictions,target):
    import av
    import numpy as np
    from PIL import Image,ImageDraw
    w,h,fps=inputs["width"],inputs["height"],inputs["fps"];case=inputs["case"]
    labels=["Full-scene RGB GT","Static object O_s","6 steps","4 steps"]
    def caption(i):
        t=inputs["indices"]["future_timestamps_us"][i];start=inputs["indices"]["condition_timestamps_us"][-1]
        return f"F{i+1}/16 | sim {t/1e6:.3f}s ({(t-start)/1e6:+.3f}s)"
    with av.open(str(directory/"comparison.mp4"),"w") as container:
        stream=container.add_stream("libx264",rate=fps);stream.width,stream.height,stream.pix_fmt=4*w,h+76,"yuv420p"
        stream.options={"crf":"18","preset":"fast","threads":"1"}
        for i in range(16):
            canvas=Image.new("RGB",(4*w,h+76),"#202936");draw=ImageDraw.Draw(canvas)
            for col,(label,frames) in enumerate(zip(labels,[rgb,static,*predictions])):
                draw.text((col*w+8,5),label,font=base.font(18),fill="white");canvas.paste(Image.fromarray(frames[i]),(col*w,76))
            draw.text((10,31),f"{case['task']} / {case['episode']} / {case['phase']} | s={case['anchor']} | {caption(i)}",font=base.font(17),fill="white")
            draw.text((10,54),"Checkpoint10000 / seed0 / shift3 | observer left, ego right per panel | 6fps display",font=base.font(14),fill="#d5dfef")
            frame=av.VideoFrame.from_ndarray(np.asarray(canvas),format="rgb24");frame.pts,frame.time_base=i,1/fps
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    for name,first,label in (("contact.png",rgb,labels[0]),("object_contact.png",target,"Object-only GT")):
        canvas=Image.new("RGB",(4*w,76+4*(h+32)),"#202936");draw=ImageDraw.Draw(canvas)
        draw.text((10,6),f"{case['task']} / {case['episode']} / {case['phase']} | s={case['anchor']} | eligible={case['phase_eligible']}",font=base.font(22),fill="white")
        for col,text in enumerate([label,*labels[1:]]):draw.text((col*w+8,43),text,font=base.font(18),fill="white")
        for row,i in enumerate((0,4,9,15)):
            y=76+row*(h+32);draw.text((10,y+4),caption(i),font=base.font(17),fill="white")
            for col,frames in enumerate([first,static,*predictions]):canvas.paste(Image.fromarray(frames[i]),(col*w,y+32))
        canvas.save(directory/name)
    _,decoded=base.read_prediction(directory/"comparison.mp4",4*w,h+76,fps)
    return decoded


def process_case(output,inputs,implementation_sha,plan_sha):
    import numpy as np
    case=inputs["case"]
    fingerprint={"implementation_sha256":implementation_sha,"plan_sha256":plan_sha,"case":case,"source_paths":inputs["paths"],
                 "source_sha256":inputs["hashes"],"indices":inputs["indices"],"anchor_source_index":inputs["anchor_source_index"],
                 "predictions":[{k:item[k] for k in ("steps","trace_path","trace_sha256","prediction_path","prediction_sha256")} for item in inputs["items"]]}
    signature=full.canonical_hash(fingerprint);target_dir=output/"samples"/case["id"]
    if target_dir.exists():
        report=load_json(target_dir/"verification.json")
        require(report["status"]=="passed" and report["input_signature"]==signature,"Cached case identity differs")
        for name,artifact in report["artifacts"].items():require(digest(target_dir/name)==artifact["sha256"],"Cached artifact changed")
        metrics=load_json(target_dir/"metrics.json");require(metrics["input_signature"]==signature,"Cached metric signature differs")
        return metrics,True
    staging=Path(tempfile.mkdtemp(prefix=f".{case['id']}-",dir=output/"samples"))
    try:
        w,h=inputs["width"],inputs["height"];ordinals=inputs["indices"]["future_source_indices"]
        rgb=base.read_source(inputs["paths"]["rgb"],ordinals,w,h)
        objects=base.read_source(inputs["paths"]["object"],[inputs["anchor_source_index"],*ordinals],w,h)
        target,static=objects[1:],np.broadcast_to(objects[0],objects[1:].shape)
        predictions=[];decodes=[];replays={}
        for item in inputs["items"]:
            frames,metadata=base.read_prediction(item["prediction_path"],w,h,inputs["fps"])
            predictions.append(frames);decodes.append(metadata)
            if inputs["prior_overlap"]:
                declared=item["trace"]["prior_24_replay"]
                old,_=base.read_prediction(declared["previous_prediction"],w,h,inputs["fps"])
                delta=np.abs(old.astype(np.int16)-frames.astype(np.int16))
                replay={"status":"pixel_identical" if np.array_equal(old,frames) else "different",
                        "max_absolute_pixel_difference":int(delta.max()),"mean_absolute_pixel_difference":float(delta.mean())}
                require(replay["status"]==declared["status"] and replay["max_absolute_pixel_difference"]==declared["decoded_max_abs_difference_uint8"]
                        and math.isclose(replay["mean_absolute_pixel_difference"],declared["decoded_mean_abs_difference_uint8"],abs_tol=1e-12),"Independent prior replay differs")
                replays[f"steps{item['steps']}"]=replay
        metrics={"schema":SCHEMA+".case_metrics","status":"passed","input_signature":signature,
                 **{k:case[k] for k in ("id","task","episode","phase","phase_eligible","fallback_reason","anchor","shard_index","variant_order")},
                 "prior24_overlap":inputs["prior_overlap"],"physical_gpu_ids":inputs["items"][0]["trace"]["physical_gpu_ids"],
                 **measure(static,predictions,target),"prior24_replay":replays,
                 "decoded_prediction_rgb_sha256":{f"steps{item['steps']}":hashlib.sha256(frames.tobytes()).hexdigest() for item,frames in zip(inputs["items"],predictions)},
                 "decoded_target_rgb_sha256":hashlib.sha256(target.tobytes()).hexdigest(),
                 "static_anchor_rgb_sha256":hashlib.sha256(objects[0].tobytes()).hexdigest(),
                 "timing":{f"steps{item['steps']}":item["trace"]["timing"] for item in inputs["items"]}}
        base.write_json(staging/"metrics.json",metrics)
        comparison=render(staging,inputs,rgb,static,predictions,target)
        report={"schema":SCHEMA+".case_verification","status":"passed","input_signature":signature,"input":fingerprint,
                "matched_input":{k:inputs["items"][0]["trace"][k] for k in shared.TRACE_FIELDS if k!="num_inference_steps"},
                "input_tensor_sha256":inputs["items"][0]["trace"]["input_tensor_sha256"],
                "initial_noise_sha256":inputs["items"][0]["trace"]["initial_noise_sha256"],
                "actual_production_input_and_cpu_noise_verified":True,"predictions_decoded":decodes,"comparison_decode":comparison,
                "prior24_replay":replays,"artifacts":{name:{"sha256":digest(staging/name),"bytes":(staging/name).stat().st_size}
                   for name in ("metrics.json","comparison.mp4","contact.png","object_contact.png")}}
        base.write_json(staging/"verification.json",report);staging.replace(target_dir)
        return metrics,False
    finally:
        if staging.exists():shutil.rmtree(staging)


def unique_windows(cases):
    # Predeclared representative: pre_onset before mid_motion, never chosen by score.
    kept={}
    for case in sorted(cases,key=lambda c:(c["task"],c["episode"],c["anchor"],PHASES.index(c["phase"]),c["id"])):
        kept.setdefault((case["task"],case["episode"],case["anchor"]),case)
    return list(kept.values())


def aggregate_group(cases,expected_tasks):
    grouped=defaultdict(lambda:defaultdict(list))
    for case in cases:grouped[case["task"]][case["episode"]].append(case)
    per_task={}
    for task,episodes in sorted(grouped.items()):
        methods={}
        for method in METHODS:
            methods[method]={key:statistics.fmean(statistics.fmean(c["methods"][method][key] for c in rows) for rows in episodes.values()) for key in SCALARS}
            if method!="static":
                for out,source in (("mae_win_rate_vs_static","mae_beats_static"),("mae_tie_rate_vs_static","mae_ties_static"),
                                   ("mae_win_rate_vs_steps6","mae_beats_steps6"),("mae_tie_rate_vs_steps6","mae_ties_steps6")):
                    methods[method][out]=statistics.fmean(statistics.fmean(float(c["methods"][method][source]) for c in rows) for rows in episodes.values())
        per_task[task]={"case_count":sum(map(len,episodes.values())),"episode_count":len(episodes),"methods":methods}
    macro=None
    if per_task:
        macro={method:{key:statistics.fmean(t["methods"][method][key] for t in per_task.values()) for key in next(iter(per_task.values()))["methods"][method]} for method in METHODS}
        for values in [macro,*[t["methods"] for t in per_task.values()]]:
            frozen,reference=values["static"]["common_foreground_mae"],values["steps6"]["common_foreground_mae"]
            for method in METHODS[1:]:
                v=values[method];gain=frozen-v["common_foreground_mae"];delta=v["common_foreground_mae"]-reference
                v.update(mae_improvement_vs_static=gain,relative_mae_improvement_vs_static=gain/frozen if frozen>EPS else None,
                         mae_degradation_vs_steps6=delta,relative_mae_degradation_vs_steps6=delta/reference if reference>EPS else None,
                         iou_degradation_vs_steps6=values["steps6"]["silhouette_iou"]-v["silhouette_iou"])
    return {"case_count":len(cases),"episode_count":sum(len(v) for v in grouped.values()),"covered_task_count":len(grouped),
            "expected_task_count":len(expected_tasks),"missing_tasks":sorted(set(expected_tasks)-set(grouped)),
            "task_macro":macro,"per_task":per_task}


def aggregate(cases,plan):
    prior_ids={c["id"] for c in load_json(plan["prior_24_plan"])["cases"]};groups={}
    for cohort,rows in (("all",cases),("additional456",[c for c in cases if not c["prior24_overlap"]]),
                        ("prior24",[c for c in cases if c["prior24_overlap"]])):
        cohort_tasks=sorted({c["task"] for c in plan["cases"] if (cohort=="all" or (c["id"] in prior_ids)==(cohort=="prior24"))})
        groups[cohort+"/overall_unique"]=aggregate_group(unique_windows(rows),cohort_tasks)
        groups[cohort+"/all_phase_records"]=aggregate_group(rows,cohort_tasks)
        for phase in PHASES:
            phase_rows=[c for c in rows if c["phase"]==phase]
            groups[f"{cohort}/{phase}/all"]=aggregate_group(phase_rows,cohort_tasks)
            groups[f"{cohort}/{phase}/eligible"]=aggregate_group([c for c in phase_rows if c["phase_eligible"]],cohort_tasks)
            groups[f"{cohort}/{phase}/fallback"]=aggregate_group([c for c in phase_rows if not c["phase_eligible"]],cohort_tasks)
    return groups


def timing_group(cases):
    methods={f"steps{n}":{field:previous.distribution([c["timing"][f"steps{n}"][field] for c in cases]) for field in TIMINGS} for n in STEPS}
    paired={}
    for field in TIMINGS:
        six,four=methods["steps6"][field]["mean"],methods["steps4"][field]["mean"]
        paired[field]={"ratio_of_means_6_over_4":six/four if four is not None and four>EPS else None,
                       "relative_latency_reduction_4_vs_6":1-four/six if six is not None and six>EPS else None,
                       "paired_ratio_6_over_4":previous.distribution([c["timing"]["steps6"][field]/c["timing"]["steps4"][field] for c in cases if c["timing"]["steps4"][field]>EPS]),
                       "paired_difference_6_minus_4_seconds":previous.distribution([c["timing"]["steps6"][field]-c["timing"]["steps4"][field] for c in cases])}
    return {"paired_case_count":len(cases),"scored_call_count":2*len(cases),"methods":methods,"paired":paired}


def timing_summary(cases):
    return {"overall_all_calls":timing_group(cases),
            "by_physical_gpu":{gpu:timing_group([c for c in cases if c["physical_gpu_ids"]==gpu]) for gpu in sorted({c["physical_gpu_ids"] for c in cases})},
            "by_phase":{phase:timing_group([c for c in cases if c["phase"]==phase]) for phase in PHASES},
            "by_variant_order":{str(list(order)):timing_group([c for c in cases if tuple(c["variant_order"])==order]) for order in ((6,4),(4,6))}}


def duplicate_checks(cases):
    grouped=defaultdict(list)
    for c in cases:grouped[(c["task"],c["episode"],c["anchor"])].append(c)
    checks=[]
    for (task,episode,anchor),rows in grouped.items():
        if len(rows)<2:continue
        reference=unique_windows(rows)[0]
        require(all(c["decoded_target_rgb_sha256"]==reference["decoded_target_rgb_sha256"] and c["static_anchor_rgb_sha256"]==reference["static_anchor_rgb_sha256"] for c in rows),"Duplicate-window GT/static differs")
        checks.append({"task":task,"episode":episode,"anchor":anchor,"case_ids":[c["id"] for c in rows],"representative":reference["id"],
                       "all_fallback":all(not c["phase_eligible"] for c in rows),
                       "same_decoded_prediction_pixels":{method:all(c["decoded_prediction_rgb_sha256"][method]==reference["decoded_prediction_rgb_sha256"][method] for c in rows) for method in METHODS[1:]}})
    return checks


def definitions():
    return {"metric_target":"object_only_gt","foreground_threshold_uint8":16,
            "common_mask_rule":"Per frame/pixel maxRGB>16 union of objectGT|static_O_s|current6|current4. All three scored methods share the MAE denominator.",
            "mae_rule":"Pool16frames RGB absolute error on common mask/(foreground pixels*3*255),0..1,lower better.",
            "iou_rule":"Per-method foreground intersection/union with objectGT pooled over16frames;higher better.",
            "static_rule":"Actual raw object video frame condition_source_indices[-1],O_s,repeated16;never generated first frame.",
            "aggregation_rule":"All groups:within episode equal records/windows,within task equal episodes,then equal covered tasks. Overall_unique first deduplicates(task,episode,anchor),pre_onset preferred deterministically. Phase groups retain all records and split phase_eligible/fallback. all_phase_records is explicitly supplementary and double-counts16 physical windows.",
            "coverage_rule":"480 phase records,240episodes,50tasks,464unique temporal windows. Eligible pre215/mid210;fallback25/30. A missing pixel-onset detection does not mean no physical action. Missing eligible tasks are explicitly listed and not assigned zero scores.",
            "cohort_rule":"additional456=228episodes/50tasks/440unique windows not in the prior24-case4/6 experiment;prior24=12episodes/12tasks/24unique. These test examples were previously used for checkpoint evaluation:neither cohort is independent new validation. Compare4vs6 within each cohort,not raw cohort means as a generalization effect.",
            "win_rule":"Static/6-step MAE wins use >1e-12 improvement,ties abs(delta)<=1e-12,weighted by episode/task as the quality means. Positive modelMAE-6MAE means degradation. Relative aggregate differences divide by6 aggregate MAE;null for near-zero denominator. Win rates are not action success rates.",
            "timing_rule":"All480paired cases/960actual scored calls retained,including duplicate physical windows;no deduplication. Samecase6/4 sameGPU,balanced order by shard/phase. Report perGPU and pooled mean/median/p95 plus distribution of paired6/4ratios. p95 uses NumPy linear interpolation. Pooled timing is call-weighted,not task macro.",
            "timing_scope":"Model:CUDA-synchronized conditionVAE/CLIP+NDiT+full25decode/output conversion. Processing:pre-read-warmed conditionload+model+MP4export. Excludes loading,8warmups total,input hashes,trace writes,GC,replay checks;not cold-start end-to-end latency.",
            "background_rule":"GT-background predicted foreground rate/intensity also reflect displacement errors;not a pure clutter-retention measure.",
            "empty_conventions":"Empty commonmask MAE0;empty pairunion IoU1;zeroGTbackground rates0;zero-denominator ratiosnull.",
            "cross_report_warning":"This four-image mask differs from previous5/7-image masks and8k/10k masks. All quality scores are recomputed;do not directly compare MAE across reports.",
            "scope":"Complete existing clean test split,fixedGT-based selection and explicitfallbacks. Exploratory solver-step comparison,not independent validation,action success,or statistical-significance evidence."}


def csv_document(cases,groups,timing):
    out=io.StringIO(newline="")
    fields=["row_type","group","task","episode","id","phase","phase_eligible","prior24_overlap","anchor","physical_gpu_ids","method",*SCALARS,
            "mae_win_rate_vs_static","mae_win_rate_vs_steps6","mae_degradation_vs_steps6","relative_mae_degradation_vs_steps6",
            "timing_field","n","mean","median","p95"]
    writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
    for c in cases:
        for method in METHODS:
            writer.writerow({"row_type":"case",**{k:c[k] for k in ("task","episode","id","phase","phase_eligible","prior24_overlap","anchor","physical_gpu_ids")},"method":method,
                             **{k:v for k,v in c["methods"][method].items() if k in fields}})
    for name,group in groups.items():
        for method,score in (group["task_macro"] or {}).items():
            writer.writerow({"row_type":"task_macro","group":name,"method":method,**{k:v for k,v in score.items() if k in fields}})
        for task,stats in group["per_task"].items():
            for method,score in stats["methods"].items():writer.writerow({"row_type":"task","group":name,"task":task,"method":method,**{k:v for k,v in score.items() if k in fields}})
    for name,stats in [("all_calls",timing["overall_all_calls"]),*[(f"gpu{gpu}",s) for gpu,s in timing["by_physical_gpu"].items()]]:
        for method,values in stats["methods"].items():
            for field,value in values.items():writer.writerow({"row_type":"timing","group":name,"method":method,"timing_field":field,**value})
    return out.getvalue()


STYLE="""<style>body{max-width:1792px;margin:24px auto;padding:0 18px;background:#151b25;color:#eef3fc;font:16px/1.6 system-ui}a{color:#a9ceff}section,article{padding:18px;margin:24px 0;background:#253144}table{border-collapse:collapse;width:100%}td,th{padding:7px;border:1px solid #526174;text-align:left}video,img{display:block;width:100%;height:auto}select{padding:8px;margin:6px}[hidden]{display:none}</style>"""


def document(title,body):
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title>'+STYLE+body+'</html>'


def task_page(task,cases):
    cards=[]
    for c in cases:
        folder=f"samples/{c['id']}";s6=c["methods"]["steps6"];s4=c["methods"]["steps4"]
        eligibility="eligible" if c["phase_eligible"] else f"fallback: {c['fallback_reason']}"
        cards.append(f'''<article data-phase="{c['phase']}" data-eligible="{str(c['phase_eligible']).lower()}"><h2>{html.escape(c['episode'])} / {c['phase']} / s={c['anchor']}</h2>
<p>{html.escape(eligibility)}；{'此前24例之一' if c['prior24_overlap'] else '本轮新增阶段记录'}。MAE 6→4：{s6['common_foreground_mae']:.6f}→{s4['common_foreground_mae']:.6f}；IoU {s6['silhouette_iou']:.4f}→{s4['silhouette_iou']:.4f}。</p>
<video controls loop playsinline preload="none" src="{folder}/comparison.mp4"></video>
<p>全景RGB GT | static O_s | 6步 | 4步；每格observer左、ego右。</p>
<details><summary>物体GT四列关键帧（未来1/5/10/16）</summary><a href="{folder}/object_contact.png"><img loading="lazy" src="{folder}/object_contact.png"></a></details>
<p><a href="{folder}/contact.png">全景对照关键帧</a> · <a href="{folder}/metrics.json">逐帧指标与计时</a> · <a href="{folder}/verification.json">输入、噪声、视频来源校验</a></p></article>''')
    return document(task+":6/4步",f'''<h1>{html.escape(task)}：6步与4步</h1><p><a href="index.html">回总览</a>；{len(cases)}条记录。GT用于选窗和评估，不输入推理。预览视频不自动加载。</p>
<select id="phase"><option value="">全部阶段</option><option>pre_onset</option><option>mid_motion</option></select><select id="eligible"><option value="">eligible与fallback</option><option value="true">eligible</option><option value="false">fallback</option></select>
{''.join(cards)}<script>function filter(){{document.querySelectorAll('article').forEach(e=>e.hidden=['phase','eligible'].some(k=>document.getElementById(k).value&&e.dataset[k]!==document.getElementById(k).value));}}['phase','eligible'].forEach(k=>document.getElementById(k).addEventListener('change',filter));</script>''')


def index_page(cases,status,groups,timing,duplicates):
    tables=[]
    for name,stats in groups.items():
        rows=[]
        for method,v in (stats["task_macro"] or {}).items():
            delta=v.get("relative_mae_degradation_vs_steps6");wins=v.get("mae_win_rate_vs_static")
            rows.append(f"<tr><td>{method}</td><td>{v['common_foreground_mae']:.6f}</td><td>{v['silhouette_iou']:.6f}</td><td>{'—' if wins is None else f'{wins:.1%}'}</td><td>{'—' if delta is None else f'{delta:+.2%}'}</td></tr>")
        tables.append(f"<section><h2>{name}</h2><p>{stats['case_count']}条用于该汇总，{stats['episode_count']}episodes，{stats['covered_task_count']}/{stats['expected_task_count']}tasks；缺失任务：{', '.join(stats['missing_tasks']) or '无'}。</p><table><tr><th>方法</th><th>共同前景MAE↓</th><th>IoU↑</th><th>MAE胜静止比例</th><th>相对6步MAE变化（正为退化）</th></tr>{''.join(rows)}</table></section>")
    timings=[]
    for name,stats in [("所有实际调用",timing["overall_all_calls"]),*[("GPU"+gpu,s) for gpu,s in timing["by_physical_gpu"].items()]]:
        rows=[]
        for method,values in stats["methods"].items():
            for field in ("model_inference_seconds","processing_seconds"):
                v=values[field]
                if v["n"]:rows.append(f"<tr><td>{method}</td><td>{field}</td><td>{v['n']}</td><td>{v['mean']:.3f}</td><td>{v['median']:.3f}</td><td>{v['p95']:.3f}</td></tr>")
        pairs=[]
        for field in ("model_inference_seconds","processing_seconds"):
            p=stats["paired"][field]
            if p["ratio_of_means_6_over_4"] is not None:pairs.append(f"{field}:均值比6/4={p['ratio_of_means_6_over_4']:.3f}×，4步耗时减少{p['relative_latency_reduction_4_vs_6']:.2%}；逐例配对比中位{p['paired_ratio_6_over_4']['median']:.3f}×")
        timings.append(f"<section><h2>计时：{name}</h2><table><tr><th>步数</th><th>范围</th><th>n</th><th>均值秒</th><th>中位秒</th><th>p95秒</th></tr>{''.join(rows)}</table><p>{'<br>'.join(pairs)}</p></section>")
    tasks=sorted({c["task"] for c in cases})
    tasklinks=''.join(f'<li><a href="task_{html.escape(t)}.html">{html.escape(t)}</a>：{sum(c["task"]==t for c in cases)}条记录</li>' for t in tasks)
    same=sum(all(d["same_decoded_prediction_pixels"].values()) for d in duplicates)
    return document("Checkpoint10000全clean：6步与4步",f'''<h1>Checkpoint10000：完整clean测试集6步与4步</h1>
<p>状态：{status}；完成{len(cases)}/480阶段记录。{'结果完整。' if status=='passed' else '部分结果，不能作为完整50任务结论。'}</p>
<p>固定240episodes/50tasks、480阶段记录，实际464个不同物理窗口；总体质量预先按pre_onset优先去重，再窗口→episode→task等权。eligible pre215/mid210；fallback25/30单列。当前可核验重复组{len(duplicates)}，其中{same}组的6/4输出均逐像素相同；若不相同仍按预定代表项，不按指标择优。</p>
<p>四路共同mask=物体GT | 真实static O_s | 6步 | 4步，三种方法共用分母。GT物体计分，全景RGB GT只展示；GT用于原固定选窗和事后对照，不输入模型。新mask与旧报告不同，旧MAE不能直接拼接。像素分数不保证形状完整或动作成功。</p>
<p>新增456阶段记录（228episodes、440unique）和此前24例分别汇总。同一批test已用于过去checkpoint评估，新增不是独立新验证集；不要直接用两个cohort均值差声称泛化变化。未检出像素位移onset不代表绝对没有物理动作。</p>
<p>耗时保留全部960实际调用，不因重复窗口去重。同一case在同GPU做6/4、各shard/phase顺序平衡。模型计时含条件VAE/CLIP、DiT、25帧解码；处理计时为预读后的条件读取+模型+导出，排除加载、8次warmup、哈希检查、trace写入、GC和重放。不是冷启动端到端耗时。</p>
<p><a href="metrics.json">完整指标JSON</a> · <a href="metrics.csv">CSV</a> · <a href="summary.json">审计摘要</a></p>
<h2>按任务浏览（分页，无480个视频自动加载）</h2><ul>{tasklinks}</ul>{''.join(tables)}{''.join(timings)}''')


_WORKER={}

def worker_init(run,output,plan,plan_sha,shards,implementation_sha):
    import torch
    from vpp2_obj.infer import load_config,make_dataset
    torch.set_num_threads(1)
    dataset=make_dataset(load_config(plan["config"]),Path(plan["test_manifest"]))
    _WORKER.update(run=Path(run),output=Path(output),plan=plan,plan_sha=plan_sha,shards=shards,implementation_sha=implementation_sha,
                   dataset=dataset,index={(r["task"],r["episode"]):i for i,r in enumerate(dataset.records)},files=full.Files(),caches=set())


def worker_case(case):
    s=_WORKER
    inputs=prepare_case(s["run"],s["plan"],s["plan_sha"],s["shards"][case["shard_index"]],case,s["dataset"],s["index"],s["files"],s["caches"])
    metrics,reused=process_case(s["output"],inputs,s["implementation_sha"],s["plan_sha"])
    return metrics,reused,{path:value for path,(_,value) in s["files"].entries.items()}


def evaluate(run,output,allow_partial,workers):
    import time
    started=time.monotonic()
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/".evaluate.lock").open("a")
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise ValueError("Another evaluator owns this output")
    files=full.Files();plan,plan_sha,dataset,index,prior_ids=prepare_plan(run,files)
    state,state_sha,shards,shard_hashes,results=snapshot(run,plan,plan_sha)
    if not allow_partial:require(state["status"]=="complete","Run incomplete;use --allow-partial")
    paths=(Path(__file__),HELPER,previous.HELPER,full.HELPER,shared.HELPER)
    implementation={str(p.resolve()):digest(p) for p in paths};implementation_sha=full.canonical_hash(implementation)
    contract={"schema":SCHEMA+".contract","run":str(run),"plan_sha256":plan_sha,"implementation":implementation,"definitions":definitions()}
    if (output/"evaluation_contract.json").exists():require(load_json(output/"evaluation_contract.json")==contract,"Frozen evaluation contract changed;use new output")
    else:full.atomic_json(output/"evaluation_contract.json",contract)
    (output/"samples").mkdir(exist_ok=True)
    available=[c for c in plan["cases"] if all((c["id"],n) in results for n in STEPS)]
    pending=[c["id"] for c in plan["cases"] if c not in available]
    completed=[];reused=0;checked={path:value for path,(_,value) in files.entries.items()}
    def accept(result):
        nonlocal reused
        metric,used,hashes=result;completed.append(metric);reused+=used
        for path,value in hashes.items():
            require(path not in checked or checked[path]==value,"Input content differs across CPU workers")
            checked[path]=value
        full.atomic_json(output/"progress.json",{"status":"partial","completed_case_count":len(completed),"available_at_snapshot":len(available),"planned_case_count":480,
                                                 "reused_case_count":reused,"last_case":metric["id"],"elapsed_seconds":time.monotonic()-started})
        print(f"{'Reused' if used else 'Evaluated'} {len(completed)}/{len(available)} available (480 planned): {metric['id']}",flush=True)
    args=(str(run),str(output),plan,plan_sha,shards,implementation_sha)
    if workers==1:
        worker_init(*args)
        for c in available:accept(worker_case(c))
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),initializer=worker_init,initargs=args) as pool:
            futures=[pool.submit(worker_case,c) for c in available]
            for future in as_completed(futures):accept(future.result())
    order={c["id"]:i for i,c in enumerate(plan["cases"])};completed.sort(key=lambda c:order[c["id"]])
    final_state,final_sha,final_shards,final_hashes,final_results=snapshot(run,plan,plan_sha)
    for key,row in results.items():require(final_results.get(key)==row,"Published results changed rather than appending")
    final=len(completed)==480 and final_state["status"]=="complete"
    require(allow_partial or final,"Missing matching predictions")
    status="passed" if final else "partial"
    groups=aggregate(completed,plan);timing=timing_summary(completed);duplicates=duplicate_checks(completed)
    compact=[{**{k:v for k,v in c.items() if k!="methods"},"methods":{m:{k:v for k,v in values.items() if k!="per_future_frame"} for m,values in c["methods"].items()}} for c in completed]
    metrics={"schema":SCHEMA+".metrics","status":status,"case_count":len(completed),"planned_case_count":480,**definitions(),
             "coverage":full.coverage(plan,completed),"planned_temporal_coverage":full.temporal_coverage(plan["cases"]),
             "completed_temporal_coverage":full.temporal_coverage(completed),"duplicate_window_checks":duplicates,
             "aggregates":groups,"timing":timing,"cases":compact}
    pages={f"task_{task}.html":task_page(task,[c for c in completed if c["task"]==task]) for task in sorted({c["task"] for c in completed})}
    for name,content in pages.items():full.atomic_text(output/name,content)
    page=index_page(completed,status,groups,timing,duplicates)
    full.atomic_text(output/"index.html",page)
    links=full.check_links(page,output)+sum(full.check_links(content,output) for content in pages.values())
    for path,expected in implementation.items():require(digest(path)==expected,"Implementation changed during evaluation")
    # Immutable inputs are rehashed in the parent; running shard/run snapshots
    # are separately recorded, never included in this immutable-source loop.
    for path,expected in checked.items():require(digest(path)==expected,f"Source changed during CPU evaluation:{path}")
    contents={"metrics.json":full.json_text(metrics),"metrics.csv":csv_document(completed,groups,timing),"index.html":page}
    for name,value in contents.items():full.atomic_text(output/name,value)
    replay_rows=[{"id":c["id"],"method":method,**v} for c in completed for method,v in c["prior24_replay"].items()]
    replay_summary={"checked_prediction_count":len(replay_rows),"expected_prediction_count":48,
                    "pixel_identical_count":sum(r["status"]=="pixel_identical" for r in replay_rows),
                    "max_absolute_pixel_difference":max((r["max_absolute_pixel_difference"] for r in replay_rows),default=None),
                    "different": [r for r in replay_rows if r["status"]!="pixel_identical"],
                    "scope":"Only the24overlap cases x2variants decoded16MP4frames;passed does not imply all replay outputs identical."}
    artifacts={name:{"sha256":digest(output/name),"bytes":(output/name).stat().st_size} for name in (*contents,*pages)}
    summary={"schema":SCHEMA,"status":status,"created_at_utc":datetime.now(timezone.utc).isoformat(),"run":str(run),"case_count":len(completed),
             "planned_case_count":480,"pending_case_ids":pending,"reused_case_count":reused,"elapsed_seconds":time.monotonic()-started,"cpu_workers":workers,
             "plan_sha256":plan_sha,"initial_run_json_sha256":state_sha,"final_run_json_sha256":final_sha,
             "initial_shard_snapshots":shard_hashes,"final_shard_snapshots":final_hashes,"final_run_metadata":final_state,
             "coverage":metrics["coverage"],"planned_temporal_coverage":metrics["planned_temporal_coverage"],"completed_temporal_coverage":metrics["completed_temporal_coverage"],
             "duplicate_window_checks":duplicates,"prior24_replay":replay_summary,"aggregates":groups,"timing":timing,
             "implementation":implementation,"input_content_sha256":checked,"html_local_links_checked":links,
             "case_verifications":[{"id":c["id"],"path":f"samples/{c['id']}/verification.json","sha256":digest(output/"samples"/c["id"]/"verification.json")} for c in completed],
             "gpu_used":False,"model_loaded":False,"original_predictions_modified":False,"artifacts":artifacts,"scope":definitions()["scope"]}
    full.atomic_json(output/"summary.json",summary)
    full.atomic_json(output/"progress.json",{"status":status,"completed_case_count":len(completed),"planned_case_count":480,"reused_case_count":reused,"elapsed_seconds":time.monotonic()-started})
    print(full.json_text({"status":status,"case_count":len(completed),"output":str(output),"overall_unique":groups["all/overall_unique"]["task_macro"],
                          "model_timing":timing["overall_all_calls"]["paired"]["model_inference_seconds"],"elapsed_seconds":time.monotonic()-started}))
    return summary


def self_test():
    import copy
    import numpy as np
    gt=np.zeros((16,8,8,3),dtype=np.uint8);gt[:,1:3,1:3]=255
    static=np.zeros_like(gt);extra=gt.copy();extra[:,5:7,5:7]=255
    result=measure(static,[gt,extra],gt)
    require(result["methods"]["static"]["common_foreground_mae"]==.5 and result["methods"]["steps4"]["common_foreground_mae"]==.5,"Four-image shared denominator failed")
    require(result["methods"]["steps6"]["silhouette_iou"]==1 and result["methods"]["steps4"]["silhouette_iou"]==.5,"IoU failed")
    require(result["methods"]["steps4"]["relative_mae_degradation_vs_steps6"] is None,"Zero reference must be null")
    rows=[]
    for task,count,value in (("A",1,.2),("B",3,.8)):
        for ep in range(count):
            methods=copy.deepcopy(result["methods"])
            for method in METHODS:methods[method]["common_foreground_mae"]=value
            rows.append({"id":f"{task}{ep}pre","task":task,"episode":str(ep),"anchor":0,"phase":"pre_onset","methods":methods,
                         "timing":{f"steps{n}":{k:float(n) for k in TIMINGS} for n in STEPS}})
    agg=aggregate_group(rows,["A","B","missing"])
    require(math.isclose(agg["task_macro"]["steps4"]["common_foreground_mae"],.5) and agg["missing_tasks"]==["missing"],"Unequal task/episode aggregation failed")
    duplicate=copy.deepcopy(rows[0]);duplicate.update(id="A0mid",phase="mid_motion")
    duplicate["methods"]["steps4"]["common_foreground_mae"]=.99
    selected=unique_windows([duplicate,*rows])
    require(len(selected)==4 and next(c for c in selected if c["task"]=="A")["id"]=="A0pre","Dedup must prefer pre regardless of score/input order")
    timing=timing_group(rows)
    require(timing["paired"]["model_inference_seconds"]["ratio_of_means_6_over_4"]==1.5
            and timing["paired"]["model_inference_seconds"]["paired_ratio_6_over_4"]["median"]==1.5,"Paired timing failed")
    empty=measure(static,[static,static],static)
    require(empty["methods"]["steps4"]["common_foreground_mae"]==0 and empty["methods"]["steps4"]["silhouette_iou"]==1,"Empty mask convention failed")
    print(full.json_text({"status":"self_test_passed","checks":["shared4-image mask","pairIoU","zero ratio","task/episode weights and missing task","predeclared duplicate representative","paired timing","empty masks"],"gpu_used":False}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path);parser.add_argument("--output",type=Path)
    parser.add_argument("--allow-partial",action="store_true");parser.add_argument("--workers",type=int,default=4);parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    import torch
    torch.set_num_threads(1)
    if args.self_test:self_test();return
    require(args.run and args.output and 1<=args.workers<=4,"Provide --run --output and1..4CPUworkers")
    run,output=args.run.resolve(),args.output.resolve()
    require(run!=output and run not in output.parents and output not in run.parents,"Output must be separate from immutable inference run")
    evaluate(run,output,args.allow_partial,args.workers)


if __name__=="__main__":main()

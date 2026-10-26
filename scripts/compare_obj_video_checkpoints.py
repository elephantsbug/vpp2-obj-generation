#!/usr/bin/env python3
"""CPU comparison of matched step-4000, step-8000 and step-10000 predictions.

Full-scene RGB is display-only; all metrics use object-only GT. Existing runs
are immutable inputs. No model, VAE, text encoder or CUDA operation is called.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import csv
import html
from html.parser import HTMLParser
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import unquote, urlsplit

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

HELPER = Path(__file__).with_name("build_obj_video_review.py")
spec = importlib.util.spec_from_file_location("_checkpoint_review_source", HELPER)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
require, digest, load_json, write_json = base.require, base.digest, base.load_json, base.write_json
STEPS = (4000, 8000, 10000)
CHOSEN = (0, 4, 9, 15)
FOREGROUND_THRESHOLD = 16
TRACE_FIELDS = (
    "config", "config_sha256", "manifest", "manifest_sha256", "task", "episode",
    "anchor_index", "sampling", "condition_video_path", "text_embedding_path",
    "text_embedding_sha256", "text_embedding_cache_signature", "prompt", "seed",
    "num_inference_steps", "device", "exported_frames", "decoded_frame_slice",
    "prefix_clamped", "target_video_read", "target_first_frame_read", "display_fps",
    "condition_shape", "context_shape", "context_mask", "model_visible_text_tokens",
    "memory_strategy", "ground_truth_used_for_window_selection",
)
OPTIONAL_MATH_FIELDS = (
    "sigma_shift", "rand_device", "generator_device", "tiled", "model_dtype",
    "vae_dtype", "clip_dtype", "mixed_precision", "text_cfg_scale", "action_cfg_scale",
)
SOURCE_FIELDS = (
    "source_path", "source_sha256", "indices", "width", "height", "fps",
    "dedup_index_path", "dedup_index_sha256", "metric_target_video_path",
    "metric_target_video_sha256",
)
SCALAR_METRICS = (
    "common_foreground_mae", "silhouette_iou", "gt_background_predicted_foreground_rate",
    "mean_prediction_intensity_on_gt_background",
)


def assert_matching(items):
    """Reject a changed condition or execution strategy before rendering."""
    require(len(items) == 3, "Exactly three checkpoint records are required")
    first = items[0]
    for item in items:
        require(item["id"] == first["id"], "Case IDs or order differ")
        for field in TRACE_FIELDS:
            require(field in item["trace"], f"{item['id']}: missing comparison field {field}")
            require(item["trace"][field] == first["trace"][field],
                    f"{item['id']}: checkpoint comparison differs in {field}")
        for field in OPTIONAL_MATH_FIELDS:
            require((field in item["trace"], item["trace"].get(field)) ==
                    (field in first["trace"], first["trace"].get(field)),
                    f"{item['id']}: optional inference setting differs: {field}")
        for field in SOURCE_FIELDS:
            require(item[field] == first[field], f"{item['id']}: source/trajectory differs: {field}")


def checkpoint_step(path):
    match = re.fullmatch(r"step_(\d+)\.(?:pt|pth)", Path(path).name)
    require(match is not None, f"Cannot identify checkpoint step: {path}")
    return int(match.group(1))


def validate_cached_text(item):
    import torch
    trace = item["trace"]
    path = Path(trace["text_embedding_path"])
    require(digest(path) == trace["text_embedding_sha256"], "Text embedding changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload["schema"] == "vpp2.frozen_wan_text_embedding_cache.v1", "Unexpected text cache schema")
    require(payload["prompt"] == trace["prompt"] and payload["cache_signature"] == trace["text_embedding_cache_signature"],
            "Text cache identity mismatch")
    context, mask = payload["context"], payload["mask"]
    require(tuple(context.shape) == (128, 4096) and context.dtype == torch.bfloat16
            and bool(torch.isfinite(context).all()), "Invalid cached context")
    require(tuple(mask.shape) == (128,) and mask.dtype == torch.bool
            and mask.tolist() == trace["context_mask"], "Trace changed the real text mask")
    count = int(mask.sum())
    require(count == trace["model_visible_text_tokens"] == payload["valid_token_count"]
            and torch.equal(mask, torch.arange(128) < count), "Invalid cached token count/padding")


def prepare_run(run, step):
    """Reuse production trajectory/media validation, then verify extra inputs."""
    items, provenance = base.prepare(run, "full-scene")
    require(provenance["selection_plan_sha256"] is not None, "A frozen selection.json is required")
    checkpoints, cached = set(), set()
    for item in items:
        trace = item["trace"]
        path = Path(trace["checkpoint"]).resolve()
        require(path.is_file() and checkpoint_step(path) == step, f"Expected checkpoint step {step}")
        checkpoints.add(str(path))
        require(trace["prefix_clamped"] is False, "This comparison preserves the no-clamping baseline")
        require(not {"video", "target_video_path", "future_source_indices", "future_timestamps_us"}
                .intersection(trace["sampling"]), "Inference trace includes target-only sampling fields")
        require(digest(item["dedup_index_path"]) == item["dedup_index_sha256"], "Dedup trajectory index changed")
        require(digest(item["metric_target_video_path"]) == item["metric_target_video_sha256"], "Object GT changed")
        # The older 2000/4000 comparison stores the current run's fingerprints
        # under `current`; the general review helper checks top-level ones only.
        frozen_review = item["trace_path"].parent / "review/decode_check.json"
        if frozen_review.is_file():
            old = load_json(frozen_review)
            current = old.get("current")
            if isinstance(current, dict):
                require(current.get("prediction_sha256") == item["prediction_sha256"], "Prediction changed since nested frozen review")
                require(current.get("trace_sha256") == item["trace_sha256"], "Trace changed since nested frozen review")
        memory = trace.get("memory_strategy", {})
        require(all(field in memory for field in ("method", "dit_dtype", "vae_dtype", "clip_dtype", "quantization")),
                "Trace must record placement, all component dtypes and quantization")
        cache_key = (trace["text_embedding_path"], trace["text_embedding_sha256"],
                     tuple(trace["context_mask"]), trace["model_visible_text_tokens"])
        if cache_key not in cached:
            validate_cached_text(item)
            cached.add(cache_key)
    require(len(checkpoints) == 1, "Each run must contain exactly one checkpoint")
    return items, {"run": str(run), "step": step, "checkpoint": next(iter(checkpoints)), **provenance}


def prepare(runs):
    loaded, provenance = [], []
    for run, step in zip(runs, STEPS):
        items, origin = prepare_run(run, step)
        loaded.append(items)
        provenance.append(origin)
    ids = [item["id"] for item in loaded[0]]
    require(all([item["id"] for item in items] == ids for items in loaded), "All runs must have exactly the same ordered cases")
    groups = list(zip(*loaded))
    for group in groups:
        assert_matching(group)
    return groups, provenance


def input_fingerprints(groups, provenance):
    """Content hashes for input artifacts; huge checkpoints use recorded stat identity."""
    hashes = {str(Path(__file__).resolve()): digest(__file__), str(HELPER): digest(HELPER)}
    checkpoints = {}
    for origin in provenance:
        hashes[origin["batch_trace"]] = origin["batch_trace_sha256"]
        hashes[origin["selection_plan"]] = origin["selection_plan_sha256"]
    for group in groups:
        for item in group:
            trace = item["trace"]
            for path, value in (
                (item["trace_path"], item["trace_sha256"]), (item["prediction_path"], item["prediction_sha256"]),
                (item["source_path"], item["source_sha256"]),
                (item["metric_target_video_path"], item["metric_target_video_sha256"]),
                (item["dedup_index_path"], item["dedup_index_sha256"]),
                (trace["config"], trace["config_sha256"]), (trace["manifest"], trace["manifest_sha256"]),
                (trace["text_embedding_path"], trace["text_embedding_sha256"]),
            ):
                path = str(path)
                require(path not in hashes or hashes[path] == value, "Conflicting input fingerprints")
                hashes[path] = value
            checkpoints[trace["checkpoint"]] = (trace["checkpoint_bytes"], trace["checkpoint_mtime_ns"])
            frozen_review = item["trace_path"].parent / "review/decode_check.json"
            if frozen_review.is_file():
                hashes[str(frozen_review)] = digest(frozen_review)
    return hashes, checkpoints


def check_unchanged(hashes, checkpoints):
    for path, expected in hashes.items():
        require(digest(path) == expected, f"Input changed during comparison: {path}")
    for path, expected in checkpoints.items():
        stat = Path(path).stat()
        require((stat.st_size, stat.st_mtime_ns) == expected, f"Checkpoint identity changed: {path}")


def metric_one(prediction, target, common):
    import numpy as np
    pm, gm = prediction.max(axis=-1) > FOREGROUND_THRESHOLD, target.max(axis=-1) > FOREGROUND_THRESHOLD
    pixels = int(common.sum())
    difference = np.abs(prediction.astype(np.int16) - target.astype(np.int16))
    error = int(difference[common].sum(dtype=np.int64))
    intersection, union = int((pm & gm).sum()), int((pm | gm).sum())
    background = ~gm
    bg_pixels = int(background.sum())
    false_positive_pixels = int((pm & background).sum())
    bg_intensity = int(prediction[background].sum(dtype=np.int64))
    return {
        "common_foreground_mae": error / (pixels * 3 * 255) if pixels else 0.0,
        "common_foreground_pixels": pixels, "absolute_rgb_error_sum_uint8": error,
        "silhouette_iou": intersection / union if union else 1.0,
        "silhouette_intersection_pixels": intersection, "silhouette_union_pixels": union,
        "gt_background_predicted_foreground_rate": false_positive_pixels / bg_pixels if bg_pixels else 0.0,
        "gt_background_pixels": bg_pixels, "predicted_foreground_on_gt_background_pixels": false_positive_pixels,
        "mean_prediction_intensity_on_gt_background": bg_intensity / (bg_pixels * 3 * 255) if bg_pixels else 0.0,
    }


def measure(predictions, target):
    import numpy as np
    require(len(predictions) == 3 and all(x.shape == target.shape for x in predictions), "Metric shapes differ")
    common = np.logical_or.reduce([frame.max(axis=-1) > FOREGROUND_THRESHOLD for frame in (target, *predictions)])
    scores = {}
    for step, prediction in zip(STEPS, predictions):
        scores[f"step{step}"] = {
            **metric_one(prediction, target, common),
            "per_future_frame": [metric_one(prediction[i], target[i], common[i]) for i in range(16)],
        }
    return {"common_foreground_pixels": int(common.sum()),
            "common_foreground_pixels_per_frame": common.sum(axis=(1, 2)).tolist(), "models": scores}


def macro_average(cases):
    import numpy as np
    return {f"step{step}": {field: float(np.mean([case["models"][f"step{step}"][field] for case in cases]))
                             for field in SCALAR_METRICS} for step in STEPS}


def decode_group(group):
    reference = group[0]
    ordinals = reference["indices"]["future_source_indices"]
    w, h, fps = reference["width"], reference["height"], reference["fps"]
    rgb = base.read_source(reference["source_path"], ordinals, w, h)
    target = base.read_source(reference["metric_target_video_path"], ordinals, w, h)
    predictions, decode = [], []
    for item in group:
        frames, metadata = base.read_prediction(item["prediction_path"], w, h, fps)
        predictions.append(frames)
        decode.append(metadata)
    require(all(frames.shape == (16, h, w, 3) for frames in [rgb, target, *predictions]), "Decoded shape differs")
    return {"base": reference, "models": group, "rgb": rgb, "target": target,
            "predictions": predictions, "prediction_decode": decode}


def labels(object_reference=False):
    return ["Object-only GT" if object_reference else "Full-scene RGB GT", *[f"Step {step}: object prediction" for step in STEPS]]


def comparison_video(path, item):
    import av
    import numpy as np
    from PIL import Image, ImageDraw
    source = item["base"]
    w, h, fps = source["width"], source["height"], source["fps"]
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 4 * w, h + 76, "yuv420p"
        stream.options = {"crf": "18", "preset": "fast", "threads": "1"}
        for index in range(16):
            canvas = Image.new("RGB", (4 * w, h + 76), "#202936")
            draw = ImageDraw.Draw(canvas)
            for col, (label, frames) in enumerate(zip(labels(), [item["rgb"], *item["predictions"]])):
                draw.text((col * w + 8, 5), label, font=base.font(18), fill="white")
                canvas.paste(Image.fromarray(frames[index]), (col * w, 76))
            draw.text((10, 31), f"Future {index + 1}/16 | dedup {source['indices']['future_indices'][index]} | {base.time_label(source,index)}", font=base.font(17), fill="#d5dfef")
            draw.text((10, 54), f"Anchor {source['trace']['anchor_index']} | observer left / ego right in every panel | playback {fps} fps", font=base.font(14), fill="#d5dfef")
            frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")
            frame.pts, frame.time_base = index, 1 / fps
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def contact(item, object_reference=False):
    from PIL import Image, ImageDraw
    source = item["base"]
    w, h, top, gap = source["width"], source["height"], 76, 32
    canvas = Image.new("RGB", (4 * w, top + 4 * (h + gap)), "#202936")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f"{source['id']} | {source['trace']['episode']} | anchor {source['trace']['anchor_index']}", font=base.font(22), fill="white")
    for col, label in enumerate(labels(object_reference)):
        draw.text((col * w + 8, 43), label, font=base.font(18), fill="#d5dfef")
    reference = item["target"] if object_reference else item["rgb"]
    for row, index in enumerate(CHOSEN):
        y = top + row * (h + gap)
        draw.text((10, y + 4), f"Future {index + 1} | dedup {source['indices']['future_indices'][index]} | {base.time_label(source,index)}", font=base.font(18), fill="white")
        for col, frames in enumerate([reference, *item["predictions"]]):
            canvas.paste(Image.fromarray(frames[index]), (col * w, y + gap))
    return canvas


def overview(items):
    from PIL import Image, ImageDraw
    w, h, top, gap = items[0]["base"]["width"], items[0]["base"]["height"], 108, 38
    canvas = Image.new("RGB", (4 * w, top + len(items) * (h + gap)), "#202936")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 7), "Matched checkpoints: 4000 / 8000 / 10000", font=base.font(26), fill="white")
    draw.text((10, 42), "Future frame 16 only; inspect the videos and 1/5/10/16 contacts for motion and timing.", font=base.font(19), fill="#d5dfef")
    for col, label in enumerate(labels()):
        draw.text((col * w + 8, 76), label, font=base.font(18), fill="white")
    for row, item in enumerate(items):
        source = item["base"]
        y = top + row * (h + gap)
        draw.text((10, y + 5), f"{source['id']} | F16 {base.time_label(source,15)}", font=base.font(18), fill="white")
        for col, frames in enumerate([item["rgb"], *item["predictions"]]):
            canvas.paste(Image.fromarray(frames[-1]), (col * w, y + gap))
    return canvas


def metrics_document(cases):
    return {
        "schema": "vpp2_obj.three_checkpoint_pixel_metrics.v1", "status": "passed", "case_count": len(cases),
        "checkpoint_steps": list(STEPS), "metric_target": "object_only_gt", "foreground_threshold_uint8": FOREGROUND_THRESHOLD,
        "primary_metric": "case_macro_mean_of_common_foreground_mae",
        "common_mask_rule": "Per frame/pixel: maxRGB>16 for GT OR prediction4000 OR prediction8000 OR prediction10000. All three models use exactly the same RGB MAE denominator.",
        "mae_rule": "Within a case, sum absolute uint8 RGB error on the four-way mask over all16 frames; divide by(common mask pixels *3*255). Range0..1; lower is better.",
        "iou_rule": "Per model, foreground intersection(GT,prediction)/union(GT,prediction), pooled over16 frames. Higher is better. IoU deliberately uses each pair's conventional union, not the four-way MAE mask.",
        "gt_background_predicted_foreground_rate_rule": "Count(prediction maxRGB>16 AND GT maxRGB<=16)/count(GT maxRGB<=16), pooled over16 frames. Object position/timing errors can increase this rate; it is not exclusively background or distractor retention.",
        "background_intensity_rule": "Sum predicted RGB intensity over all GT-background pixels/(GT-background pixels *3*255); includes below-threshold intensity. Also affected by displaced foreground objects.",
        "aggregation_rule": "Arithmetic mean of per-case scores, giving each case equal weight; no all-background image MAE is reported as a primary score.",
        "empty_conventions": "Empty common mask gives MAE0; empty pair foreground union gives IoU1; no GT-background pixels gives background rate/intensity0 and a recorded denominator0.",
        "cross_report_warning": "These MAEs use GT plus THREE predictions. The old step2000/4000 report used a different common mask; MAE values are not directly comparable across those reports.",
        "limitations": "Fixed curated cases, not representative of the full test set. Pixel alignment is sensitive to timing, visibility, geometry and lossy MP4 encoding. These are not task-success rates.",
        "aggregate_case_macro_mean": macro_average(cases), "cases": cases,
    }


def write_csv(path, metrics):
    fields = ["row_type", "case_id", "task", "episode", "anchor", "checkpoint_step", *SCALAR_METRICS]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in metrics["cases"]:
            for step in STEPS:
                writer.writerow({"row_type": "case", "case_id": case["id"], "task": case["task"],
                                 "episode": case["episode"], "anchor": case["anchor"], "checkpoint_step": step,
                                 **{name: case["models"][f"step{step}"][name] for name in SCALAR_METRICS}})
        for step in STEPS:
            writer.writerow({"row_type": "case_macro_mean", "checkpoint_step": step,
                             **metrics["aggregate_case_macro_mean"][f"step{step}"]})


def page(items, output, metrics):
    esc = html.escape
    rows = []
    for step in STEPS:
        m = metrics["aggregate_case_macro_mean"][f"step{step}"]
        rows.append(f"<tr><td>{step}</td><td>{m['common_foreground_mae']:.6f}</td><td>{m['silhouette_iou']:.6f}</td><td>{m['gt_background_predicted_foreground_rate']:.6f}</td><td>{m['mean_prediction_intensity_on_gt_background']:.6f}</td></tr>")
    cards = []
    for item in items:
        source, trace = item["base"], item["base"]["trace"]
        folder = f"samples/{source['id']}"
        originals = " · ".join(f'<a href="{base.link(model["prediction_path"],output)}">原始 step{step} 预测</a>' for step, model in zip(STEPS, item["models"]))
        cards.append(f'''<article><h2>{esc(source['id'])}</h2><p>{esc(trace['task'])} / {esc(trace['episode'])} · anchor {trace['anchor_index']} · seed {trace['seed']} · {trace['num_inference_steps']} denoising steps</p>
<p><strong>Prompt：</strong>{esc(trace['prompt'])}</p><video controls loop playsinline preload="metadata" src="{folder}/comparison.mp4"></video>
<p>四列：全景 RGB GT | step4000 | step8000 | step10000。每格 observer 左、ego 右。</p><p>{originals} · <a href="{folder}/verification.json">配对与来源校验</a></p>
<details><summary>全景 RGB / 三个预测：未来1、5、10、16帧</summary><a href="{folder}/contact.png"><img loading="lazy" src="{folder}/contact.png" alt="Full-scene reference and three predictions at four future times"></a></details>
<details><summary>物体 GT / 三个预测：检查形状、运动和黑背景</summary><a href="{folder}/object_contact.png"><img loading="lazy" src="{folder}/object_contact.png" alt="Object-only GT and three predictions at four future times"></a></details></article>''')
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Checkpoint 4000 / 8000 / 10000 对照</title>
<style>body{{margin:0;background:#151b25;color:#eef3fc;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1800px;margin:auto;padding:24px}}a{{color:#a9ceff}}article,.note{{padding:20px;margin:24px 0;background:#253144;border-radius:10px}}video,img{{display:block;width:100%;height:auto}}video{{background:black}}p{{overflow-wrap:anywhere}}summary{{cursor:pointer}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border:1px solid #526174;text-align:left}}.table{{overflow:auto}}</style></head>
<body><main><h1>Checkpoint 4000 / 8000 / 10000：{len(items)} 个相同窗口</h1>
<p class="note">输入 episode、anchor、9帧条件、16帧未来采样、seed、prompt、embedding、配置和 dtype/内存策略均经核对一致，仅权重 checkpoint 不同。全景 RGB GT 只用于展示，所有指标均对黑背景物体 GT 计算。每个图像内 observer 在左、ego 在右；播放 FPS 不等于源仿真时间。</p>
<p class="note">这是固定精选窗口比较，不能代表完整测试集或作为动作成功率。主指标使用 GT 与三个预测的共同前景区域，按 case 等权平均。旧2000/4000两模型报告使用不同共同 mask，MAE 不可直接跨报告比较。GT背景上的预测前景率也会因物体位置/时序误差增加，不能直接等同背景残留。</p>
<div class="table"><table><thead><tr><th>Checkpoint</th><th>共同前景 MAE ↓</th><th>Mask IoU ↑</th><th>GT背景上的预测前景率</th><th>GT背景上的预测平均亮度</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p><a href="metrics.json">指标定义与逐帧数据 JSON</a> · <a href="metrics.csv">指标 CSV</a> · <a href="summary.json">审计摘要</a> · <a href="overview.png">末帧总览</a></p>
<p>总览只显示未来第16帧；动作顺序和时机请看视频与逐例四时刻图。</p><a href="overview.png"><img src="overview.png" alt="Future16 overview of four columns"></a>{''.join(cards)}</main></body></html>'''


def check_html_links(staging, output):
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.paths = []

        def handle_starttag(self, tag, attrs):
            self.paths.extend(value for key, value in attrs if key in ("href", "src"))

    links = Links()
    links.feed((staging / "index.html").read_text())
    for value in links.paths:
        uri = urlsplit(value)
        require(not uri.scheme, "Review should contain local links only")
        path = unquote(uri.path)
        require((staging / path).is_file() or (output / path).is_file(), f"Broken HTML link: {value}")
    return len(links.paths)


def render(runs, output):
    require(not output.exists() and not output.is_symlink(), f"Refusing to overwrite {output}")
    groups, provenance = prepare(runs)
    hashes, checkpoint_stats = input_fingerprints(groups, provenance)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        (staging / "samples").mkdir()
        items, case_metrics, reports = [], [], []
        for group in groups:
            item = decode_group(group)
            source, trace = item["base"], item["base"]["trace"]
            target = staging / "samples" / source["id"]
            target.mkdir()
            comparison_video(target / "comparison.mp4", item)
            _, comparison_decode = base.read_prediction(target / "comparison.mp4", 4 * source["width"], source["height"] + 76, source["fps"])
            contact(item).save(target / "contact.png")
            contact(item, object_reference=True).save(target / "object_contact.png")
            result = {"id": source["id"], "task": trace["task"], "episode": trace["episode"],
                      "anchor": trace["anchor_index"], **measure(item["predictions"], item["target"])}
            case_metrics.append(result)
            report = {
                "schema": "vpp2_obj.three_checkpoint_case_verification.v1", "status": "passed", "id": source["id"],
                "sampling": source["indices"], "invariant_trace_fields": list(TRACE_FIELDS),
                "optional_inference_fields_checked": list(OPTIONAL_MATH_FIELDS), "comparison_equal_except_checkpoint": True,
                "execution_note": "Checkpoint identity, output paths and measured runtime/memory statistics may differ; all listed model input and inference settings must match.",
                "matched_input": {field: trace[field] for field in TRACE_FIELDS},
                "full_scene_gt": {"path": str(source["source_path"]), "sha256": source["source_sha256"], "usage": "display only"},
                "object_gt": {"path": source["metric_target_video_path"], "sha256": source["metric_target_video_sha256"], "usage": "metrics and supplementary contact sheet"},
                "dedup_index": {"path": source["dedup_index_path"], "sha256": source["dedup_index_sha256"]},
                "checkpoint_identity": "Path, byte size and mtime_ns checked; full checkpoint weights were not loaded or hashed.",
                "predictions": [{"step": step, "checkpoint": model["trace"]["checkpoint"],
                                 "checkpoint_bytes": model["trace"]["checkpoint_bytes"],
                                 "checkpoint_mtime_ns": model["trace"]["checkpoint_mtime_ns"],
                                 "trace": str(model["trace_path"]), "trace_sha256": model["trace_sha256"],
                                 "prediction": str(model["prediction_path"]), "prediction_sha256": model["prediction_sha256"],
                                 "decode": decode} for step, model, decode in zip(STEPS, group, item["prediction_decode"])],
                "comparison_decode": comparison_decode, "contact_future_frames_one_based": [1, 5, 10, 16],
                "ground_truth_supplied_to_model": False, "gpu_used": False,
                "artifacts": {name: {"sha256": digest(target / name), "bytes": (target / name).stat().st_size}
                              for name in ("comparison.mp4", "contact.png", "object_contact.png")},
            }
            write_json(target / "verification.json", report)
            reports.append({"id": source["id"], "status": "passed", "verification": f"samples/{source['id']}/verification.json",
                            "verification_sha256": digest(target / "verification.json")})
            items.append(item)
            print(f"Compared and staged {source['id']}", flush=True)
        metrics = metrics_document(case_metrics)
        write_json(staging / "metrics.json", metrics)
        write_csv(staging / "metrics.csv", metrics)
        overview(items).save(staging / "overview.png")
        (staging / "index.html").write_text(page(items, output, metrics), encoding="utf-8")
        check_unchanged(hashes, checkpoint_stats)
        summary = {
            "schema": "vpp2_obj.three_checkpoint_comparison.v1", "status": "passed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "case_count": len(groups),
            "checkpoint_steps": list(STEPS), "runs": provenance, "output": str(output),
            "all_cases_have_matching_inputs_and_inference_settings": True, "checkpoint_only_model_change": True,
            "display_columns": labels(), "within_each_panel": "observer left, ego right",
            "metric_target": "object_only_gt", "common_mask_contains": ["object_gt", "step4000", "step8000", "step10000"],
            "aggregate_case_macro_mean": metrics["aggregate_case_macro_mean"],
            "ground_truth_usage": "Original fixed GT-based selection and post-inference display/metrics; no GT target frames passed to the three inference calls according to checked traces.",
            "limitations": metrics["limitations"], "cross_report_warning": metrics["cross_report_warning"],
            "gpu_used": False, "model_loaded": False, "original_predictions_modified": False,
            "input_content_sha256": hashes,
            "checkpoint_stat_identity": {path: {"bytes": values[0], "mtime_ns": values[1]} for path, values in checkpoint_stats.items()},
            "script": str(Path(__file__).resolve()), "script_sha256": digest(__file__),
            "source_review_helper": str(HELPER), "source_review_helper_sha256": digest(HELPER),
            "cases": reports,
            "artifacts": {name: {"sha256": digest(staging / name), "bytes": (staging / name).stat().st_size}
                          for name in ("index.html", "overview.png", "metrics.json", "metrics.csv")},
        }
        write_json(staging / "summary.json", summary)
        summary["html_local_links_checked"] = check_html_links(staging, output)
        write_json(staging / "summary.json", summary)
        require(not output.exists() and not output.is_symlink(), "Output appeared during rendering")
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps({"status": "passed", "cases": len(groups), "output": str(output),
                      "aggregate_case_macro_mean": metrics["aggregate_case_macro_mean"]}, indent=2))


def self_test():
    import copy
    import numpy as np
    gt = np.zeros((16, 32, 64, 3), dtype=np.uint8)
    gt[:, 8:16, 8:16] = 255
    correct, missing, extra = gt.copy(), np.zeros_like(gt), gt.copy()
    extra[:, 20:24, 40:44] = 255
    metrics = measure([correct, missing, extra], gt)
    m = metrics["models"]
    require(m["step4000"]["common_foreground_mae"] == 0 and m["step4000"]["silhouette_iou"] == 1, "Perfect prediction metrics failed")
    require(abs(m["step8000"]["common_foreground_mae"] - .8) < 1e-12
            and abs(m["step10000"]["common_foreground_mae"] - .2) < 1e-12, "Common-mask MAE denominator differs")
    require(m["step8000"]["silhouette_iou"] == 0 and abs(m["step10000"]["silhouette_iou"] - .8) < 1e-12, "Conventional pair IoU failed")
    require(abs(m["step10000"]["gt_background_predicted_foreground_rate"] - 16 / (32 * 64 - 64)) < 1e-12, "GT-background denominator failed")
    require(all(x["common_foreground_pixels"] == metrics["common_foreground_pixels"] for x in m.values()), "Models use different MAE masks")
    empty = measure([missing, missing, missing], missing)
    require(empty["models"]["step4000"]["common_foreground_mae"] == 0 and empty["models"]["step4000"]["silhouette_iou"] == 1, "Empty conventions failed")
    means = macro_average([metrics, empty])
    require(abs(means["step8000"]["common_foreground_mae"] - .4) < 1e-12, "Case macro average is not equal weighted")
    trace = {field: f"synthetic_{field}" for field in TRACE_FIELDS}
    trace.update(anchor_index=0, episode="synthetic", task="synthetic")
    source = {field: f"synthetic_{field}" for field in SOURCE_FIELDS}
    source.update(id="synthetic", width=64, height=32, fps=Fraction(6), trace=trace,
                  indices={"future_indices": list(range(5, 81, 5)), "future_timestamps_us": list(range(160000, 2560001, 160000)), "condition_timestamps_us": [0] * 9})
    same = [copy.deepcopy(source) for _ in STEPS]
    assert_matching(same)
    for field in ("seed", "sampling", "prompt", "context_mask", "memory_strategy", "config_sha256"):
        changed = copy.deepcopy(same)
        changed[1]["trace"][field] = "changed"
        try:
            assert_matching(changed)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Failed to reject changed {field}")
    colors = [(240, 20, 20), (20, 240, 20), (20, 20, 240), (240, 240, 240)]
    videos = [np.broadcast_to(np.asarray(color, dtype=np.uint8), gt.shape).copy() for color in colors]
    item = {"base": source, "rgb": videos[0], "target": gt, "predictions": videos[1:]}
    with tempfile.TemporaryDirectory(prefix="compare-checkpoints-cpu-test-") as directory:
        path = Path(directory) / "comparison.mp4"
        comparison_video(path, item)
        decoded, report = base.read_prediction(path, 4 * 64, 32 + 76, Fraction(6))
        for col, color in enumerate(colors):
            actual = decoded[:, 80:104, col * 64 + 8:col * 64 + 56].mean(axis=(0, 1, 2))
            require(np.max(np.abs(actual - color)) < 12, "Rendered video column order or content differs")
        require(report["frames"] == 16, "Encoded frame count mismatch")
        require(contact(item).size == contact(item, True).size == (256, 76 + 4 * 64), "Contact geometry mismatch")
    print(json.dumps({"status": "self_test_passed", "gpu_used": False,
                      "checks": ["common-mask MAE", "pair IoU", "GT-background denominator", "empty masks", "case macro mean",
                                 "reject altered seed/sampling/prompt/mask/dtype-strategy/config", "four video columns", "16frames6fps", "contact geometry"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", help="Repeat exactly three times: step4000, step8000, step10000")
    parser.add_argument("--output", type=Path, help="New comparison directory; existing outputs are never overwritten")
    parser.add_argument("--validate-only", action="store_true", help="Check sources, text cache and matching inputs on CPU; write nothing")
    parser.add_argument("--self-test", action="store_true", help="Small synthetic CPU validation; no run or output required")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    if args.self_test:
        self_test()
        return
    require(args.run is not None and len(args.run) == 3, "Provide exactly three --run arguments in checkpoint order")
    runs = [path.expanduser().resolve() for path in args.run]
    require(len(set(runs)) == 3 and all(path.is_dir() for path in runs), "Runs must be three distinct existing directories")
    if args.validate_only:
        groups, provenance = prepare(runs)
        print(json.dumps({"status": "validation_passed", "cases": len(groups), "runs": provenance,
                          "gpu_used": False, "outputs_written": False}, indent=2))
        return
    require(args.output is not None, "Provide --output for rendering")
    render(runs, args.output.expanduser().absolute())


if __name__ == "__main__":
    main()

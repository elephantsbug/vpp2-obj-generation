#!/usr/bin/env python3
"""Generate reviewable episode captions from composed PNGs and original tasks.

No video frames beyond frame zero, metadata arm labels, or task-name hints are
sent to the VLM. Dataset files are never edited. A full run requires --all.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
SUFFIX = "This is a composed robot video with the observer view on the left and the robot ego view on the right."


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(value, encoding="utf-8")
    temp.replace(path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def check_caption(raw):
    """Structural checks only. Passing these does not imply semantic correctness."""
    caption = " ".join(raw.split())
    errors = []
    if not raw or "\n" in raw.strip():
        errors.append("empty_or_multiline_output")
    if not caption.startswith("The right robot gripper "):
        errors.append("expected_actual_right_arm_first")
    if caption.count(", while the left robot gripper ") != 1:
        errors.append("expected_one_while_left_arm_clause")
    if not caption.endswith('. ' + SUFFIX) or caption.count(SUFFIX) != 1:
        errors.append("incorrect_view_suffix")
    if "@@" in caption or "```" in caption or "<think>" in caption:
        errors.append("unexpected_markup_or_path_separator")
    action = caption.removesuffix(SUFFIX).strip()
    clauses = action.rstrip('.').split(', while ')
    for clause in clauses:
        if 'keeps static' in clause and clause.lower() not in {
            'the right robot gripper keeps static', 'the left robot gripper keeps static'}:
            errors.append('static_clause_has_extra_actions')
    if "wrist" in action.lower() or "keeps stationary" in action.lower():
        errors.append("unexpected_wrist_view_or_static_wording")
    if re.search(r"\b(remains stationary|stays still|remains static|does not move)\b", action, re.I):
        errors.append("use_exact_keeps_static")
    if re.search(r"[^\x00-\x7f]", caption):
        errors.append("non_ascii_output_requires_review")
    return caption, errors


def select_rows(dataset, selection, all_rows):
    rows = [json.loads(x) for x in (dataset / "manifest.jsonl").read_text().splitlines() if x.strip()]
    index = {(r["task"], r["episode"]): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate task/episode in dataset manifest")
    if all_rows:
        selected = rows
    else:
        keys = json.loads(selection.read_text())
        if not keys or len(keys) > 20:
            raise ValueError("A preview must contain 1–20 episodes; full generation requires --all")
        pairs = [(r["task"], r["episode"]) for r in keys]
        if len(set(pairs)) != len(pairs):
            raise ValueError("Duplicate preview selection")
        selected = [index[key] for key in pairs]
    from PIL import Image
    for row in selected:
        if row.get("view_order") != ["observer", "ego"] or row["geometry"]["output_size_wh"] != [448, 224]:
            raise ValueError("This template requires observer-left + ego-right 448x224 data")
        path = dataset / row["outputs"]["image"]
        if not path.resolve().is_relative_to(dataset.resolve()):
            raise ValueError("Image path outside dataset")
        if sha256(path) != row["output_sha256"]["image"]:
            raise ValueError(f"Changed initial image: {path}")
        with Image.open(path) as im:
            if im.size != (448, 224) or im.mode != "RGB":
                raise ValueError(f"Unexpected image shape/mode: {path}")
        if not row["instruction"].strip() or "@@" in row["instruction"] or "\n" in row["instruction"]:
            raise ValueError("Task must be a nonempty single-line instruction without @@")
    return selected


def record_path(output, row):
    return output / "records" / row["task"] / f"{row['episode']}.json"


def worker(gpus, rows, config, dataset_str, output_str, template):
    # Set visibility before importing torch/transformers in this spawned process.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    dataset, output = Path(dataset_str), Path(output_str)
    log_path = output / f"worker_{gpus.replace(',', '_')}.log"
    with log_path.open("a", buffering=1) as log:
        sys.stdout = log
        sys.stderr = log
        print(f"Loading local model on physical GPUs {gpus}", flush=True)
        import torch
        import transformers
        from PIL import Image
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        torch.set_num_threads(4)
        torch.manual_seed(0)
        if torch.cuda.device_count() != 2:
            raise RuntimeError("Each replica requires exactly two visible GPUs")
        load_started = time.monotonic()
        processor = AutoProcessor.from_pretrained(config["model"], local_files_only=True,
            min_pixels=config['processor_pixels'], max_pixels=config['processor_pixels'])
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config["model"], local_files_only=True, dtype=torch.bfloat16,
            device_map="balanced", max_memory={0: "120GiB", 1: "120GiB"},
            attn_implementation="sdpa")
        model.eval()
        if any(str(value) in {"cpu", "disk"} for value in model.hf_device_map.values()):
            raise RuntimeError("Model unexpectedly offloaded to CPU/disk")
        runtime = {"torch": torch.__version__, "transformers": transformers.__version__,
                   "device_map": {k: str(v) for k, v in model.hf_device_map.items()},
                   "physical_gpus": gpus, "load_seconds": round(time.monotonic() - load_started, 3)}
        write_json(output / f"runtime_{gpus.replace(',', '_')}.json", runtime)
        print(f"Model loaded in {runtime['load_seconds']} seconds", flush=True)
        for row in rows:
            started = time.monotonic()
            prompt = template.replace("{task_instruction}", row["instruction"])
            image_path = dataset / row["outputs"]["image"]
            with Image.open(image_path) as im:
                initial_image = im.convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt}]}]
            chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[chat], images=[initial_image], padding=True, return_tensors="pt")
            grid = inputs["image_grid_thw"].tolist()
            inputs = inputs.to(model.device)
            input_tokens = inputs.input_ids.shape[1]
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=config["max_new_tokens"],
                    use_cache=True)
            new_tokens = generated[0, input_tokens:]
            raw = processor.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            caption, errors = check_caption(raw)
            eos_ids = model.generation_config.eos_token_id
            eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
            terminated = bool(len(new_tokens) and new_tokens[-1].item() in eos_ids)
            if not terminated:
                errors.append("generation_did_not_end_with_eos")
            record = {"task": row["task"], "episode": row["episode"],
                "image": row["outputs"]["image"], "image_sha256": sha256(image_path),
                "original_instruction": row["instruction"], "instruction_key": row["instruction_key"],
                "instruction_index": row["instruction_index"],
                "instruction_source": row["inputs"]["instruction"],
                "raw_response": raw, "enhanced_instruction": caption,
                "format_errors": errors, "semantic_review": "pending_human_review",
                "input_tokens": input_tokens, "generated_tokens": len(new_tokens),
                "image_grid_thw": grid, "elapsed_seconds": round(time.monotonic() - started, 3),
                "physical_gpus": gpus, "config_fingerprint": config["fingerprint"]}
            write_json(record_path(output, row), record)
            print(json.dumps({"task": row["task"], "episode": row["episode"],
                "seconds": record["elapsed_seconds"], "format_errors": errors,
                "caption": caption}, ensure_ascii=False), flush=True)
            del generated, inputs, new_tokens
    return len(rows)


def export_review(output, dataset, rows, config):
    results = [json.loads(record_path(output, row).read_text()) for row in rows]
    write_json(output / "results.json", results)
    atomic_text(output / "text_original.txt", "".join(
        f"{r['original_instruction']}@@{r['image']}\n" for r in results))
    valid = [r for r in results if not r["format_errors"]]
    atomic_text(output / "text_prompt_enhanced.txt", "".join(
        f"{r['enhanced_instruction']}@@{r['image']}\n" for r in valid))
    image_link = output / "images"
    if image_link.is_symlink():
        if image_link.resolve() != (dataset / "images").resolve():
            raise ValueError("Wrong images symlink in output")
    elif image_link.exists():
        raise ValueError("Output images path already exists")
    else:
        image_link.symlink_to(dataset / "images", target_is_directory=True)
    cards = []
    markdown = ["# VPP2 prompt 样本审核", "",
        "只输入双视角首帧和原任务。observer 画面左侧对应实际右臂；所有结果待人工语义审核。",
        "格式检查不能验证臂选择、目标或动作是否正确。", ""]
    for i, r in enumerate(results, 1):
        title = f"{i:02d}. {r['task']} / {r['episode']}"
        encoded = base64.b64encode((dataset / r["image"]).read_bytes()).decode("ascii")
        status = ", ".join(r["format_errors"]) or "格式通过；语义待审核"
        cards.append(f'<article><h2>{html.escape(title)}</h2><img width="672" height="336" '
            f'src="data:image/png;base64,{encoded}" alt="observer left, ego right">'
            '<p class="mapping">左面板：observer（画面左=实际右臂）；右面板：ego</p>'
            f'<h3>原任务</h3><p>{html.escape(r["original_instruction"])}</p>'
            f'<h3>Qwen 扩写</h3><p>{html.escape(r["enhanced_instruction"])}</p>'
            f'<p class="status">{html.escape(status)}</p></article>')
        markdown += [f"## {title}", "", f"![首帧]({r['image']})", "",
            f"原任务：{r['original_instruction']}", "", r["enhanced_instruction"], "",
            f"检查：{status}", ""]
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>VPP2 prompt 样本审核</title>
<style>body{font:17px/1.6 system-ui;max-width:1050px;margin:30px auto;padding:0 20px;background:#f4f5f7;color:#182332}
article{background:white;border:1px solid #dce0e6;border-radius:10px;padding:24px;margin:24px 0}img{max-width:100%;height:auto;image-rendering:auto}
h2{font-size:21px}h3{font-size:17px;margin-bottom:4px}.mapping,.status{color:#52627b;font-size:14px}</style>
<h1>VPP2 prompt 样本审核</h1><p>Qwen2.5-VL-72B-Instruct · BF16 · GPU 4–7 · 仅首帧 + 原任务</p>
<p>先描述实际右臂（observer 画面左），再描述实际左臂。以下均为模型原始生成结果，只做空白规范化；未人工改写。
请检查机械臂身份、动作先后、工具、目标、目的地与最终状态。全量扩写尚未运行。</p>'''
    atomic_text(output / "review.html", page + "\n".join(cards) + "</html>\n")
    atomic_text(output / "review.md", "\n".join(markdown))
    summary = {"selected": len(rows), "generated": len(results), "format_passed": len(valid),
        "format_failed": len(results) - len(valid), "semantic_review": "pending_human_review",
        "mode": config["mode"], "model": config["model"], "gpu_groups": config["gpu_groups"],
        "output_root": str(output), "dataset_root": str(dataset),
        "excluded_from_enhanced_file": [f"{r['task']}/{r['episode']}" for r in results if r['format_errors']]}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT / "data/robotwin_observer_ego_vpp2")
    parser.add_argument("--model", type=Path, default=Path("/data2/models/Qwen2.5-VL-72B-Instruct"))
    parser.add_argument("--template", type=Path, default=PROJECT / "prompts/observer_ego_enhance.txt")
    parser.add_argument("--selection", type=Path, default=PROJECT / "prompts/preview_selection.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT / "runs/prompt_preview_v2")
    parser.add_argument("--gpu-groups", default="4,5;6,7")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--vlm-image-scale", type=int, choices=[1, 2], default=2,
                        help="Internal VLM resizing only; source PNG stays 448x224")
    parser.add_argument("--all", action="store_true", help="Select all episodes; run only after sample review")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs; do not write or load model")
    args = parser.parse_args()
    dataset, output = args.dataset_root.resolve(), args.output_root.resolve()
    groups = args.gpu_groups.split(";")
    if sorted(groups) != ["4,5", "6,7"]:
        raise ValueError("This project run is restricted to GPU pairs 4,5 and 6,7")
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    if output == dataset or output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError("Use a separate run directory, outside the converted dataset")
    if args.all and output == (PROJECT / "runs/prompt_preview_v2").resolve():
        raise ValueError("Use a separate --output-root for a full run")
    template = args.template.read_text()
    if template.count("{task_instruction}") != 1 or SUFFIX not in template:
        raise ValueError("Invalid template placeholder or view suffix")
    rows = select_rows(dataset, args.selection, args.all)
    config = {"model": str(args.model.resolve()), "model_config_sha256": sha256(args.model / "config.json"),
        "model_index_sha256": sha256(args.model / "model.safetensors.index.json"),
        "template_sha256": sha256(args.template), "script_sha256": sha256(__file__),
        "dataset": str(dataset), "mode": "full" if args.all else "preview", "gpu_groups": groups,
        "dtype": "bfloat16", "device_map": "balanced", "attention": "sdpa", "do_sample": False,
        "max_new_tokens": args.max_new_tokens, "processor_pixels": 448 * 224 * args.vlm_image_scale ** 2,
        "vlm_image_scale": args.vlm_image_scale,
        "selection": [{"task": r["task"], "episode": r["episode"], "image": r["outputs"]["image"],
            "image_sha256": r["output_sha256"]["image"], "instruction": r["instruction"]} for r in rows]}
    config["fingerprint"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".generation.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_path = output / "run_config.json"
        if run_path.exists() and json.loads(run_path.read_text())["fingerprint"] != config["fingerprint"]:
            raise ValueError("Output belongs to a different configuration; choose a new run directory")
        write_json(run_path, config)
        atomic_text(output / "template.txt", template)
        pending = []
        for row in rows:
            path = record_path(output, row)
            if path.exists():
                prior = json.loads(path.read_text())
                if prior["config_fingerprint"] != config["fingerprint"]:
                    raise ValueError(f"Changed generation config: {path}")
            else:
                pending.append(row)
        print(f"Selected {len(rows)}, pending {len(pending)}; mode={config['mode']}", flush=True)
        if pending:
            with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
                futures = [pool.submit(worker, group, pending[i::2], config, str(dataset), str(output), template)
                           for i, group in enumerate(groups) if pending[i::2]]
                for future in as_completed(futures):
                    print(f"Replica completed {future.result()} episodes", flush=True)
        print(json.dumps(export_review(output, dataset, rows, config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

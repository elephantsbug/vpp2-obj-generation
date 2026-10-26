#!/usr/bin/env python3
"""Convert RoboTwin observer videos using VPP2's exact robot-view composition.

Encoded video cadence is preserved. Simulation timestamps remain in a separate,
byte-identical CSV; they are deliberately not used to resample the video.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import fcntl
from fractions import Fraction
import hashlib
import importlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

import av
import numpy as np
import PIL
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
ENCODING = {"codec": "libx264", "preset": "veryfast", "crf": 20,
            "pixel_format": "yuv420p", "threads": 1, "faststart": True}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_info(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def source_files(video):
    return {"video": video, "timestamps": video.with_suffix(".csv"),
            "instruction": video.parents[2] / "instruction" / (video.stem + ".json")}


def fingerprint(files, config):
    data = {"inputs": {key: file_info(path) for key, path in files.items()},
            "config": config}
    for key in ("timestamps", "instruction"):
        data["inputs"][key]["sha256"] = digest(files[key])
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(), data


def discover(source_root, tasks=None, limit=None):
    source_root = source_root.expanduser().resolve()
    if (source_root / "demo_clean").is_dir():
        source_root = source_root / "demo_clean"
    if not source_root.is_dir():
        raise ValueError(f"Source directory does not exist: {source_root}")
    videos = sorted(source_root.glob("*/aloha_agilex/observer/video/episode_*.mp4"))
    if tasks:
        unknown = set(tasks) - {video.parents[3].name for video in videos}
        if unknown:
            raise ValueError(f"Tasks without observer videos: {sorted(unknown)}")
        videos = [video for video in videos if video.parents[3].name in tasks]
    if limit is not None:
        videos = videos[:limit]
    if not videos:
        raise ValueError(f"No observer episode videos found under {source_root}")
    return source_root, videos


def read_annotations(files):
    with files["instruction"].open(encoding="utf-8") as handle:
        seen = json.load(handle).get("seen")
    if not isinstance(seen, list) or not seen or not isinstance(seen[0], str):
        raise ValueError(f"Expected nonempty seen[0] string: {files['instruction']}")
    instruction = " ".join(seen[0].split())
    if not instruction or "@@" in instruction:
        raise ValueError(f"Instruction cannot be empty or contain @@: {files['instruction']}")
    timestamps = []
    with files["timestamps"].open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"frame", "timestamp_s"} <= set(reader.fieldnames):
            raise ValueError(f"Expected frame,timestamp_s CSV: {files['timestamps']}")
        for index, row in enumerate(reader):
            timestamp = float(row["timestamp_s"])
            if int(row["frame"]) != index or not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError(f"Invalid timestamp/frame at CSV row {index + 2}")
            if timestamps and timestamp < timestamps[-1]:
                raise ValueError(f"Decreasing timestamp at CSV row {index + 2}")
            timestamps.append(timestamp)
    if not timestamps:
        raise ValueError("Timestamp CSV contains no frames")
    return instruction, timestamps


def probe_video(path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-threads", "1", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    return {"width": int(stream["width"]), "height": int(stream["height"]),
            "frame_count": int(stream["nb_read_frames"]),
            "fps": str(Fraction(stream["avg_frame_rate"]))}


def output_paths(output_root, task, episode):
    return {"video": output_root / "videos" / task / (episode + ".mp4"),
            "image": output_root / "images" / task / (episode + ".png"),
            "timestamps": output_root / "timestamps" / task / (episode + ".csv"),
            "record": output_root / "records" / task / (episode + ".json")}


def cached_record(paths, signature):
    try:
        record = json.loads(paths["record"].read_text())
        if record["fingerprint"] != signature:
            return None
        if probe_video(paths["video"]) != record["output_video"]:
            return None
        for key in ("video", "image", "timestamps"):
            if digest(paths[key]) != record["output_sha256"][key]:
                return None
        with Image.open(paths["image"]) as image:
            image.load()
            if image.size != (416, 240) or image.mode != "RGB":
                return None
        return record
    except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError):
        return None


def encode(video, destination, image_path, prepare_image):
    """Decode every source frame once and send official RGB composites to x264."""
    with av.open(str(video)) as container, tempfile.TemporaryFile() as error_log:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        rate = stream.average_rate
        if rate is None or rate <= 0:
            raise ValueError(f"Missing/invalid encoded frame rate: {video}")
        fps = Fraction(rate)
        source = {"width": stream.width, "height": stream.height, "fps": str(fps)}
        encoder = subprocess.Popen([
            "ffmpeg", "-v", "error", "-nostdin", "-y", "-filter_threads", "1", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s:v", "416x240", "-r", str(fps),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "1",
            "-movflags", "+faststart", "-f", "mp4", str(destination),
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=error_log)
        count, previous_time, geometry = 0, None, None
        try:
            for frame in container.decode(stream):
                if frame.pts is None or frame.time_base is None:
                    raise ValueError("Cannot verify source cadence: frame has no timestamp")
                current_time = Fraction(frame.pts) * frame.time_base
                if previous_time is not None:
                    # Allow one timestamp tick of quantization for fractional rates.
                    tolerance = min(0.25, abs(float(frame.time_base * fps)) + 1e-6)
                    delta = float((current_time - previous_time) * fps)
                    if delta <= 0 or abs(delta - 1) > tolerance:
                        raise ValueError("Variable frame rate input needs explicit timing conversion")
                previous_time = current_time
                rgb = frame.to_ndarray(format="rgb24")
                composed, geometry = prepare_image([Image.fromarray(rgb), None, None], "robot")
                if count == 0:
                    composed.save(image_path, format="PNG")
                encoder.stdin.write(np.asarray(composed, dtype=np.uint8).tobytes())
                count += 1
            encoder.stdin.close()
            code = encoder.wait()
            if code:
                error_log.seek(0)
                raise RuntimeError(f"ffmpeg failed ({code}): {error_log.read().decode(errors='replace')}")
            if not count:
                raise ValueError("Input video decoded no frames")
        except BaseException:
            if encoder.poll() is None:
                encoder.terminate()
            try:
                encoder.stdin.close()
            except (OSError, BrokenPipeError):
                pass
            encoder.wait()
            raise
    source["frame_count"] = count
    return source, geometry


def run_job(video, output_root, config, prepare_image, overwrite=False):
    files = source_files(video)
    signature, provenance = fingerprint(files, config)
    task, episode = video.parents[3].name, video.stem
    paths = output_paths(output_root, task, episode)
    if not overwrite:
        cached = cached_record(paths, signature)
        if cached is not None:
            return cached, "skipped"
    # A record is the commit marker; never leave an old one for a failed rebuild.
    paths["record"].unlink(missing_ok=True)
    instruction, timestamps = read_annotations(files)
    temporary = {}
    try:
        token = uuid.uuid4().hex
        for key, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary[key] = path.with_name(f".{path.name}.{token}.tmp")
        source, geometry = encode(video, temporary["video"], temporary["image"], prepare_image)
        if len(timestamps) != source["frame_count"]:
            raise ValueError(f"CSV has {len(timestamps)} rows, video has {source['frame_count']} frames")
        output = probe_video(temporary["video"])
        expected = {**source, "width": 416, "height": 240}
        if output != expected:
            raise ValueError(f"Encoded output verification failed: {output}, expected {expected}")
        shutil.copyfile(files["timestamps"], temporary["timestamps"])
        if fingerprint(files, config)[0] != signature:
            raise ValueError("Source files changed during conversion; rerun this episode")
        record = {
            "schema_version": 1, "task": task, "episode": episode,
            "fingerprint": signature, **provenance,
            "outputs": {key: str(path.relative_to(output_root)) for key, path in paths.items()},
            "source_video": source, "output_video": output,
            "first_frame_index": 0, "instruction": instruction,
            "instruction_key": "seen", "instruction_index": 0,
            "missing_views": [1, 2], "geometry": geometry,
            "timing": {"encoded_cadence_preserved": True, "resampled": False,
                       "simulation_timestamps_copied_verbatim": True,
                       "timestamp_rows": len(timestamps), "first_timestamp_s": timestamps[0],
                       "last_timestamp_s": timestamps[-1]},
            "output_sha256": {key: digest(temporary[key]) for key in ("video", "image", "timestamps")},
        }
        temporary["record"].write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        # Publish validated outputs before the final success record.
        for key in ("video", "image", "timestamps", "record"):
            os.replace(temporary[key], paths[key])
        return record, "converted"
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def atomic_text(path, text):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=Path("/data2/yaozhongbo/robotwin_clean_50*50/demo_clean"))
    parser.add_argument("--output-root", type=Path, default=PROJECT / "data/robotwin_observer_vpp2")
    parser.add_argument("--vpp2-root", type=Path, default=PROJECT / "third_party/vpp2")
    parser.add_argument("--workers", type=positive_int, default=16)
    parser.add_argument("--task", action="append", help="Exact task name; repeat to select several")
    parser.add_argument("--limit", type=positive_int, help="First N episodes in sorted selected tasks")
    parser.add_argument("--dry-run", action="store_true", help="List selected scope without writing files")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild even valid cached episodes")
    args = parser.parse_args()
    source_root, videos = discover(args.source_root, args.task, args.limit)
    output_root = args.output_root.expanduser().resolve()
    scope = {"source_root": str(source_root), "output_root": str(output_root),
             "tasks": sorted({v.parents[3].name for v in videos}), "task_filter": args.task,
             "limit": args.limit, "selected_episodes": len(videos),
             "manifest_scope": "successful episodes from this invocation only"}
    if args.dry_run:
        missing = [str(path) for video in videos for path in source_files(video).values()
                   if not path.is_file()]
        print(json.dumps({**scope, "dry_run": True, "missing_files": missing,
                          "examples": [str(v) for v in videos[:3]]}, ensure_ascii=False, indent=2))
        return int(bool(missing))
    for executable in ("ffmpeg", "ffprobe", "git"):
        if shutil.which(executable) is None:
            raise ValueError(f"Required executable not found: {executable}")
    vpp2_root = args.vpp2_root.expanduser().resolve()
    sys.path.insert(0, str(vpp2_root / "src"))
    conditioning = importlib.import_module("vpp2.utils.conditioning")
    expected_module = vpp2_root / "src/vpp2/utils/conditioning.py"
    if Path(conditioning.__file__).resolve() != expected_module:
        raise ValueError(f"Unexpected VPP2 import: {conditioning.__file__}")
    commit = subprocess.run(["git", "-C", str(vpp2_root), "rev-parse", "HEAD"],
                            check=True, text=True, capture_output=True).stdout.strip()
    config = {"official_commit": commit, "official_conditioning_sha256": digest(expected_module),
              "script_sha256": digest(__file__), "composition": "prepare_image([observer,None,None],'robot')",
              "runtime": {"python": sys.version.split()[0], "av": av.__version__,
                          "numpy": np.__version__, "Pillow": PIL.__version__},
              "encoding": ENCODING, "frame_selection": "all", "instruction_selection": "seen[0]"}
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".convert.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another convert process holds {output_root / '.convert.lock'}") from error
        records, failures = [], []
        counts = {"converted": 0, "skipped": 0, "failed": 0}
        started = time.monotonic()
        with ProcessPoolExecutor(max_workers=min(args.workers, len(videos)),
                                 mp_context=multiprocessing.get_context("spawn")) as executor:
            pending = {executor.submit(run_job, video, output_root, config,
                                       conditioning.prepare_image, args.overwrite): video for video in videos}
            for done, future in enumerate(as_completed(pending), 1):
                video = pending[future]
                try:
                    record, status = future.result()
                    records.append(record)
                except Exception as error:
                    status = "failed"
                    failures.append({"source_video": str(video), "error": str(error)})
                    print(f"ERROR {video}: {error}", file=sys.stderr, flush=True)
                counts[status] += 1
                print(f"[{done}/{len(videos)}] {status}: {video.parents[3].name}/{video.stem}", flush=True)
        records.sort(key=lambda record: (record["task"], record["episode"]))
        summary = {**scope, **counts, "elapsed_seconds": round(time.monotonic() - started, 3),
                   "failures": sorted(failures, key=lambda failure: failure["source_video"]), "config": config}
        atomic_text(output_root / "manifest.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        atomic_text(output_root / "text_original.txt", "".join(
            r["instruction"] + "@@" + r["outputs"]["image"] + "\n" for r in records))
        atomic_text(output_root / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return int(bool(failures))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

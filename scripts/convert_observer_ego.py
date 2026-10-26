#!/usr/bin/env python3
"""Convert paired RoboTwin videos to observer-left / ego-right, 448x224 RGB.

Each full source view is resized to 224x224 without cropping or flipping.
Every frame and the observer simulation-timestamp CSV are preserved.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
from fractions import Fraction
import importlib
from itertools import zip_longest
import json
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

import convert_observer as common


PROJECT = Path(__file__).resolve().parents[1]
LAYOUT = {"name": "observer_ego_horizontal", "view_order": ["observer", "ego"],
          "view_size_wh": [224, 224], "output_size_wh": [448, 224],
          "resize": "torchvision tensor BILINEAR antialias=True",
          "official_function": "vpp2.libero.eval_helpers.compose_libero_camera_views",
          "preprocess_mode": "training_exact", "uint8_conversion": "round_clip",
          "crop": False, "flip": False}
_compose_impl = None


def init_worker(vpp2_root):
    global _compose_impl
    root = Path(vpp2_root).resolve()
    sys.path.insert(0, str(root / "src"))
    import torch
    torch.set_num_threads(1)
    helpers = importlib.import_module("vpp2.libero.eval_helpers")
    if Path(helpers.__file__).resolve() != root / "src/vpp2/libero/eval_helpers.py":
        raise ValueError(f"Unexpected VPP2 import: {helpers.__file__}")
    _compose_impl = helpers.compose_libero_camera_views


def source_files(video):
    base = video.parents[2]
    ego = base / "ego/video" / video.name
    return {"observer_video": video, "ego_video": ego,
            "observer_timestamps": video.with_suffix(".csv"),
            "ego_timestamps": ego.with_suffix(".csv"),
            "instruction": base / "instruction" / (video.stem + ".json")}


def fingerprint(files, config):
    data = {"inputs": {key: common.file_info(path) for key, path in files.items()},
            "config": config}
    for key in ("observer_timestamps", "ego_timestamps", "instruction"):
        data["inputs"][key]["sha256"] = common.digest(files[key])
    import hashlib
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(), data


def compose_views(observer_rgb, ego_rgb):
    if _compose_impl is None:
        init_worker(PROJECT / "third_party/vpp2")
    # wrist_image is only the upstream API slot name; its actual content is ego.
    array = _compose_impl(
        {"image": observer_rgb, "wrist_image": ego_rgb}, concatenation="horizontal",
        image_shapes=[[3, 224, 224], [3, 224, 224]], num_cameras=2,
        preprocess_mode="training_exact")
    return Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8))


def cached_record(paths, signature):
    try:
        record = json.loads(paths["record"].read_text())
        if record["fingerprint"] != signature or record["geometry"] != LAYOUT:
            return None
        if common.probe_video(paths["video"]) != record["output_video"]:
            return None
        for key in ("video", "image", "timestamps"):
            if common.digest(paths[key]) != record["output_sha256"][key]:
                return None
        with Image.open(paths["image"]) as image:
            image.load()
            if image.size != (448, 224) or image.mode != "RGB":
                return None
        return record
    except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError):
        return None


def encode_pair(files, destination, image_path):
    """Decode paired frames in order; reject unequal or nonconstant cadence."""
    with av.open(str(files["observer_video"])) as observer, \
            av.open(str(files["ego_video"])) as ego, tempfile.TemporaryFile() as error_log:
        streams = [container.streams.video[0] for container in (observer, ego)]
        for stream in streams:
            stream.codec_context.thread_count = 1
            if stream.average_rate is None or stream.average_rate <= 0:
                raise ValueError("Missing/invalid encoded frame rate")
        rates = [Fraction(stream.average_rate) for stream in streams]
        if rates[0] != rates[1]:
            raise ValueError(f"Camera frame rates differ: {rates}")
        fps = rates[0]
        sources = {view: {"width": stream.width, "height": stream.height, "fps": str(fps)}
                   for view, stream in zip(LAYOUT["view_order"], streams)}
        encoder = subprocess.Popen([
            "ffmpeg", "-v", "error", "-nostdin", "-y", "-filter_threads", "1", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s:v", "448x224", "-r", str(fps), "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-threads", "1", "-movflags", "+faststart", "-f", "mp4", str(destination),
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=error_log)
        count, origins, previous = 0, [None, None], [None, None]
        try:
            pairs = zip_longest(observer.decode(streams[0]), ego.decode(streams[1]))
            for frames in pairs:
                if any(frame is None for frame in frames):
                    raise ValueError("Camera decoded frame counts differ")
                relative, ticks = [], []
                for camera, frame in enumerate(frames):
                    if frame.pts is None or frame.time_base is None:
                        raise ValueError("Cannot verify source cadence: frame has no timestamp")
                    current = Fraction(frame.pts) * frame.time_base
                    if origins[camera] is None:
                        origins[camera] = current
                    elapsed = current - origins[camera]
                    tolerance = min(0.25, float(frame.time_base * fps) + 1e-6)
                    if abs(float(elapsed * fps) - count) > tolerance:
                        raise ValueError(f"Camera {camera} has nonconstant frame cadence at frame {count}")
                    if previous[camera] is not None and current <= previous[camera]:
                        raise ValueError(f"Camera {camera} has unordered frame timestamps")
                    previous[camera] = current
                    relative.append(elapsed)
                    ticks.append(frame.time_base)
                alignment_tolerance = min(float(1 / (4 * fps)), float(max(ticks)) + 1e-9)
                if abs(float(relative[0] - relative[1])) > alignment_tolerance:
                    raise ValueError(f"Camera relative timestamps differ at frame {count}")
                image = compose_views(*(frame.to_ndarray(format="rgb24") for frame in frames))
                if count == 0:
                    image.save(image_path, format="PNG")
                encoder.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
                count += 1
            encoder.stdin.close()
            code = encoder.wait()
            if code:
                error_log.seek(0)
                raise RuntimeError(f"ffmpeg failed ({code}): {error_log.read().decode(errors='replace')}")
            if not count:
                raise ValueError("Input videos decoded no frames")
        except BaseException:
            if encoder.poll() is None:
                encoder.terminate()
            try:
                encoder.stdin.close()
            except (OSError, BrokenPipeError):
                pass
            encoder.wait()
            raise
    for source in sources.values():
        source["frame_count"] = count
    return sources


def run_job(video, output_root, config, overwrite=False):
    files = source_files(video)
    signature, provenance = fingerprint(files, config)
    task, episode = video.parents[3].name, video.stem
    paths = common.output_paths(output_root, task, episode)
    if not overwrite:
        cached = cached_record(paths, signature)
        if cached is not None:
            return cached, "skipped"
    paths["record"].unlink(missing_ok=True)
    instruction, timestamps = common.read_annotations({
        "instruction": files["instruction"], "timestamps": files["observer_timestamps"]})
    _, ego_timestamps = common.read_annotations({
        "instruction": files["instruction"], "timestamps": files["ego_timestamps"]})
    if timestamps != ego_timestamps:
        raise ValueError("Observer and ego CSV timestamps/frame counts differ")
    temporary = {}
    try:
        token = uuid.uuid4().hex
        for key, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary[key] = path.with_name(f".{path.name}.{token}.tmp")
        sources = encode_pair(files, temporary["video"], temporary["image"])
        count = sources["observer"]["frame_count"]
        if len(timestamps) != count:
            raise ValueError(f"CSVs have {len(timestamps)} rows, videos have {count} frames")
        output = common.probe_video(temporary["video"])
        expected = {"width": 448, "height": 224, "fps": sources["observer"]["fps"],
                    "frame_count": count}
        if output != expected:
            raise ValueError(f"Encoded output verification failed: {output}, expected {expected}")
        shutil.copyfile(files["observer_timestamps"], temporary["timestamps"])
        if fingerprint(files, config)[0] != signature:
            raise ValueError("Source files changed during conversion; rerun this episode")
        record = {
            "schema_version": 2, "task": task, "episode": episode,
            "fingerprint": signature, **provenance,
            "outputs": {key: str(path.relative_to(output_root)) for key, path in paths.items()},
            "source_videos": sources, "output_video": output,
            "view_order": LAYOUT["view_order"], "view_size_wh": LAYOUT["view_size_wh"],
            "geometry": LAYOUT, "first_frame_index": 0, "instruction": instruction,
            "instruction_key": "seen", "instruction_index": 0,
            "timing": {"encoded_cadence_preserved": True, "resampled": False,
                       "both_camera_relative_pts_verified": True,
                       "both_camera_csv_timestamps_equal": True,
                       "simulation_timestamps_copied_verbatim": True,
                       "timestamp_rows": len(timestamps), "first_timestamp_s": timestamps[0],
                       "last_timestamp_s": timestamps[-1]},
            "output_sha256": {key: common.digest(temporary[key])
                              for key in ("video", "image", "timestamps")},
        }
        temporary["record"].write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        for key in ("video", "image", "timestamps", "record"):
            os.replace(temporary[key], paths[key])
        return record, "converted"
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def guard_layout(output_root):
    """Keep this conversion separate from the existing 416x240 dataset."""
    marker = output_root / ".layout.json"
    if marker.exists():
        if json.loads(marker.read_text()) != LAYOUT:
            raise ValueError(f"Output directory belongs to a different layout: {output_root}")
    elif any(path.name != ".convert.lock" for path in output_root.iterdir()):
        raise ValueError(f"Refusing nonempty output directory without this layout marker: {output_root}")
    else:
        common.atomic_text(marker, json.dumps(LAYOUT, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=Path("/data2/yaozhongbo/robotwin_clean_50*50/demo_clean"))
    parser.add_argument("--output-root", type=Path, default=PROJECT / "data/robotwin_observer_ego_vpp2")
    parser.add_argument("--vpp2-root", type=Path, default=PROJECT / "third_party/vpp2",
                        help="Official repository supplying the training-exact composition")
    parser.add_argument("--workers", type=common.positive_int, default=16)
    parser.add_argument("--task", action="append", help="Exact task name; repeat to select several")
    parser.add_argument("--limit", type=common.positive_int, help="First N episodes in sorted selected tasks")
    parser.add_argument("--dry-run", action="store_true", help="Check selected source paths without writing")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild even valid cached episodes")
    args = parser.parse_args()
    source_root, videos = common.discover(args.source_root, args.task, args.limit)
    output_root = args.output_root.expanduser().resolve()
    scope = {"source_root": str(source_root), "output_root": str(output_root),
             "tasks": sorted({video.parents[3].name for video in videos}), "task_filter": args.task,
             "limit": args.limit, "selected_episodes": len(videos),
             "manifest_scope": "successful episodes from this invocation only"}
    missing = [str(path) for video in videos for path in source_files(video).values()
               if not path.is_file()]
    if args.dry_run:
        print(json.dumps({**scope, "dry_run": True, "missing_files": missing, "geometry": LAYOUT,
                          "examples": [str(video) for video in videos[:3]]}, ensure_ascii=False, indent=2))
        return int(bool(missing))
    if missing:
        raise ValueError(f"Missing {len(missing)} input files; first: {missing[0]}")
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise ValueError(f"Required executable not found: {executable}")
    vpp2_root = args.vpp2_root.expanduser().resolve()
    init_worker(vpp2_root)
    import torch
    import torchvision
    config = {"script_sha256": common.digest(__file__), "helper_sha256": common.digest(common.__file__),
              "official_helper_sha256": common.digest(vpp2_root / "src/vpp2/libero/eval_helpers.py"),
              "composition": LAYOUT, "encoding": common.ENCODING,
              "runtime": {"python": sys.version.split()[0], "av": av.__version__,
                          "numpy": np.__version__, "Pillow": PIL.__version__,
                          "torch": torch.__version__, "torchvision": torchvision.__version__},
              "frame_selection": "all", "instruction_selection": "seen[0]"}
    config["official_commit"] = subprocess.run(
        ["git", "-C", str(vpp2_root), "rev-parse", "HEAD"],
        check=True, text=True, capture_output=True).stdout.strip()
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".convert.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another convert process holds {output_root / '.convert.lock'}") from error
        guard_layout(output_root)
        records, failures = [], []
        counts = {"converted": 0, "skipped": 0, "failed": 0}
        started = time.monotonic()
        with ProcessPoolExecutor(max_workers=min(args.workers, len(videos)),
                                 mp_context=multiprocessing.get_context("spawn"),
                                 initializer=init_worker, initargs=(str(vpp2_root),)) as executor:
            pending = {executor.submit(run_job, video, output_root, config, args.overwrite): video
                       for video in videos}
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
                   "total_frames": sum(record["output_video"]["frame_count"] for record in records),
                   "failures": sorted(failures, key=lambda failure: failure["source_video"]), "config": config}
        common.atomic_text(output_root / "manifest.jsonl", "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in records))
        common.atomic_text(output_root / "text_original.txt", "".join(
            record["instruction"] + "@@" + record["outputs"]["image"] + "\n" for record in records))
        common.atomic_text(output_root / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return int(bool(failures))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

#!/usr/bin/env python3
"""Index the first frame at each simulation timestamp without rewriting media.

Dataset consumers sample/pad in deduplicated frame order, then map through
source_frame_indices before decoding the original RGB or object videos.
Positive short intervals are retained; this does not create a fixed-rate grid.
"""

import argparse
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import tempfile

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
POLICY = "equal_timestamp_keep_first_v1"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_timestamps(data):
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    if not rows:
        raise ValueError("Empty timestamp CSV")
    if [int(row["frame"]) for row in rows] != list(range(len(rows))):
        raise ValueError("CSV frame IDs must be consecutive from zero")
    values = []
    for row in rows:
        value = Decimal(row["timestamp_s"]) * 1_000_000
        if not value.is_finite() or value != value.to_integral_value():
            raise ValueError("Expected finite timestamps with microsecond precision")
        values.append(int(value))
    timestamps = np.asarray(values, dtype=np.int64)
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("Timestamp order decreases; do not sort video frames")
    return timestamps


def first_frame_indices(timestamps):
    keep = np.r_[True, np.diff(timestamps) > 0]
    return np.flatnonzero(keep), np.flatnonzero(~keep)


def prepare_episode(record, dataset_root):
    task, episode = record["task"], record["episode"]
    for value in (task, episode):
        if Path(value).name != value or value in (".", ".."):
            raise ValueError("Task and episode must be plain names")
    csv_path = dataset_root / record["outputs"]["timestamps"]
    data = csv_path.read_bytes()
    digest = sha256(data)
    if digest != record["output_sha256"]["timestamps"]:
        raise ValueError(f"Timestamp CSV changed: {csv_path}")
    timestamps = read_timestamps(data)
    source_count = len(timestamps)
    if source_count != int(record["output_video"]["frame_count"]):
        raise ValueError(f"CSV/converted video metadata count mismatch: {csv_path}")
    video_path = dataset_root / record["outputs"]["video"]
    image_path = dataset_root / record["outputs"]["image"]
    if not video_path.is_file() or not image_path.is_file():
        raise FileNotFoundError(f"Missing converted video or first image: {task}/{episode}")

    # Verify the original RGB and both available object variants use the same
    # timestamp/frame-index table. No particular object variant is selected here.
    target_sources = {}
    for camera in ("observer", "ego"):
        rgb = Path(record["inputs"][f"{camera}_video"]["path"])
        if not rgb.is_file() or rgb.with_suffix(".csv").read_bytes() != data:
            raise ValueError(f"Original RGB timestamp mismatch: {rgb}")
        if int(record["source_videos"][camera]["frame_count"]) != source_count:
            raise ValueError(f"Original RGB frame count metadata mismatch: {rgb}")
        for variant in ("video_object", "video_object_full"):
            target = rgb.parent.parent / variant / rgb.name
            if not target.is_file() or target.with_suffix(".csv").read_bytes() != data:
                raise ValueError(f"Object video/timestamp mismatch: {target}")
            target_sources.setdefault(variant, {})[camera] = str(target)

    indices, dropped = first_frame_indices(timestamps)
    entry = dict(
        task=task, episode=episode, policy=POLICY,
        index_path=f"indices/{task}/{episode}.npz",
        sampling_index_space="deduplicated_frame_order",
        video_index_space="original_frame_order",
        condition_video_path=str(video_path),
        first_image_path=str(image_path),
        source_timestamps_path=str(csv_path),
        source_timestamps_sha256=digest,
        source_frame_count=source_count,
        frame_count=len(indices), dropped_frame_count=len(dropped),
        source_encoded_fps=record["output_video"]["fps"],
        uniform_simulation_cadence=False,
        target_sources=target_sources,
        object_timestamp_tables_verified=True,
        object_video_decode_counts_verified=False,
    )
    return entry, indices, timestamps[indices], dropped


def build(dataset_root, output_root, dry_run=False):
    dataset_root, output_root = dataset_root.resolve(), output_root.resolve()
    if output_root.exists() and not dry_run:
        raise FileExistsError(f"Choose a new output directory: {output_root}")
    manifest_path = dataset_root / "manifest.jsonl"
    manifest_bytes = manifest_path.read_bytes()
    source_records = [json.loads(line) for line in manifest_bytes.splitlines() if line]
    keys = [(row["task"], row["episode"]) for row in source_records]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Manifest must contain nonempty, unique episodes")

    summary = dict(
        schema_version=1, policy=POLICY, source_dataset=str(dataset_root),
        source_manifest_sha256=sha256(manifest_bytes),
        builder_sha256=sha256(Path(__file__).read_bytes()),
        output_root=str(output_root), episodes=len(source_records),
        tasks=len({task for task, _ in keys}),
        frames_before=0, frames_after=0, removed_duplicate_frames=0,
        positive_short_intervals_retained=0,
        resampled=False, source_media_modified=False,
        object_timestamp_tables_verified=0,
        object_video_decode_counts_verified=False,
        training_dataset_integration="pending",
    )
    task_counts = Counter()
    if not dry_run:
        output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}.building-",
        dir=None if dry_run else output_root.parent,
    ) as temporary:
        stage = Path(temporary)
        entries = []
        for record in sorted(source_records, key=lambda r: (r["task"], r["episode"])):
            entry, indices, timestamps, dropped = prepare_episode(record, dataset_root)
            summary["frames_before"] += entry["source_frame_count"]
            summary["frames_after"] += entry["frame_count"]
            summary["removed_duplicate_frames"] += len(dropped)
            delta = np.diff(timestamps)
            summary["positive_short_intervals_retained"] += int(
                np.count_nonzero((delta > 0) & (delta < 32_000))
            )
            summary["object_timestamp_tables_verified"] += 4
            task_counts[entry["task"]] += 1
            if not dry_run:
                path = stage / entry["index_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    path, source_frame_indices=indices, timestamps_us=timestamps,
                    dropped_source_frame_indices=dropped,
                    source_frame_count=np.int64(entry["source_frame_count"]),
                )
                entry["index_sha256"] = sha256(path.read_bytes())
            entries.append(entry)
        summary["episodes_per_task"] = dict(sorted(task_counts.items()))
        summary["removed_frame_percent"] = (
            100 * summary["removed_duplicate_frames"] / summary["frames_before"]
        )
        if not dry_run:
            (stage / "manifest.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in entries)
            )
            (stage / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
            )
            if output_root.exists():
                raise FileExistsError(output_root)
            stage.rename(output_root)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path,
                        default=PROJECT / "data/robotwin_observer_ego_vpp2")
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT / "data/robotwin_observer_ego_dedup_index_v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.dataset_root, args.output_root, args.dry_run),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

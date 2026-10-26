"""Integration tests for synchronized observer-left / ego-right conversion."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import av
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
VPP2 = Path(os.environ.get("VPP2_ROOT", PROJECT / "third_party/vpp2"))
sys.path.insert(0, str(VPP2 / "src"))
from vpp2.libero.eval_helpers import compose_libero_camera_views


class ConvertObserverEgoIntegration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vpp2-two-view-test-")
        self.root = Path(self.temp.name)
        self.source = self.root / "demo_clean"
        self.output = self.root / "converted"
        self.base = self.source / "sample/aloha_agilex"

    def tearDown(self):
        self.temp.cleanup()

    def write_view(self, view, count=4, fps=30, times=None):
        width, height = ((480, 240) if view == "observer" else (240, 320))
        path = self.base / view / "video/episode_0000000.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        frames = []
        for index in range(count):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = np.arange(width, dtype=np.uint16)[None, :] % 256
            frame[:, :, 1] = np.arange(height, dtype=np.uint16)[:, None] % 256
            # Distinct temporal signals distinguish camera order and frame alignment.
            frame[:, :, 2] = 20 + index * 15 if view == "observer" else 170 + index * 15
            frames.append(frame)
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-filter_threads", "1", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "pipe:0", "-an", "-c:v", "libx264rgb", "-crf", "0",
            "-preset", "ultrafast", "-threads", "1", str(path),
        ], input=b"".join(f.tobytes() for f in frames), check=True, capture_output=True)
        if times is None:
            times = [0, .004, .036, .068][:count]
        with path.with_suffix(".csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame", "timestamp_s"])
            writer.writerows(enumerate(times))
        instruction = self.base / "instruction/episode_0000000.json"
        instruction.parent.mkdir(exist_ok=True)
        instruction.write_text(json.dumps({"seen": ["Move the block onto the pad."]}))
        return path

    def run_convert(self, *extra):
        return subprocess.run([
            sys.executable, str(PROJECT / "scripts/convert_observer_ego.py"),
            "--source-root", str(self.source), "--output-root", str(self.output),
            "--vpp2-root", str(VPP2), "--workers", "1", *extra,
        ], capture_output=True, text=True, timeout=120)

    @staticmethod
    def decode(path):
        with av.open(str(path)) as container:
            return [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]

    def assert_failed_without_record(self, result):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((self.output / "records/sample/episode_0000000.json").exists())
        manifest = self.output / "manifest.jsonl"
        if manifest.exists():
            self.assertFalse(manifest.read_text().strip())

    def test_direct_resize_order_timing_and_resume(self):
        observer = self.write_view("observer")
        ego = self.write_view("ego")
        result = self.run_convert()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        first_o, first_e = self.decode(observer)[0], self.decode(ego)[0]
        official = compose_libero_camera_views(
            {"image": first_o, "wrist_image": first_e}, concatenation="horizontal",
            image_shapes=[[3, 224, 224], [3, 224, 224]], num_cameras=2,
            preprocess_mode="training_exact",
        )
        expected = np.clip(np.rint(official), 0, 255).astype(np.uint8)
        with Image.open(self.output / "images/sample/episode_0000000.png") as image:
            self.assertEqual(image.size, (448, 224))
            np.testing.assert_array_equal(np.asarray(image), expected)
        video = self.output / "videos/sample/episode_0000000.mp4"
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            self.assertEqual((stream.width, stream.height, stream.average_rate), (448, 224, 30))
            frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        self.assertEqual(len(frames), 4)
        for index, frame in enumerate(frames):
            self.assertLess(abs(frame[60:160, 60:160, 2].mean() - (20 + index * 15)), 12)
            self.assertLess(abs(frame[60:160, 284:384, 2].mean() - (170 + index * 15)), 12)
        self.assertEqual((self.output / "timestamps/sample/episode_0000000.csv").read_bytes(),
                         observer.with_suffix(".csv").read_bytes())
        original_time = video.stat().st_mtime_ns
        resumed = self.run_convert()
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        self.assertEqual(video.stat().st_mtime_ns, original_time)

    def test_missing_ego_dry_run_creates_no_output(self):
        self.write_view("observer")
        result = self.run_convert("--dry-run")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_misaligned_timestamps_are_rejected(self):
        self.write_view("observer")
        self.write_view("ego", times=[0, .004, .040, .068])
        self.assert_failed_without_record(self.run_convert())

    def test_shorter_ego_is_not_silently_truncated(self):
        self.write_view("observer")
        self.write_view("ego", count=3, times=[0, .004, .036, .068])
        self.assert_failed_without_record(self.run_convert())

    def test_different_camera_fps_is_rejected(self):
        self.write_view("observer")
        self.write_view("ego", fps=25)
        self.assert_failed_without_record(self.run_convert())


if __name__ == "__main__":
    unittest.main()

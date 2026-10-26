"""CPU integration checks: geometry, temporal alignment, resume and failure handling."""

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
from vpp2.utils.conditioning import prepare_image


class ConvertObserverIntegration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vpp2-observer-test-")
        self.root = Path(self.temp.name)
        self.source = self.root / "demo_clean"
        self.output = self.root / "prepared"

    def tearDown(self):
        self.temp.cleanup()

    def make_episode(self, task, size=(320, 240), csv_frames=4):
        width, height = size
        base = self.source / task / "aloha_agilex"
        video = base / "observer/video/episode_0000000.mp4"
        video.parent.mkdir(parents=True)
        instruction = base / "instruction/episode_0000000.json"
        instruction.parent.mkdir()
        instruction.write_text(json.dumps({"seen": ["Move the red block to the blue pad."],
                                           "unseen": ["Unused alternative."]}))
        # Vary space and time to detect cropping, channel and frame-order mistakes.
        frames = []
        for index in range(4):
            array = np.zeros((height, width, 3), dtype=np.uint8)
            array[:, :, 0] = np.arange(width, dtype=np.uint16)[None, :] % 256
            array[:, :, 1] = np.arange(height, dtype=np.uint16)[:, None] % 256
            array[:, :, 2] = 30 + index * 50
            frames.append(array)
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", "30", "-i", "pipe:0", "-an",
            "-c:v", "libx264rgb", "-crf", "0", "-preset", "ultrafast",
            "-threads", "1", str(video),
        ], input=b"".join(frame.tobytes() for frame in frames), check=True,
            capture_output=True)
        with video.with_suffix(".csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame", "timestamp_s"])
            for index, timestamp in enumerate([0, .004, .036, .068][:csv_frames]):
                writer.writerow([index, f"{timestamp:.6f}"])
        return video

    def run_prepare(self, *extra):
        return subprocess.run([
            sys.executable, str(PROJECT / "scripts/convert_observer.py"),
            "--source-root", str(self.source), "--output-root", str(self.output),
            "--vpp2-root", str(VPP2), "--workers", "2", *extra,
        ], text=True, capture_output=True, timeout=90)

    def test_geometry_frame_order_timestamps_and_resume(self):
        sources = [self.make_episode(task, size) for task, size in [
            ("wide", (480, 240)), ("square", (240, 240)), ("portrait", (240, 320)),
        ]]
        result = self.run_prepare()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        saved_times = {}
        for source in sources:
            task = source.parents[3].name
            video = self.output / "videos" / task / source.name
            first_png = self.output / "images" / task / f"{source.stem}.png"
            copied_csv = self.output / "timestamps" / task / f"{source.stem}.csv"
            with av.open(str(source)) as container:
                originals = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
            expected, _ = prepare_image([Image.fromarray(originals[0]), None, None], "robot")
            with Image.open(first_png) as first:
                np.testing.assert_array_equal(np.asarray(first), np.asarray(expected))
            self.assertEqual(copied_csv.read_bytes(), source.with_suffix(".csv").read_bytes())
            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                self.assertEqual((stream.width, stream.height), (416, 240))
                self.assertEqual(stream.average_rate, 30)
                decoded = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
            self.assertEqual(len(decoded), len(originals))
            for index, frame in enumerate(decoded):
                # H.264 is lossy: check distant black panels and broad color regions.
                self.assertLessEqual(int(frame[:, 320:, :].max()), 5)
                self.assertLess(abs(float(frame[80:160, 80:160, 2].mean()) -
                                    (30 + index * 50)), 12)
            saved_times[video] = video.stat().st_mtime_ns
        self.assertEqual(len((self.output / "manifest.jsonl").read_text().splitlines()), 3)
        self.assertEqual(len((self.output / "text_original.txt").read_text().splitlines()), 3)
        resumed = self.run_prepare()
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        self.assertEqual(saved_times, {p: p.stat().st_mtime_ns for p in saved_times})

    def test_dry_run_does_not_create_output(self):
        self.make_episode("dry")
        result = self.run_prepare("--dry-run")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.output.exists())

    def test_bad_timestamps_do_not_publish_success(self):
        self.make_episode("bad", csv_frames=3)
        result = self.run_prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.output / "records/bad/episode_0000000.json").exists())
        manifest = self.output / "manifest.jsonl"
        if manifest.exists():
            self.assertFalse(manifest.read_text().strip())


if __name__ == "__main__":
    unittest.main()

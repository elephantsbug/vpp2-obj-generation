"""Integration checks for RGB-conditioned, object-video targets."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import av
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
import convert_observer_ego as pair
import convert_observer_ego_obj_full as obj


class ObjectTargetConversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pair.init_worker(PROJECT / 'third_party/vpp2')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='vpp2-obj-pair-test-')
        self.root = Path(self.temp.name)
        self.base = self.root / 'source/task/aloha_agilex'
        self.rgb_root, self.obj_root = self.root / 'rgb', self.root / 'obj'
        self.config = {'test': 'rgb-obj-pair-v1'}
        for view in ('observer', 'ego'):
            for kind in ('video', 'video_object_full'):
                self.write_source(view, kind)
        self.rgb_source = self.base / 'observer/video/episode_0000000.mp4'
        self.rgb_record, _ = pair.run_job(self.rgb_source, self.rgb_root, self.config)

    def tearDown(self):
        self.temp.cleanup()

    def write_source(self, view, kind, count=4):
        path = self.base / view / kind / 'episode_0000000.mp4'
        path.parent.mkdir(parents=True, exist_ok=True)
        frames = []
        for i in range(count):
            a = np.full((48, 64, 3), 180 if kind == 'video' else 0, np.uint8)
            a[8:30, (4 if view == 'observer' else 34):(24 if view == 'observer' else 54)] = [200, 20 + 20*i, 40]
            frames.append(a)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                        '-s', '64x48', '-r', '30', '-i', 'pipe:0', '-an', '-c:v', 'libx264rgb',
                        '-crf', '0', '-preset', 'ultrafast', '-threads', '1', str(path)],
                       input=b''.join(a.tobytes() for a in frames), check=True, capture_output=True)
        self.write_times(path.with_suffix('.csv'), [0, .004, .036, .068])
        instruction = self.base / 'instruction/episode_0000000.json'
        instruction.parent.mkdir(exist_ok=True)
        if not instruction.exists():
            instruction.write_text(json.dumps({'seen': ['Move the red object.']}))
        return path

    @staticmethod
    def write_times(path, times):
        with path.open('w', newline='') as f:
            w = csv.writer(f); w.writerow(['frame', 'timestamp_s']); w.writerows(enumerate(times))

    def run_obj(self):
        return obj.run_job(self.rgb_source, self.obj_root, self.rgb_root, self.config)

    def test_geometry_pair_manifest_and_resume(self):
        record, status = self.run_obj()
        self.assertEqual(status, 'converted')
        files = obj.source_files(self.rgb_source, self.rgb_root)
        frames = []
        for key in ('observer_video', 'ego_video'):
            with av.open(str(files[key])) as c:
                frames.append(next(c.decode(video=0)).to_ndarray(format='rgb24'))
        expected = np.asarray(pair.compose_views(*frames))
        with Image.open(self.obj_root / record['outputs']['image']) as image:
            np.testing.assert_array_equal(np.asarray(image), expected)
        self.assertEqual(record['output_video'], self.rgb_record['output_video'])
        self.assertTrue(record['timing']['rgb_obj_csv_timestamps_equal'])
        before = (self.obj_root / record['outputs']['video']).stat().st_mtime_ns
        _, status = self.run_obj()
        self.assertEqual(status, 'skipped')
        self.assertEqual((self.obj_root / record['outputs']['video']).stat().st_mtime_ns, before)
        prompt = {'task': 'task', 'episode': 'episode_0000000',
                  'original_instruction': record['instruction'], 'enhanced_instruction': 'Test caption.',
                  'review_flags': ['test-review-flag'],
                  'video_sha256': self.rgb_record['output_sha256']['video'],
                  'image_sha256': self.rgb_record['output_sha256']['image']}
        prompt_file = self.root / 'prompts.jsonl'; prompt_file.write_text(json.dumps(prompt) + '\n')
        rows = obj.paired_rows([record], self.obj_root, self.rgb_root, prompt_file)
        self.assertEqual(rows[0]['condition_video_path'], str(files['rgb_video']))
        self.assertEqual(rows[0]['target_video_path'], str(self.obj_root / record['outputs']['video']))
        self.assertEqual(rows[0]['prompt_review_flags'], ['test-review-flag'])
        prompt['video_sha256'] = 'different'; prompt_file.write_text(json.dumps(prompt) + '\n')
        with self.assertRaisesRegex(ValueError, 'different RGB media'):
            obj.paired_rows([record], self.obj_root, self.rgb_root, prompt_file)

    def test_cross_domain_timestamp_offset_rejected(self):
        for view in ('observer', 'ego'):
            self.write_times(self.base / view / 'video_object_full/episode_0000000.csv', [0, .004, .040, .068])
        with self.assertRaisesRegex(ValueError, 'OBJ and RGB simulation timestamps'):
            self.run_obj()
        self.assertFalse((self.obj_root / 'records/task/episode_0000000.json').exists())

    def test_obj_shorter_than_rgb_is_rejected(self):
        for view in ('observer', 'ego'):
            self.write_source(view, 'video_object_full', count=3)
        with self.assertRaisesRegex(ValueError, 'videos have 3 frames'):
            self.run_obj()
        self.assertFalse((self.obj_root / 'records/task/episode_0000000.json').exists())

    def test_changed_rgb_output_rejected(self):
        rgb_video = self.rgb_root / self.rgb_record['outputs']['video']
        with rgb_video.open('ab') as f: f.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'RGB reference output hash changed'):
            self.run_obj()

    def test_output_domain_guard(self):
        with self.assertRaisesRegex(ValueError, 'separate'):
            obj.guard_output(self.rgb_root, self.rgb_root, self.root / 'source')
        obj.guard_output(self.obj_root, self.rgb_root, self.root / 'source')
        marker = self.obj_root / '.content.json'
        marker.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'different content'):
            obj.guard_output(self.obj_root, self.rgb_root, self.root / 'source')


if __name__ == '__main__':
    unittest.main()

"""CPU checks for paired source-frame alignment, padding and deterministic sampling."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from vpp2_obj.dataset import PairedObjectVideoDataset


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MetadataDataset(PairedObjectVideoDataset):
    """Exercise real sampling through workers without decoding media."""

    def __getitem__(self, index):
        return self.sample_window(index)


def metadata_collate(rows):
    # Ordinary NumPy IPC avoids torch's Unix-socket FD sharer in restricted CI.
    return np.asarray(rows, dtype=np.int64)


class PairedObjectVideoDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vpp2-object-dataset-")
        self.root = Path(self.temp.name)
        self.rows = []

    def tearDown(self):
        self.temp.cleanup()

    def add_episode(self, task="a", length=100, prompt="Move the block."):
        index = len(self.rows)
        episode = f"episode_{index:07d}"
        source = np.arange(length, dtype=np.int64) * 2
        path = self.root / f"{episode}.npz"
        np.savez(path, source_frame_indices=source,
                 timestamps_us=np.arange(length, dtype=np.int64) * 32000,
                 source_frame_count=np.array(int(source[-1]) + 1, dtype=np.int64))
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        embedding = self.root / f"{digest}.pt"
        # Padding context deliberately nonzero, to detect accidental masking.
        context = torch.arange(128, dtype=torch.bfloat16)[:, None].expand(128, 4096).clone()
        payload = {
            "schema": "vpp2.frozen_wan_text_embedding_cache.v1",
            "context": context, "mask": torch.arange(128) < 7,
            "input_ids": torch.cat((torch.arange(7), torch.zeros(121, dtype=torch.int64))),
            "prompt": prompt, "prompt_sha256": digest, "prompt_hash": digest,
            "cache_signature": "fixture-signature", "valid_token_count": 7,
        }
        if not embedding.exists():
            torch.save(payload, embedding)
        row = {
            "task": task, "episode": episode, "prompt": prompt, "prompt_review_flags": [],
            "sampling_index_space": "deduplicated_frame_order",
            "dedup_index_path": str(path), "dedup_index_sha256": sha(path),
            "source_frame_count": int(source[-1]) + 1, "deduplicated_frame_count": length,
            "condition_video_path": str(self.root / f"rgb_{episode}.mp4"),
            "target_video_path": str(self.root / f"obj_{episode}.mp4"),
            "video_metadata": {"height": 4, "width": 8, "frame_count": int(source[-1]) + 1},
            "text_embedding_path": str(embedding), "text_embedding_sha256": sha(embedding),
            "text_embedding_prompt_sha256": digest,
            "text_embedding_cache_signature": "fixture-signature",
            "text_embedding_context_length": 128, "text_embedding_valid_tokens": 7,
        }
        self.rows.append(row)
        return row

    def manifest(self):
        path = self.root / "manifest.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        return path

    def dataset(self, cls=PairedObjectVideoDataset, **kwargs):
        return cls(self.manifest(), video_size=(4, 8), **kwargs)

    def test_negative_history_clamps_before_source_mapping_and_boundaries(self):
        self.add_episode(length=300)
        dataset = self.dataset()
        window = dataset.window_indices(0, 35)
        self.assertEqual(window["condition_indices"].tolist(), [0, 0, 0, 0, 0, 0, 0, 5, 35])
        self.assertEqual(window["condition_source_indices"].tolist(), [0, 0, 0, 0, 0, 0, 0, 10, 70])
        self.assertEqual(window["condition_image_is_pad"].tolist(), [False, True, True, True, True, True, True, False, False])
        self.assertEqual(window["future_indices"].tolist(), list(range(40, 116, 5)))
        self.assertEqual(window["future_source_indices"].tolist(), list(range(80, 231, 10)))
        self.assertEqual(window["image_is_pad"].shape, (25,))
        self.assertFalse(window["image_is_pad"][9:].any())
        dataset.window_indices(0, 219)  # Final legal anchor: 219 + 80 = 299.
        with self.assertRaisesRegex(ValueError, "Incomplete future"):
            dataset.window_indices(0, 220)
        with self.assertRaises(IndexError):
            dataset.window_indices(0, -1)
        with self.assertRaises(IndexError):
            dataset.window_indices(-1, 0)
        first = dataset.window_indices(0, 0)
        self.assertEqual(first["condition_source_indices"].tolist(), [0] * 9)
        self.assertEqual(first["condition_image_is_pad"].tolist(), [False] + [True] * 7 + [False])

    def write_video(self, path, count, base):
        frames = [np.full((4, 8, 3), base + index, dtype=np.uint8) for index in range(count)]
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-filter_threads", "1", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", "8x4", "-r", "30", "-i", "pipe:0",
            "-an", "-c:v", "libx264rgb", "-crf", "0", "-preset", "ultrafast",
            "-threads", "1", str(path),
        ], input=b"".join(frame.tobytes() for frame in frames), capture_output=True, check=True)

    def test_actual_two_video_decode_normalization_and_real_mask(self):
        row = self.add_episode(length=10)
        self.write_video(row["condition_video_path"], row["source_frame_count"], 10)
        self.write_video(row["target_video_path"], row["source_frame_count"], 150)
        dataset = self.dataset(history_stride=2, history_frames=3, prediction_stride=2, prediction_frames=2)
        sample = dataset.get_window(0, 3)
        # episode0, history[-1,1,3] -> source[0,0,2,6]; future[5,7] -> source[10,14].
        expected = torch.tensor([10, 10, 12, 16, 160, 164], dtype=torch.float32) / 127.5 - 1
        torch.testing.assert_close(sample["video"][0, :, 0, 0], expected)
        self.assertEqual(sample["video"].shape, (3, 6, 4, 8))
        torch.testing.assert_close(sample["condition_video"], sample["video"][:, :4])
        self.assertEqual(sample["image_is_pad"].tolist(), [False, True, False, False, False, False])
        self.assertEqual(sample["context_mask"].tolist(), [True] * 7 + [False] * 121)
        self.assertEqual(float(sample["context"][127, 0]), 127)
        self.assertNotIn("condition_observation_video", sample)
        self.assertNotIn("action", sample)
        # Returned tensors cannot mutate the resident cached embedding.
        sample["context"].zero_()
        sample["context_mask"].zero_()
        context, mask = dataset._load_embedding(row)
        self.assertEqual(float(context[127, 0]), 127)
        self.assertEqual(int(mask.sum()), 7)

    def test_default_shape_and_inference_never_reads_target(self):
        row = self.add_episode(length=100)
        dataset = self.dataset()
        calls = []

        def decode(path, indices):
            calls.append((str(path), indices))
            value = -0.5 if str(path) == row["condition_video_path"] else 0.75
            return torch.full((3, len(indices), 4, 8), value)

        with patch.object(dataset, "_decode_frames", side_effect=decode):
            sample = dataset.get_window(0, 19)
            self.assertEqual(sample["video"].shape, (3, 25, 4, 8))
            self.assertTrue(torch.all(sample["video"][:, :9] == -0.5))
            self.assertTrue(torch.all(sample["video"][:, 9:] == 0.75))
            self.assertEqual(len(calls), 2)
            calls.clear()
            inference = dataset.get_window(0, 99, include_target=False)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], row["condition_video_path"])
            self.assertNotIn("video", inference)
            self.assertNotIn("future_source_indices", inference)
            self.assertEqual(inference["future_indices"][-1], 179)
            self.assertEqual(inference["condition_indices"][-1], 99)
        with patch.object(dataset, "_decode_frames", side_effect=AssertionError("must not decode")), \
                patch.object(dataset, "_load_embedding", side_effect=AssertionError("must not load embeddings")):
            dataset.window_indices(0, 99, include_target=False)

    def test_task_balance_window_weighting_epoch_and_resume_determinism(self):
        self.add_episode(task="a", length=81)  # one legal window
        self.add_episode(task="a", length=179)  # 99 legal windows
        self.add_episode(task="b", length=100)
        dataset = self.dataset(samples_per_epoch=6000, seed=7)
        first = [dataset.sample_window(i) for i in range(len(dataset))]
        tasks = Counter(dataset.records[episode]["task"] for episode, _ in first)
        self.assertEqual(tasks, {"a": 3000, "b": 3000})
        # This rejects episode-uniform sampling (which would give ~1500 short samples).
        short_count = sum(episode == 0 for episode, _ in first)
        self.assertTrue(10 < short_count < 60, short_count)
        for index in (5999, 35, 0, 412):
            self.assertEqual(dataset.sample_window(index), first[index])
        dataset.set_epoch(1)
        second = [dataset.sample_window(i) for i in range(30)]
        self.assertNotEqual(first[:30], second)
        duplicate = self.dataset(samples_per_epoch=6000, seed=7)
        duplicate.set_epoch(1)
        self.assertEqual(second, [duplicate.sample_window(i) for i in range(30)])
        dataset.set_epoch(0)
        self.assertEqual(first[:30], [dataset.sample_window(i) for i in range(30)])
        self.assertEqual(len(self.dataset()), 120)
        evaluation = self.dataset(training=False)
        self.assertEqual(len(evaluation), 3)
        expected = [(0, 0), (1, 49), (2, 9)]
        self.assertEqual([evaluation.sample_window(i) for i in range(3)], expected)
        evaluation.set_epoch(10)
        self.assertEqual([evaluation.sample_window(i) for i in range(3)], expected)

    def test_dataloader_workers_match_single_process_across_epochs(self):
        self.add_episode(task="a", length=100)
        self.add_episode(task="b", length=120)
        dataset = self.dataset(cls=MetadataDataset, samples_per_epoch=12)
        for epoch in (0, 3):
            dataset.set_epoch(epoch)
            expected = np.asarray([dataset[i] for i in range(len(dataset))])
            try:
                actual = np.concatenate(list(DataLoader(
                    dataset, batch_size=3, num_workers=2, persistent_workers=False,
                    collate_fn=metadata_collate, timeout=15,
                )))
            except PermissionError as error:
                self.skipTest(f"Worker shared memory unavailable in sandbox: {error}")
            np.testing.assert_array_equal(actual, expected)

    def test_embedding_hash_identity_signature_shape_and_lru(self):
        for index in range(3):
            self.add_episode(prompt=f"Move object {index}.")
        dataset = self.dataset(embedding_cache_size=2)
        for row in self.rows:
            dataset._load_embedding(row)
        self.assertEqual(len(dataset._embedding_cache), 2)
        self.assertNotIn(self.rows[0]["text_embedding_path"], [key[0] for key in dataset._embedding_cache])
        row = self.rows[0]
        path = Path(row["text_embedding_path"])
        original = torch.load(path, weights_only=True)
        for field, replacement, message in [
            ("prompt", "Other prompt", "prompt identity"),
            ("cache_signature", "wrong", "signature"),
            ("context", torch.zeros(127, 4096, dtype=torch.bfloat16), "context"),
            ("mask", torch.ones(128, dtype=torch.bool), "token count"),
            ("context", torch.full((128, 4096), float("nan"), dtype=torch.bfloat16), "nonfinite"),
        ]:
            with self.subTest(field=field, message=message):
                changed = dict(original, **{field: replacement})
                torch.save(changed, path)
                with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                    self.dataset()._load_embedding(row)
                with self.assertRaisesRegex(ValueError, message):
                    self.dataset(verify_hashes=False)._load_embedding(row)
        torch.save(original, path)

    def test_manifest_and_dedup_validation_and_short_inference_episode(self):
        row = self.add_episode(length=100)
        row["prompt_review_flags"] = ["needs_review"]
        with self.assertRaisesRegex(ValueError, "requires review"):
            self.dataset()
        row["prompt_review_flags"] = []
        row["text_embedding_prompt_sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "prompt SHA"):
            self.dataset()
        row["text_embedding_prompt_sha256"] = hashlib.sha256(row["prompt"].encode()).hexdigest()
        row["dedup_index_sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "Dedup index SHA"):
            self.dataset()
        row["dedup_index_sha256"] = sha(row["dedup_index_path"])
        row["deduplicated_frame_count"] += 1
        with self.assertRaisesRegex(ValueError, "frame counts"):
            self.dataset()
        row["deduplicated_frame_count"] -= 1
        self.add_episode(task="short", length=80)
        dataset = self.dataset()
        self.assertEqual(dataset.total_windows, 20)
        self.assertEqual(dataset.episode_length(1), 80)
        dataset.window_indices(1, 79, include_target=False)
        with self.assertRaisesRegex(ValueError, "Incomplete future"):
            dataset.window_indices(1, 0)
        self.rows = [self.rows[1]]
        inference_only = self.dataset(training=False)
        self.assertEqual(len(inference_only), 0)
        inference_only.window_indices(0, 79, include_target=False)
        with self.assertRaisesRegex(ValueError, "No episodes"):
            self.dataset()


if __name__ == "__main__":
    unittest.main()

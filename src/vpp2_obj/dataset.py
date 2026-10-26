"""Paired RGB-history / object-future windows in deduplicated frame order.

Video files retain their original frames. Only the NPZ index determines which
source frames to decode; neither encoded FPS nor timestamps resample the video.
Set ``persistent_workers=False`` on the DataLoader, and call ``set_epoch`` before
creating each training iterator so new workers receive the current epoch.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import operator
from pathlib import Path

import av
import numpy as np
import torch
from torch.utils.data import Dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value, name: str) -> int:
    value = operator.index(value)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


class PairedObjectVideoDataset(Dataset):
    """Load nine RGB conditions followed by sixteen object-only target frames.

    Training indices independently seed their own RNG. With task balancing,
    tasks rotate in an epoch-specific permutation; within a task every legal
    anchor has equal probability, including anchors requiring history padding.
    An index therefore gives the same sample regardless of worker scheduling or
    the sampler's resume offset. Evaluation returns one fixed midpoint window
    per episode by default and ignores epoch changes.

    ``verify_hashes`` checks the small NPZ and embedding artifacts, never media.
    Embeddings are validated lazily and retained in a bounded per-worker LRU.
    """

    def __init__(
        self,
        manifest_path,
        history_stride=30,
        history_frames=8,
        prediction_stride=5,
        prediction_frames=16,
        video_size=(224, 448),
        balance_tasks=True,
        seed=100008,
        samples_per_epoch=None,
        verify_hashes=True,
        training=True,
        embedding_cache_size=8,
    ):
        self.manifest_path = Path(manifest_path).resolve()
        self.history_stride = _positive_int(history_stride, "history_stride")
        self.history_frames = _positive_int(history_frames, "history_frames")
        self.prediction_stride = _positive_int(prediction_stride, "prediction_stride")
        self.prediction_frames = _positive_int(prediction_frames, "prediction_frames")
        if len(video_size) != 2:
            raise ValueError("video_size must be (height, width)")
        self.video_size = tuple(_positive_int(v, "video_size") for v in video_size)
        self.balance_tasks = bool(balance_tasks)
        self.training = bool(training)
        self.seed = operator.index(seed)
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        self.epoch = 0
        self.verify_hashes = bool(verify_hashes)
        self.embedding_cache_size = _positive_int(embedding_cache_size, "embedding_cache_size")
        self._embedding_cache = OrderedDict()
        self._source_indices = []
        self._timestamps_us = []
        self.records = []
        keys = set()
        with self.manifest_path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row["task"], row["episode"])
                if key in keys:
                    raise ValueError(f"Duplicate episode {key} at manifest line {line_number}")
                keys.add(key)
                self._validate_record(row)
                source, timestamps = self._load_dedup_index(row)
                self.records.append(row)
                self._source_indices.append(source)
                self._timestamps_us.append(timestamps)
        if not self.records:
            raise ValueError("Manifest has no episodes")
        self.future_horizon = self.prediction_stride * self.prediction_frames
        self._window_counts = np.asarray(
            [max(0, len(source) - self.future_horizon) for source in self._source_indices],
            dtype=np.int64,
        )
        self._eligible_episodes = np.flatnonzero(self._window_counts)
        if not len(self._eligible_episodes) and self.training:
            raise ValueError("No episodes have a complete future window")
        self._all_cumulative = np.cumsum(self._window_counts[self._eligible_episodes])
        self.total_windows = int(self._all_cumulative[-1]) if len(self._all_cumulative) else 0
        self._task_episodes = {}
        for index in self._eligible_episodes:
            self._task_episodes.setdefault(self.records[index]["task"], []).append(int(index))
        self.tasks = sorted(self._task_episodes)
        self._task_cumulative = {
            task: np.cumsum(self._window_counts[indices])
            for task, indices in self._task_episodes.items()
        }
        default_length = self.total_windows if self.training else len(self._eligible_episodes)
        self.samples_per_epoch = (
            default_length if samples_per_epoch is None
            else _positive_int(samples_per_epoch, "samples_per_epoch")
        )
        if not len(self._eligible_episodes) and samples_per_epoch is not None:
            raise ValueError("samples_per_epoch requires at least one complete future window")
        self._task_order = np.arange(len(self.tasks))
        self.set_epoch(0)

    def _path(self, value) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.manifest_path.parent / path

    def _validate_record(self, row):
        if row.get("prompt_review_flags"):
            raise ValueError(f"Prompt requires review: {row['task']}/{row['episode']}")
        if row.get("sampling_index_space") != "deduplicated_frame_order":
            raise ValueError("Manifest must use deduplicated_frame_order")
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError("Prompt must be a nonempty string")
        if row.get("text_embedding_cache_must_be_rebuilt", False):
            raise ValueError("Manifest marks the text embedding cache as stale")
        digest = hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest()
        if row.get("text_embedding_prompt_sha256") != digest:
            raise ValueError("Manifest embedding prompt SHA does not match prompt")
        if not row.get("text_embedding_cache_signature"):
            raise ValueError("Missing text embedding cache signature")
        if row.get("text_embedding_context_length") != 128:
            raise ValueError("Expected a 128-token embedding context")
        for name in ("dedup_index_path", "text_embedding_path", "condition_video_path", "target_video_path"):
            if not row.get(name):
                raise ValueError(f"Missing {name}")
        if self._path(row["condition_video_path"]).resolve() == self._path(row["target_video_path"]).resolve():
            raise ValueError("RGB condition and object target must use different video paths")
        metadata = row.get("video_metadata", {})
        if (metadata.get("height"), metadata.get("width")) != self.video_size:
            raise ValueError(f"Manifest resolution must be {self.video_size}; resizing is disabled")

    def _load_dedup_index(self, row):
        path = self._path(row["dedup_index_path"])
        if self.verify_hashes and _sha256(path) != row.get("dedup_index_sha256"):
            raise ValueError(f"Dedup index SHA mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            source = data["source_frame_indices"].copy()
            timestamps = data["timestamps_us"].copy()
            source_count = int(data["source_frame_count"])
        if source.dtype != np.int64 or source.ndim != 1 or not len(source):
            raise ValueError(f"Invalid source_frame_indices: {path}")
        if source[0] != 0 or np.any(np.diff(source) <= 0) or source[-1] >= source_count:
            raise ValueError(f"Source indices must start at zero and strictly increase within the video: {path}")
        if timestamps.dtype != np.int64 or timestamps.shape != source.shape or np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"Invalid deduplicated timestamps: {path}")
        if source_count != row.get("source_frame_count") or len(source) != row.get("deduplicated_frame_count"):
            raise ValueError(f"Dedup frame counts differ from manifest: {path}")
        if row["video_metadata"].get("frame_count") != source_count:
            raise ValueError(f"Source video frame count differs from dedup index: {path}")
        return source, timestamps

    def __len__(self):
        return self.samples_per_epoch

    def set_epoch(self, epoch):
        """Call before a new iterator; persistent DataLoader workers are unsupported."""
        epoch = operator.index(epoch)
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = epoch if self.training else 0
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 0x5441534B]))
        self._task_order = rng.permutation(len(self.tasks))

    def episode_length(self, episode_index):
        episode_index = operator.index(episode_index)
        if not 0 <= episode_index < len(self.records):
            raise IndexError(f"episode_index out of range: {episode_index}")
        return len(self._source_indices[episode_index])

    def sample_window(self, index):
        """Return (manifest episode index, deduplicated anchor), without video I/O."""
        index = operator.index(index)
        if not 0 <= index < len(self):
            raise IndexError(f"sample index out of range: {index}")
        if not self.training:
            episode = int(self._eligible_episodes[index % len(self._eligible_episodes)])
            return episode, int((self._window_counts[episode] - 1) // 2)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        if self.balance_tasks:
            task = self.tasks[int(self._task_order[index % len(self.tasks)])]
            episodes = self._task_episodes[task]
            cumulative = self._task_cumulative[task]
        else:
            episodes, cumulative = self._eligible_episodes, self._all_cumulative
        position = int(rng.integers(int(cumulative[-1])))
        slot = int(np.searchsorted(cumulative, position, side="right"))
        previous = int(cumulative[slot - 1]) if slot else 0
        return int(episodes[slot]), position - previous

    def window_indices(self, episode_index, anchor_index, include_target=True):
        """Describe a window without decoding video or loading text embeddings.

        In inference mode, future logical indices may extend beyond the recorded
        episode. They have no corresponding source indices or timestamps yet.
        """
        length = self.episode_length(episode_index)
        episode_index, anchor_index = operator.index(episode_index), operator.index(anchor_index)
        if not 0 <= anchor_index < length:
            raise IndexError(f"anchor_index must be in [0, {length - 1}]")
        if include_target and anchor_index + self.future_horizon >= length:
            raise ValueError(f"Incomplete future: anchor {anchor_index} + {self.future_horizon} >= {length}")
        row = self.records[episode_index]
        requested = anchor_index + np.arange(-(self.history_frames - 1), 1) * self.history_stride
        # Clamp logical positions BEFORE indexing the raw-video frame map.
        condition = np.concatenate(([0], np.maximum(requested, 0))).astype(np.int64)
        pad = np.concatenate(([False], requested < 0))
        future = anchor_index + np.arange(1, self.prediction_frames + 1) * self.prediction_stride
        source = self._source_indices[episode_index]
        timestamps = self._timestamps_us[episode_index]
        result = {
            "task": row["task"], "episode": row["episode"],
            "episode_index": episode_index, "anchor_index": anchor_index,
            "deduplicated_length": length,
            "condition_indices": torch.from_numpy(condition),
            "condition_source_indices": torch.from_numpy(source[condition].copy()),
            "condition_timestamps_us": torch.from_numpy(timestamps[condition].copy()),
            "condition_history_offsets": torch.from_numpy(np.concatenate(([-anchor_index], requested - anchor_index)).astype(np.int64)),
            "condition_image_is_pad": torch.from_numpy(pad),
            "future_indices": torch.from_numpy(future.astype(np.int64)),
            "image_is_pad": torch.from_numpy(np.concatenate((pad, np.zeros(self.prediction_frames, dtype=bool))) if include_target else pad.copy()),
        }
        if include_target:
            result["future_source_indices"] = torch.from_numpy(source[future].copy())
            result["future_timestamps_us"] = torch.from_numpy(timestamps[future].copy())
        return result

    def _decode_frames(self, path, indices):
        """Decode exact raw frame ordinals, preserving repeated requested frames."""
        ordinals = [operator.index(index) for index in indices]
        wanted = set(ordinals)
        frames = {}
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError(f"Video has no video stream: {path}")
            stream = container.streams.video[0]
            stream.thread_count = 1
            if (stream.height, stream.width) != self.video_size:
                raise ValueError(f"Video resolution differs from {self.video_size}: {path}")
            for ordinal, frame in enumerate(container.decode(stream)):
                if ordinal in wanted:
                    array = frame.to_ndarray(format="rgb24")
                    if array.shape != (*self.video_size, 3):
                        raise ValueError(f"Decoded frame resolution changed: {path}")
                    frames[ordinal] = array
                    if len(frames) == len(wanted):
                        break
        missing = wanted.difference(frames)
        if missing:
            raise ValueError(f"Video is missing requested source frames {sorted(missing)}: {path}")
        array = np.stack([frames[ordinal] for ordinal in ordinals])
        return torch.from_numpy(array).permute(3, 0, 1, 2).contiguous().float().div_(127.5).sub_(1)

    def _load_embedding(self, row):
        path = self._path(row["text_embedding_path"])
        key = (str(path), row.get("text_embedding_sha256"), row["text_embedding_cache_signature"], row["text_embedding_prompt_sha256"])
        if key in self._embedding_cache:
            self._embedding_cache.move_to_end(key)
            context, mask = self._embedding_cache[key]
            if row.get("text_embedding_valid_tokens") != int(mask.sum()):
                raise ValueError(f"Embedding valid token count mismatch: {path}")
            return context.clone(), mask.clone()
        if self.verify_hashes and _sha256(path) != row.get("text_embedding_sha256"):
            raise ValueError(f"Text embedding SHA mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        digest = row["text_embedding_prompt_sha256"]
        if payload.get("schema") != "vpp2.frozen_wan_text_embedding_cache.v1":
            raise ValueError(f"Unsupported embedding schema: {path}")
        if payload.get("prompt") != row["prompt"] or payload.get("prompt_sha256") != digest or payload.get("prompt_hash") != digest:
            raise ValueError(f"Text embedding prompt identity mismatch: {path}")
        if path.stem != digest or payload.get("cache_signature") != row["text_embedding_cache_signature"]:
            raise ValueError(f"Text embedding cache signature or filename mismatch: {path}")
        context, mask, ids = payload.get("context"), payload.get("mask"), payload.get("input_ids")
        if not isinstance(context, torch.Tensor) or context.shape != (128, 4096) or context.dtype != torch.bfloat16:
            raise ValueError(f"Expected bfloat16 context[128,4096]: {path}")
        if not isinstance(mask, torch.Tensor) or mask.shape != (128,) or mask.dtype != torch.bool:
            raise ValueError(f"Expected bool mask[128]: {path}")
        if not isinstance(ids, torch.Tensor) or ids.shape != (128,) or ids.dtype != torch.int64:
            raise ValueError(f"Expected int64 input_ids[128]: {path}")
        count = int(mask.sum())
        if count < 1 or payload.get("valid_token_count") != count or row.get("text_embedding_valid_tokens") != count:
            raise ValueError(f"Embedding valid token count mismatch: {path}")
        if not torch.equal(mask, torch.arange(128) < count):
            raise ValueError(f"Embedding mask is not right padded: {path}")
        if not bool(torch.isfinite(context).all()):
            raise ValueError(f"Embedding contains nonfinite values: {path}")
        # Preserve real encoder outputs, including padded positions and real mask.
        self._embedding_cache[key] = (context, mask)
        while len(self._embedding_cache) > self.embedding_cache_size:
            self._embedding_cache.popitem(last=False)
        return context.clone(), mask.clone()

    def get_window(self, episode_index, anchor_index, include_target=True):
        result = self.window_indices(episode_index, anchor_index, include_target=include_target)
        row = self.records[operator.index(episode_index)]
        context, mask = self._load_embedding(row)
        condition = self._decode_frames(self._path(row["condition_video_path"]), result["condition_source_indices"].tolist())
        result.update(condition_video=condition, context=context, context_mask=mask, prompt=row["prompt"])
        if include_target:
            target = self._decode_frames(self._path(row["target_video_path"]), result["future_source_indices"].tolist())
            result["video"] = torch.cat((condition, target), dim=1)
            result["video_num_frames"] = result["video"].shape[1]
        return result

    def __getitem__(self, index):
        episode, anchor = self.sample_window(index)
        result = self.get_window(episode, anchor)
        result.update(idx=int(index), dataset_idx=int(index), trajectory_idx=episode, step_idx=anchor)
        return result

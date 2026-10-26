"""Configuration and split checks for the first object-video baseline.

These checks use CPU metadata only. They never instantiate a model or a CUDA
context, so the launcher can provide a genuinely inexpensive --dry-run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT / "configs/obj_video_baseline_v1.yaml"


def load_config(path=DEFAULT_CONFIG, overrides=()) -> DictConfig:
    def read(filename, parents=()):
        filename = Path(filename).resolve()
        if filename in parents:
            raise ValueError(f"Cyclic configuration inheritance: {filename}")
        value = OmegaConf.load(filename)
        base = value.pop("_base_", None)
        if base is not None:
            value = OmegaConf.merge(read(filename.parent / base, (*parents, filename)), value)
        return value

    path = Path(path)
    if not path.is_absolute():
        path = PROJECT / path
    cfg = OmegaConf.merge(read(path), OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    # Paths in project configs are project-relative, independent of shell cwd.
    for key, value in list(cfg.paths.items()):
        if isinstance(value, str):
            cfg.paths[key] = str((PROJECT / value).resolve())
    cfg.output_dir = str((PROJECT / str(cfg.output_dir)).resolve())
    for section, fields in ((cfg.model, ("model_id", "dit_checkpoint_path", "tokenizer_model_id")),
                            (cfg.data.train, ("manifest_path",)),
                            (cfg.data.test, ("manifest_path",))):
        for field in fields:
            section[field] = str((PROJECT / str(section[field])).resolve())
    for name in ("resume", "resume_ckpt"):
        if cfg.get(name):
            cfg[name] = str((PROJECT / str(cfg[name])).resolve())
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(cfg.train_mode == "video_only" and cfg.checkpoint_mode == "full",
            "The first baseline trains the full Video DiT only.")
    require(cfg.model._target_ == "vpp2_obj.model.create_model", "Use the object-video model adapter.")
    require(float(cfg.model.condition_latent_loss_weight) == 1.0,
            "The user-selected first baseline retains prefix loss weight 1.0.")
    require(not cfg.model.load_text_encoder and cfg.model.load_clip_encoder,
            "Use cached text and the frozen RGB CLIP encoder.")
    require(cfg.model.tokenizer_max_len == cfg.model.video_dit_config.text_len == 128,
            "This prompt/cache revision has a 128-token context.")
    require(cfg.model.video_dit_config.video_attention_mask_mode == "bidirectional",
            "Keep the upstream bidirectional attention.")
    require(cfg.model.condition_clip_frame == "first", "CLIP must use the RGB episode-first frame.")
    require(cfg.mixed_precision == "bf16", "The first baseline uses BF16.")
    for split in ("train", "test"):
        ds = cfg.data[split]
        require(ds._target_ == "vpp2_obj.dataset.PairedObjectVideoDataset", "Use the paired dataset.")
        require((ds.history_frames, ds.history_stride, ds.prediction_frames, ds.prediction_stride)
                == (8, 30, 16, 5), "Expected RGB history8/stride30 plus episode first, future16/stride5.")
        require(tuple(ds.video_size) == (224, 448), "Keep observer-left/ego-right at H=224, W=448.")
        require(bool(ds.training) == (split == "train"), "Only the training split may sample training windows.")
        require(bool(ds.balance_tasks) == (split == "train"), "Balance training tasks; keep test windows fixed.")
    require(cfg.data.train.manifest_path != cfg.data.test.manifest_path, "Train and test manifests must differ.")
    require(cfg.get("eval_every", 0) == 0 and cfg.data.get("val") is None,
            "This baseline uses the separate inference CLI for held-out review; no implicit train-set evaluation.")
    require(not cfg.get("preflight", {}).get("enabled", False), "The upstream action preflight is not applicable.")
    for field in ("batch_size", "gradient_accumulation_steps", "max_steps", "reference_global_batch_size"):
        require(isinstance(cfg[field], int) and cfg[field] > 0, f"{field} must be a positive integer.")
    require(cfg.num_workers >= 0 and cfg.warmup_steps >= 0, "Worker/warmup counts cannot be negative.")
    require(not (cfg.get("resume") and cfg.get("resume_ckpt")), "Choose full-state resume or weight initialization.")


def check_data_sources(cfg):
    """Check the frozen partition without decoding videos or loading tensors."""
    contract_path = Path(cfg.paths.data_contract)
    contract = json.loads(contract_path.read_text())
    summary = json.loads(Path(contract["dataset_split"]["summary_path"]).read_text())
    records, fingerprints = {}, {}
    for split in ("train", "test"):
        path = Path(cfg.data[split].manifest_path).resolve()
        if path != Path(contract["manifests"][split]).resolve():
            raise ValueError(f"{split} must use the frozen manifest in the data contract: {path}")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != summary["artifacts"][split + ".jsonl"]["sha256"]:
            raise ValueError(f"The frozen {split} manifest differs from its recorded SHA256.")
        rows = [json.loads(line) for line in data.splitlines() if line]
        mapping = {(row["task"], row["episode"]): row for row in rows}
        if len(mapping) != len(rows) or not rows:
            raise ValueError(f"Empty or duplicate episodes in {split}.")
        if any(row["prompt_review_flags"] for row in rows):
            raise ValueError(f"Unresolved prompt review flags in {split}.")
        if any(row["deduplicated_frame_count"] <= 80 for row in rows):
            raise ValueError("Every episode must supply at least one complete 16-frame future window.")
        records[split], fingerprints[split] = mapping, digest
    if records["train"].keys() & records["test"].keys():
        raise ValueError("Train/test episode overlap.")
    excluded = { (r["task"], r["episode"]) for r in (
        json.loads(line) for line in Path(contract["manifests"]["excluded_from_train_and_test"]).read_text().splitlines() if line)}
    if excluded & (records["train"].keys() | records["test"].keys()):
        raise ValueError("Excluded prompt-review episodes entered a split.")
    media = [{str(Path(row[field]).resolve()) for row in records[split].values()
              for field in ("condition_video_path", "target_video_path")} for split in ("train", "test")]
    if media[0] & media[1]:
        raise ValueError("Train/test media overlap.")
    cache = Path(contract["text_embeddings"]["root"])
    cache_status = json.loads((cache / "status.json").read_text())
    signature = contract["text_embeddings"]["cache_signature"]
    if cache_status["status"] != "complete" or cache_status["cache_signature"] != signature:
        raise ValueError("Text cache must be complete and match the contract.")
    for split in ("train", "test"):
        for row in records[split].values():
            if row["text_embedding_cache_signature"] != signature or row["text_embedding_cache_must_be_rebuilt"]:
                raise ValueError("A row refers to a stale text cache.")
    for name in ("model_id", "dit_checkpoint_path", "tokenizer_model_id"):
        if not Path(cfg.model[name]).exists():
            raise FileNotFoundError(cfg.model[name])
    if cfg.get("resume_ckpt") and not Path(cfg.resume_ckpt).is_file():
        raise FileNotFoundError(cfg.resume_ckpt)
    if cfg.get("resume") and not Path(cfg.resume).is_dir():
        raise FileNotFoundError(cfg.resume)
    return {
        "train_episodes": len(records["train"]), "test_episodes": len(records["test"]),
        "excluded_episodes": len(excluded), "manifest_sha256": fingerprints,
        "text_cache_signature": signature,
        "complete_future_windows": {split: sum(row["deduplicated_frame_count"] - 80
                                               for row in records[split].values()) for split in records},
    }

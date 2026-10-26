"""Train the paired object-video baseline using the upstream video trainer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import DEFAULT_CONFIG, check_data_sources, load_config


def run_training(cfg):
    # Heavy imports occur only after --dry-run has returned.
    import torch
    from accelerate import PartialState
    from accelerate.utils import DistributedType, broadcast_object_list, set_seed
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from vpp2.utils import misc
    from vpp2.utils.logging_config import setup_logging
    from vpp2.utils.training_profiler import TrainingProfiler
    from vpp2.video_training import trainer_class

    if not torch.cuda.is_available():
        raise RuntimeError("Actual Stage-2 training requires CUDA; use --dry-run for CPU review.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not 0 <= local_rank < torch.cuda.device_count():
        raise ValueError("LOCAL_RANK is outside the visible CUDA devices.")
    torch.cuda.set_device(local_rank)
    distributed = PartialState()
    output = Path(cfg.output_dir)
    # Rank zero reserves a fresh run atomically; other ranks cannot create the
    # directory first and make rank zero mistake this launch for an old run.
    startup_error = [None]
    if distributed.is_main_process:
        try:
            output.mkdir(parents=True, exist_ok=bool(cfg.resume))
            config_name = "resume_config.yaml" if cfg.resume else "config.yaml"
            OmegaConf.save(cfg, output / config_name, resolve=True)
        except Exception as exc:
            startup_error[0] = f"Cannot create run output {output}: {exc}"
    broadcast_object_list(startup_error)
    if startup_error[0] is not None:
        raise RuntimeError(startup_error[0])
    distributed.wait_for_everyone()
    setup_logging(is_main_process=distributed.is_main_process, log_file=output / "train.log")
    misc.register_work_dir(str(output))
    set_seed(int(cfg.seed))
    # Validate the dataset and one actual CPU batch before loading the 14B model.
    dataset = instantiate(cfg.data.train)
    from torch.utils.data import default_collate
    probe = default_collate([dataset[0]])
    if tuple(probe["video"].shape[1:]) != (3, 25, 224, 448):
        raise ValueError("Unexpected RGB-prefix + obj-future batch layout.")
    if not torch.equal(probe["video"][:, :, :9], probe["condition_video"]):
        raise ValueError("Training prefix must equal the explicit RGB conditions.")
    del probe
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=f"cuda:{local_rank}")
    # Reuse the original AdamW, scheduler, accumulation, checkpoint and resume
    # implementation. No action model or action/state dataset is instantiated.
    trainer = trainer_class()(
        model=model, train_dataset=dataset, val_dataset=None, cfg=cfg,
        evaluator=None, profiler=TrainingProfiler.from_config(cfg),
    )
    if trainer.accelerator.distributed_type == DistributedType.DEEPSPEED:
        from .distributed import route_deepspeed_training_loss
        route_deepspeed_training_loss(trainer.model)
    trainer.train()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--allow-batch-change", action="store_true",
                       help="Allow a diagnostic global batch different from recipe's 832.")
    train.add_argument("overrides", nargs="*", help="OmegaConf key=value overrides")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    data = check_data_sources(cfg)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 1:
        raise ValueError("WORLD_SIZE must be positive.")
    actual_batch = int(cfg.batch_size * cfg.gradient_accumulation_steps * world)
    epoch_samples = int(cfg.data.train.get("samples_per_epoch") or data["complete_future_windows"]["train"])
    if epoch_samples % actual_batch:
        raise ValueError(
            f"samples_per_epoch={epoch_samples} must be divisible by global batch {actual_batch}; "
            "otherwise Accelerate flushes a partial accumulation at the epoch boundary."
        )
    summary = {
        **data, "train_mode": cfg.train_mode, "video_initializer": cfg.model.dit_checkpoint_path,
        "condition_frames": 9, "object_future_frames": 16, "video_frames": 25,
        "condition_latent_loss_weight": cfg.model.condition_latent_loss_weight,
        "history_stride": 30, "prediction_stride": 5, "uniform_time_resampling": False,
        "world_size": world, "micro_batch_per_gpu": cfg.batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "samples_per_epoch": epoch_samples,
        "global_batch": actual_batch, "reference_global_batch": cfg.reference_global_batch_size,
        "max_steps": cfg.max_steps, "output_dir": cfg.output_dir,
        "dry_run": args.dry_run, "hardware_fit_verified": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0
    if actual_batch != cfg.reference_global_batch_size and not args.allow_batch_change:
        raise ValueError("Global batch differs from 832; set accumulation or --allow-batch-change for a diagnostic.")
    if cfg.resume and not Path(cfg.resume).is_dir():
        raise FileNotFoundError(cfg.resume)
    os.environ.setdefault("VPP2_TORCH_LOAD_MMAP", "1")
    run_training(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

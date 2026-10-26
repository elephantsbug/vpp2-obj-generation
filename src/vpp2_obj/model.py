"""RoboTwin RGB-history/object-future adapter for the unchanged Video-10k loss."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from vpp2.models.wan21_14b.wan21 import Wan21VideoModel


class ObjectVideoModel(Wan21VideoModel):
    """First baseline: nine RGB frames followed by sixteen object-only frames.

    All seven latent positions retain the upstream flow-matching objective.
    Prefix/future losses are diagnostics, not two independently added objectives.
    Inference uses the upstream sampler without overwriting the noisy RGB prefix.
    """

    condition_frames = 9
    future_frames = 16
    total_frames = 25
    condition_latent_frames = 3
    total_latent_frames = 7

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._check_baseline()
        self.dit.requires_grad_(True)
        self._freeze_auxiliary_components()

    def _check_baseline(self):
        if self.condition_latent_loss_weight != 1.0:
            raise ValueError("The first object-video baseline requires condition_latent_loss_weight=1.0.")
        if int(self.vae.temporal_downsample_factor) != 4:
            raise ValueError("The first object-video baseline requires VAE temporal_downsample_factor=4.")

    def _freeze_auxiliary_components(self):
        for component in (self.vae, self.clip_encoder, self.text_encoder):
            if component is not None:
                component.eval().requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_auxiliary_components()
        return self

    def forward(self, sample, tiled: bool = False):
        # Distributed training must enter through the prepared engine's __call__
        # so DeepSpeed can register output hooks, including loss/GAS scaling.
        return self.training_loss(sample, tiled=tiled)

    def encode_prompt(self, prompt):
        raise ValueError(
            "Object-video training/inference requires cached context and its real context_mask; "
            "the upstream prompt encoder changes the padding mask."
        )

    def _check_cached_text(self, context, context_mask, batch_size: int):
        if not isinstance(context, torch.Tensor) or not isinstance(context_mask, torch.Tensor):
            raise ValueError("Cached context and its real context_mask tensors are required.")
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError("Cached context/context_mask must have shapes [B,L,D]/[B,L].")
        if context.shape[:2] != context_mask.shape or context.shape[0] != batch_size:
            raise ValueError("Cached context/context_mask batch and sequence dimensions must match.")
        if context.shape[2] != self.text_dim:
            raise ValueError(f"Cached context dimension must equal text_dim={self.text_dim}.")
        if context_mask.dtype != torch.bool:
            raise ValueError("context_mask must preserve the cached boolean token mask.")

    def _encode_separate_condition_latents(
        self, condition_video, condition_observation_video=None, tiled: bool = False
    ):
        if condition_observation_video is not None:
            raise ValueError("Nine RGB frames already include the current observation; no extra observation is allowed.")
        if condition_video.ndim != 5 or condition_video.shape[1:3] != (3, self.condition_frames):
            raise ValueError("condition_video must have shape [B,3,9,H,W].")
        latents = super()._encode_separate_condition_latents(condition_video, tiled=tiled)
        if latents.shape[2] != self.condition_latent_frames:
            raise ValueError("Nine RGB condition frames must encode to exactly three latent frames.")
        return latents

    def build_inputs(self, sample, tiled: bool = False):
        self._check_baseline()
        if sample.get("condition_observation_video") is not None:
            raise ValueError("No extra condition_observation_video is allowed in the nine-frame baseline.")
        video, condition = sample.get("video"), sample.get("condition_video")
        if not isinstance(video, torch.Tensor) or video.ndim != 5 or video.shape[1:3] != (3, self.total_frames):
            raise ValueError("sample['video'] must be [B,3,25,H,W]: nine RGB frames then sixteen object frames.")
        expected = (video.shape[0], 3, self.condition_frames, *video.shape[-2:])
        if not isinstance(condition, torch.Tensor) or tuple(condition.shape) != expected:
            raise ValueError(f"Explicit RGB condition_video must have shape {expected}.")
        if video.device != condition.device or video.dtype != condition.dtype:
            raise ValueError("Mixed video and RGB condition must use the same device and dtype.")
        # One scalar equality check, with no full-video CPU copy. A shared prefix
        # view avoids even that synchronization when the caller can supply one.
        prefix = video[:, :, :self.condition_frames]
        same_view = prefix.data_ptr() == condition.data_ptr() and prefix.stride() == condition.stride()
        if not same_view and not torch.equal(prefix, condition):
            raise ValueError("The first nine video frames must exactly match the RGB condition_video.")
        self._check_cached_text(sample.get("context"), sample.get("context_mask"), video.shape[0])
        inputs = super().build_inputs(sample, tiled=tiled)
        if inputs["input_latents"].shape[2] != self.total_latent_frames:
            raise ValueError("The 25-frame mixed target must encode to exactly seven latent frames.")
        if inputs["condition_latents"].shape[2] != self.condition_latent_frames:
            raise ValueError("The RGB prefix must encode to exactly three latent frames.")
        return inputs

    @torch.no_grad()
    def _segment_loss_per_sample(self, pred, target, image_is_pad):
        """Detached per-segment means using the upstream four-frame padding rule."""
        error = F.mse_loss(pred.detach().float(), target.detach().float(), reduction="none").mean(dim=(1, 3, 4))
        valid = torch.ones_like(error)
        if image_is_pad is not None:
            tail_pad = image_is_pad[:, 1:].reshape(image_is_pad.shape[0], -1, 4).all(dim=2)
            latent_pad = torch.cat((image_is_pad[:, :1], tail_pad), dim=1)
            valid = (~latent_pad).to(device=error.device, dtype=error.dtype)
        means = []
        for segment in (slice(0, self.condition_latent_frames), slice(self.condition_latent_frames, None)):
            weights = valid[:, segment]
            means.append((error[:, segment] * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0))
        return tuple(means)

    def training_loss(self, sample, tiled: bool = False):
        # Keep the short upstream Wan21VideoModel.training_loss sequence explicit
        # so timestep weighting is shared by total loss and both diagnostics.
        # The original helper owns the optimized objective and normalization.
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        noise = torch.randn_like(input_latents)
        timestep = self.train_video_scheduler.sample_training_t(
            batch_size=input_latents.shape[0], device=self.device, dtype=input_latents.dtype
        )
        noisy_latents = self.train_video_scheduler.add_noise(input_latents, noise, timestep)
        target = self.train_video_scheduler.training_target(input_latents, noise, timestep)
        pred = self.dit(
            x=noisy_latents,
            timestep=timestep,
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            action=None,
            clip_fea=inputs["clip_fea"],
            condition_latents=inputs["condition_latents"],
        )
        loss_per_sample = super()._compute_video_loss_per_sample(
            pred_video=pred,
            target_video=target,
            image_is_pad=inputs["image_is_pad"],
            condition_latent_frames=inputs["condition_latents"].shape[2],
        )
        weight = self.train_video_scheduler.training_weight(timestep).to(
            device=loss_per_sample.device, dtype=loss_per_sample.dtype
        )
        loss_video = (loss_per_sample * weight).mean()
        prefix, future = self._segment_loss_per_sample(pred, target, inputs["image_is_pad"])
        # Transfer three diagnostic scalars together instead of synchronizing
        # separately for each logger field.
        values = torch.stack((loss_video.detach(), (prefix * weight).mean(), (future * weight).mean())).cpu().tolist()
        return loss_video, dict(zip(("loss_video", "loss_rgb_prefix", "loss_obj_future"), values))

    @torch.no_grad()
    def infer_object_future(
        self,
        *,
        condition_video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        generator: Optional[torch.Generator] = None,
        tiled: bool = False,
    ):
        """Generate sixteen object frames from RGB/history and cached text only.

        There is intentionally no target-video, prompt, extra-observation, or
        prefix-clamping argument. The VAE decodes all 25 generated frames first.
        """
        self._check_baseline()
        if not isinstance(condition_video, torch.Tensor):
            raise ValueError("condition_video must be a tensor containing nine RGB frames.")
        if condition_video.ndim == 4:
            condition_video = condition_video.unsqueeze(0)
        if condition_video.ndim != 5 or condition_video.shape[:3] != (1, 3, self.condition_frames):
            raise ValueError("Inference condition_video must be [3,9,H,W] or [1,3,9,H,W].")
        if isinstance(context, torch.Tensor) and context.ndim == 2:
            context = context.unsqueeze(0)
        if isinstance(context_mask, torch.Tensor) and context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        self._check_cached_text(context, context_mask, 1)
        result = super().infer(
            prompt=None,
            input_image=condition_video[:, :, 0],
            num_frames=self.total_frames,
            context=context,
            context_mask=context_mask,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            generator=generator,
            tiled=tiled,
            condition_video=condition_video,
        )
        if len(result["video"]) != self.total_frames:
            raise ValueError("The VAE must decode all 25 frames before exporting the sixteen object frames.")
        return {**result, "video": result["video"][self.condition_frames:self.total_frames]}


def create_model(*, video_dit_config, video_scheduler, model_dtype, device, **kwargs):
    """Hydra factory matching the upstream Video-10k factory's arguments."""
    from omegaconf import OmegaConf

    if float(kwargs.get("condition_latent_loss_weight", 1.0)) != 1.0:
        raise ValueError("The first object-video baseline requires condition_latent_loss_weight=1.0.")

    def plain(value):
        return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value

    return ObjectVideoModel.from_wan21_14b_pretrained(
        video_dit_config=plain(video_dit_config),
        video_scheduler=plain(video_scheduler),
        torch_dtype=model_dtype,
        device=device,
        **kwargs,
    )

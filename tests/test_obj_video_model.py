"""CPU-only checks of the object adapter against the real upstream loss/sampler."""

from pathlib import Path
from types import SimpleNamespace
import inspect
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "third_party/vpp2/src"))
sys.path.insert(0, str(PROJECT / "src"))

from vpp2.models.wan21_14b.wan21 import Wan21VideoModel
from vpp2_obj.model import ObjectVideoModel, create_model


class StubVAE(torch.nn.Module):
    temporal_downsample_factor = 4
    upsampling_factor = 8

    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(1.0))
        self.model = SimpleNamespace(z_dim=3)
        self.encoded_videos = []
        self.decoded_latents = None

    def encode(self, videos, **kwargs):
        results = []
        for video in videos:
            self.encoded_videos.append(video.detach().clone())
            frames = video.mean(dim=(-1, -2), keepdim=True)
            tail = frames[:, 1:].reshape(3, -1, 4, 1, 1).mean(dim=2)
            latent = torch.cat((frames[:, :1], tail), dim=1)
            results.append(latent.expand(-1, -1, video.shape[-2] // 8, video.shape[-1] // 8).clone())
        return results

    def decode(self, latents, **kwargs):
        self.decoded_latents = latents.detach().clone()
        frames = (latents.shape[2] - 1) * 4 + 1
        # Distinct output frame values make the [9:25] crop observable.
        values = torch.linspace(-1, 1, frames, device=latents.device).reshape(1, 1, frames, 1, 1)
        return values.expand(1, 3, frames, latents.shape[-2] * 8, latents.shape[-1] * 8)


class StubDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.2))
        self.bias = torch.nn.Parameter(torch.tensor(0.05))
        self.calls = []

    def forward(self, *, x, timestep, context, context_mask, action, clip_fea, condition_latents):
        self.calls.append({"x": x.detach().clone(), "timestep": timestep.detach().clone(),
                           "mask": context_mask.detach().clone(), "condition": condition_latents.detach().clone()})
        return x * self.scale + self.bias + condition_latents.mean(dim=2, keepdim=True) * 0.01


class StubCLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(1.0))
        self.videos = []

    def forward(self, videos):
        self.videos.extend(video.detach().clone() for video in videos)
        return torch.stack([video.mean().reshape(1, 1).expand(2, 4) for video in videos])


def make_model(cls=ObjectVideoModel, **kwargs):
    return cls(dit=StubDiT(), vae=StubVAE(), clip_encoder=StubCLIP(),
               text_encoder=torch.nn.Linear(4, 4), text_dim=4,
               device="cpu", torch_dtype=torch.float32, **kwargs)


def make_sample(batch=2):
    rgb = torch.arange(9, dtype=torch.float32).reshape(1, 1, 9, 1, 1).expand(batch, 3, 9, 16, 16).clone() / 10
    obj = -torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 16, 1, 1).expand(batch, 3, 16, 16, 16).clone() / 20
    mask = torch.tensor([True, True, False, False]).expand(batch, -1).clone()
    return {"video": torch.cat((rgb, obj), dim=2), "condition_video": rgb,
            "context": torch.ones(batch, 4, 4), "context_mask": mask,
            "image_is_pad": torch.zeros(batch, 25, dtype=torch.bool)}


class ObjectVideoModelTests(unittest.TestCase):
    def test_rgb_condition_and_mixed_target_map_to_three_and_seven_latents(self):
        model, sample = make_model(), make_sample(batch=1)
        inputs = model.build_inputs(sample)
        self.assertEqual(inputs["input_latents"].shape, (1, 3, 7, 2, 2))
        self.assertEqual(inputs["condition_latents"].shape, (1, 3, 3, 2, 2))
        torch.testing.assert_close(inputs["input_latents"][:, :, :3], inputs["condition_latents"], rtol=0, atol=0)
        self.assertTrue(torch.all(inputs["input_latents"][:, :, 3:] < 0))
        self.assertEqual([v.shape[1] for v in model.vae.encoded_videos], [25, 9])
        torch.testing.assert_close(model.vae.encoded_videos[0], sample["video"][0])
        torch.testing.assert_close(model.vae.encoded_videos[1], sample["condition_video"][0])
        torch.testing.assert_close(model.clip_encoder.videos[0].float(), sample["condition_video"][0, :, :1])
        torch.testing.assert_close(inputs["context_mask"], sample["context_mask"])
        self.assertTrue(torch.all(inputs["context"][:, 2:] == 1))  # No zeroing/expanding text padding.

    def test_total_loss_and_dit_gradient_equal_upstream_with_or_without_padding(self):
        for padded in (False, True):
            with self.subTest(padded=padded):
                model, upstream, sample = make_model(), make_model(Wan21VideoModel), make_sample()
                if padded:
                    sample["image_is_pad"][0, 1:5] = True
                    sample["image_is_pad"][0, 21:25] = True
                    sample["image_is_pad"][1, 9] = True  # Partially padded latent remains valid.
                for instance in (model, upstream):
                    instance.train_video_scheduler.sample_training_t = lambda **kwargs: torch.tensor([200.0, 600.0])
                torch.manual_seed(712)
                actual, metrics = model.training_loss(sample)
                torch.manual_seed(712)
                expected, _ = upstream.training_loss(sample)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual.backward()
                expected.backward()
                for p, q in zip(model.dit.parameters(), upstream.dit.parameters()):
                    torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
                self.assertEqual(metrics["loss_video"], actual.item())
                self.assertEqual(set(metrics), {"loss_video", "loss_rgb_prefix", "loss_obj_future"})
                if not padded:
                    self.assertAlmostEqual(metrics["loss_video"],
                                           (3 * metrics["loss_rgb_prefix"] + 4 * metrics["loss_obj_future"]) / 7,
                                           places=6)
                torch.testing.assert_close(model.dit.calls[-1]["mask"], sample["context_mask"])
                for auxiliary in (model.vae, model.clip_encoder, model.text_encoder):
                    self.assertTrue(all(p.grad is None for p in auxiliary.parameters()))

    def test_segment_logs_use_valid_positions_and_actual_timestep_weight(self):
        model, sample = make_model(), make_sample()
        sample["image_is_pad"][0, 1:5] = True
        sample["image_is_pad"][0, 21:25] = True
        sample["image_is_pad"][1, 9:25] = True
        pred = torch.arange(1, 8, dtype=torch.float32).reshape(1, 1, 7, 1, 1).expand(2, 3, 7, 2, 2).clone()
        pred.requires_grad_(True)
        prefix, future = model._segment_loss_per_sample(pred, torch.zeros_like(pred), sample["image_is_pad"])
        torch.testing.assert_close(prefix, torch.tensor([5.0, 14.0 / 3]))
        torch.testing.assert_close(future, torch.tensor([77.0 / 3, 0.0]))
        self.assertFalse(prefix.requires_grad)
        self.assertFalse(future.requires_grad)
        # Drive the real training wrapper with known prediction/target tensors
        # and unequal scheduler weights to distinguish per-sample from global averaging.
        with patch.object(model.train_video_scheduler, "training_target", return_value=torch.zeros_like(pred)), \
             patch.object(model.dit, "forward", return_value=pred), \
             patch.object(model.train_video_scheduler, "training_weight", return_value=torch.tensor([2.0, 0.5])):
            _, logs = model.training_loss(sample)
        self.assertAlmostEqual(logs["loss_rgb_prefix"], float((prefix * torch.tensor([2.0, 0.5])).mean()), places=6)
        self.assertAlmostEqual(logs["loss_obj_future"], float((future * torch.tensor([2.0, 0.5])).mean()), places=6)

    def test_missing_or_mismatched_rgb_prefix_and_extra_observation_are_rejected(self):
        model = make_model()
        for mutation, message in (
            (lambda s: s.pop("condition_video"), "Explicit RGB"),
            (lambda s: s.update(video=s["video"][:, :, 9:]), "25"),
            (lambda s: s.update(condition_video=s["condition_video"] + 1), "exactly match"),
            (lambda s: s.update(condition_observation_video=s["video"][:, :, :1]), "No extra"),
            (lambda s: s.update(context_mask=s["context_mask"].long()), "boolean"),
        ):
            with self.subTest(message=message):
                sample = make_sample()
                mutation(sample)
                with self.assertRaisesRegex(ValueError, message):
                    model.build_inputs(sample)
        with self.assertRaisesRegex(ValueError, "cached context"):
            model.encode_prompt("Do not change the mask.")

    def test_weight_and_latent_layout_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "weight=1.0"):
            make_model(condition_latent_loss_weight=0.0)
        model = make_model()
        sample = make_sample()
        with patch.object(model.vae, "encode", return_value=torch.zeros(2, 3, 2, 2, 2)):
            with self.assertRaisesRegex(ValueError, "three latent"):
                model.build_inputs(sample)
        with patch.object(model.vae, "encode", side_effect=[torch.zeros(2, 3, 6, 2, 2),
                                                            torch.zeros(2, 3, 3, 2, 2)]):
            with self.assertRaisesRegex(ValueError, "seven latent"):
                model.build_inputs(sample)
        model.condition_latent_loss_weight = 0
        with self.assertRaisesRegex(ValueError, "weight=1.0"):
            model.build_inputs(sample)

    def test_only_dit_is_trainable_and_auxiliary_modules_stay_in_eval(self):
        model = make_model().train()
        self.assertTrue(model.dit.training)
        self.assertTrue(all(p.requires_grad for p in model.dit.parameters()))
        for auxiliary in (model.vae, model.clip_encoder, model.text_encoder):
            self.assertFalse(auxiliary.training)
            self.assertTrue(all(not p.requires_grad for p in auxiliary.parameters()))
        model.eval()
        self.assertFalse(model.dit.training)

    def test_infer_uses_original_sampler_without_prefix_clamp_and_decodes_before_crop(self):
        model, sample = make_model(), make_sample(batch=1)
        condition, context, mask = sample["condition_video"][0], sample["context"][0], sample["context_mask"][0]
        frames = model.infer_object_future(condition_video=condition, context=context,
                                          context_mask=mask, num_inference_steps=2, seed=29)["video"]
        object_calls = [{key: value.clone() for key, value in call.items()} for call in model.dit.calls]
        decoded = model.vae.decoded_latents.clone()
        self.assertEqual(len(frames), 16)
        self.assertEqual(decoded.shape[2], 7)
        self.assertEqual([v.shape[1] for v in model.vae.encoded_videos], [9])
        self.assertTrue(all(call["condition"].shape[2] == 3 for call in object_calls))
        self.assertTrue(all(torch.equal(call["mask"], mask.unsqueeze(0)) for call in object_calls))
        model.dit.calls.clear()
        full = Wan21VideoModel.infer(model, prompt=None, input_image=condition[:, 0], num_frames=25,
                                    context=context, context_mask=mask, condition_video=condition,
                                    num_inference_steps=2, seed=29)["video"]
        for actual, expected in zip(object_calls, model.dit.calls):
            torch.testing.assert_close(actual["x"], expected["x"], rtol=0, atol=0)
        torch.testing.assert_close(decoded, model.vae.decoded_latents, rtol=0, atol=0)
        for actual, expected in zip(frames, full[9:25]):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        # The noisy prefix is neither restored to clean RGB nor updated on a prescribed trajectory.
        self.assertFalse(torch.equal(object_calls[0]["x"][:, :, :3], object_calls[0]["condition"]))
        timestep = object_calls[1]["timestep"].reshape(1, 1, 1, 1, 1) / 1000
        prescribed = (1 - timestep) * object_calls[0]["condition"] + timestep * object_calls[0]["x"][:, :, :3]
        self.assertFalse(torch.allclose(object_calls[1]["x"][:, :, :3], prescribed))

    def test_object_inference_has_no_future_target_or_extra_condition_argument(self):
        model, sample = make_model(), make_sample(batch=1)
        kwargs = {"condition_video": sample["condition_video"], "context": sample["context"],
                  "context_mask": sample["context_mask"]}
        for forbidden in ("video", "target_video", "condition_observation_video", "prompt", "num_frames"):
            with self.subTest(argument=forbidden), self.assertRaises(TypeError):
                model.infer_object_future(**kwargs, **{forbidden: sample["video"]})
        self.assertFalse(any(p.kind == inspect.Parameter.VAR_KEYWORD
                             for p in inspect.signature(model.infer_object_future).parameters.values()))
        with self.assertRaisesRegex(ValueError, "9"):
            model.infer_object_future(**{**kwargs, "condition_video": sample["condition_video"][:, :, :5]})

    def test_factory_preserves_subclass_and_converts_hydra_configs(self):
        from omegaconf import OmegaConf
        config = OmegaConf.create({"text_dim": 4})
        scheduler = OmegaConf.create({"train_shift": 5})
        sentinel = object()
        with patch.object(ObjectVideoModel, "from_wan21_14b_pretrained", return_value=sentinel) as loader:
            result = create_model(video_dit_config=config, video_scheduler=scheduler,
                                  model_dtype=torch.bfloat16, device="cpu", model_id="unused",
                                  dit_checkpoint_path="unused", tokenizer_model_id="unused",
                                  condition_latent_loss_weight=1.0)
        self.assertIs(result, sentinel)
        self.assertIs(type(loader.call_args.kwargs["video_dit_config"]), dict)
        self.assertIs(type(loader.call_args.kwargs["video_scheduler"]), dict)
        self.assertIs(loader.call_args.kwargs["torch_dtype"], torch.bfloat16)
        with patch.object(ObjectVideoModel, "from_wan21_14b_pretrained") as loader:
            with self.assertRaisesRegex(ValueError, "weight=1.0"):
                create_model(video_dit_config={}, video_scheduler={}, model_dtype=torch.float32,
                             device="cpu", condition_latent_loss_weight=0)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()

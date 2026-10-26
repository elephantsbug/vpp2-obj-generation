"""CPU regression for the installed DeepSpeed loss/forward accumulation path.

We execute the installed forward/backward methods and output-hook manager by
extracting their AST, without importing DeepSpeed or initializing CUDA/process
groups. Communication, loss-scale=1 BF16 handling, and lifecycle bookkeeping
are stubbed. This checks gradient scaling and project routing, not distributed
numerics, real BF16 rounding, ZeRO memory use, or GPU checkpointing behavior.
"""

import ast
import contextlib
import copy
from importlib.metadata import distribution
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "third_party/vpp2/src"))
sys.path.insert(0, str(PROJECT / "src"))

from vpp2_obj.distributed import route_deepspeed_training_loss


class _ZeroOptimizerStub:
    def scale_if_loss(self, loss):
        return loss  # The baseline's BF16 loss scaler has scale=1.

    def clear_backward_seen_flag(self):
        pass


class _DummyOptimizerStub:
    pass


def _installed_deepspeed_methods():
    runtime = Path(distribution("deepspeed").locate_file("deepspeed/runtime"))
    engine_tree = ast.parse((runtime / "engine.py").read_text())
    engine_class = next(node for node in engine_tree.body
                        if isinstance(node, ast.ClassDef) and node.name == "DeepSpeedEngine")
    names = {"forward", "backward", "_backward_prologue_per_tensor", "__getattr__"}
    methods = [node for node in engine_class.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in methods} != names:
        raise AssertionError("Installed DeepSpeed no longer exposes the audited engine methods.")
    for node in methods:
        node.decorator_list = []  # Omit NVTX instrumentation, not method logic.
    utils_tree = ast.parse((runtime / "utils.py").read_text())
    helper_names = {"OutputBackwardHookManager", "register_output_backward_hooks", "maybe_loss_for_backward"}
    helpers = [node for node in utils_tree.body
               if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in helper_names]
    if {node.name for node in helpers} != helper_names:
        raise AssertionError("Installed DeepSpeed no longer exposes the audited output-hook implementation.")
    namespace = {
        "torch": torch,
        "ZeROOptimizer": _ZeroOptimizerStub,
        "DummyOptim": _DummyOptimizerStub,
        "compiled_autograd": lambda *args, **kwargs: contextlib.nullcontext(),
        "autocast_if_enabled": lambda *args, **kwargs: contextlib.nullcontext(),
    }
    module = ast.fix_missing_locations(ast.Module(body=helpers + methods, type_ignores=[]))
    exec(compile(module, "<installed-deepspeed-cpu-regression>", "exec"), namespace)
    return namespace


_DS = _installed_deepspeed_methods()


class _TinyObjective(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.6, -0.4], dtype=torch.float64))
        self.auxiliary = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.auxiliary.weight.copy_(torch.eye(2, dtype=torch.float64))
        self.auxiliary.eval().requires_grad_(False)

    def training_loss(self, sample):
        prediction = self.auxiliary(sample["x"]) @ self.weight
        loss = (prediction - sample["y"]).square().mean()
        return loss, {"loss_video": float(loss.detach()), "unscaled_marker": 11.0}

    def forward(self, sample):
        return self.training_loss(sample)


class _CPUEngine(torch.nn.Module):
    """The real installed engine methods around a CPU objective and no-op runtime."""

    forward = _DS["forward"]
    backward = _DS["backward"]
    _backward_prologue_per_tensor = _DS["_backward_prologue_per_tensor"]
    __getattr__ = _DS["__getattr__"]

    def __init__(self, module, gas, torch_style_backward=True):
        super().__init__()
        # DeepSpeed's attribute delegation expects module in __dict__.
        object.__setattr__(self, "module", module)
        self.optimizer = _ZeroOptimizerStub()
        self.gas = gas
        self._is_compiled_autograd_enabled = False
        self._compile_kwargs = {}
        self._running_engine_backward = False
        self._support_torch_style_backward = torch_style_backward
        self._scale_wrt_gas = True
        self.prologue_calls = 0

    def gradient_accumulation_steps(self):
        return self.gas

    def autotuning_profile_model_info(self):
        return False

    def is_deepcompile_enabled(self):
        return False

    def is_deepcompile_active(self):
        return False

    def eigenvalue_enabled(self):
        return False

    def zero_optimization(self):
        return True

    def amp_enabled(self):
        return False

    def _backward_prologue(self):
        self.prologue_calls += 1

    def _backward_epilogue(self):
        pass


@contextlib.contextmanager
def _cpu_engine_identity():
    """Patch only import/type identity for the strict project routing helper.

    The engine's executed methods and output hooks remain the installed source.
    No production code is modified and no DeepSpeed package import is needed.
    """
    package = ModuleType("deepspeed")
    package.__path__ = []
    runtime = ModuleType("deepspeed.runtime")
    runtime.__path__ = []
    engine_module = ModuleType("deepspeed.runtime.engine")
    engine_module.DeepSpeedEngine = _CPUEngine
    with patch.dict(sys.modules, {"deepspeed": package, "deepspeed.runtime": runtime,
                                  "deepspeed.runtime.engine": engine_module}):
        yield


def _samples(gas, micro_batch=2):
    generator = torch.Generator(device="cpu").manual_seed(71)
    return {"x": torch.randn(gas * micro_batch, 2, generator=generator, dtype=torch.float64),
            "y": torch.randn(gas * micro_batch, generator=generator, dtype=torch.float64)}


class ObjectVideoAccumulationTests(unittest.TestCase):
    def test_routed_accumulation_matches_large_batch_gradient_and_parameter_update(self):
        for gas in (1, 8, 52):
            with self.subTest(gas=gas):
                reference = _TinyObjective()
                model = copy.deepcopy(reference)
                engine = _CPUEngine(model, gas)
                original_module_loss = model.training_loss
                data = _samples(gas)
                whole_loss, _ = reference(data)
                whole_loss.backward()
                with _cpu_engine_identity():
                    route_deepspeed_training_loss(engine)
                    routed_method = engine.training_loss
                    route_deepspeed_training_loss(engine)
                self.assertIs(engine.training_loss, routed_method)  # Idempotent, not nested.
                self.assertEqual(model.training_loss, original_module_loss)
                for index in range(gas):
                    micro = {name: values[2 * index:2 * (index + 1)] for name, values in data.items()}
                    expected_loss, expected_logs = model.training_loss(micro)
                    actual_loss, actual_logs = engine.training_loss(micro)
                    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
                    self.assertEqual(actual_logs, expected_logs)
                    engine.backward(actual_loss)  # No extra manual division.
                torch.testing.assert_close(model.weight.grad, reference.weight.grad, rtol=1e-10, atol=1e-12)
                self.assertEqual(engine.prologue_calls, gas)
                self.assertIsNone(model.auxiliary.weight.grad)
                self.assertFalse(model.auxiliary.weight.requires_grad)
                self.assertFalse(model.auxiliary.training)
                auxiliary_before = model.auxiliary.weight.detach().clone()
                for objective in (model, reference):
                    optimizer = torch.optim.SGD([objective.weight], lr=0.1)
                    optimizer.step()
                torch.testing.assert_close(model.weight, reference.weight, rtol=1e-10, atol=1e-12)
                torch.testing.assert_close(model.auxiliary.weight, auxiliary_before, rtol=0, atol=0)

    def test_old_delegated_path_reproduces_gas_times_gradient(self):
        for gas in (1, 8, 52):
            with self.subTest(gas=gas):
                reference = _TinyObjective()
                engine = _CPUEngine(copy.deepcopy(reference), gas)
                data = _samples(gas)
                reference.training_loss(data)[0].backward()
                for index in range(gas):
                    micro = {name: values[2 * index:2 * (index + 1)] for name, values in data.items()}
                    loss, _ = engine.training_loss(micro)  # Real __getattr__ delegates past forward hooks.
                    engine.backward(loss)
                torch.testing.assert_close(engine.module.weight.grad, reference.weight.grad * gas,
                                           rtol=1e-10, atol=1e-12)
                self.assertEqual(engine.prologue_calls, 0)

    def test_normal_forward_scaling_does_not_depend_on_torch_style_backward_flag(self):
        for enabled in (False, True):
            with self.subTest(torch_style_backward=enabled):
                reference = _TinyObjective()
                engine = _CPUEngine(copy.deepcopy(reference), 8, torch_style_backward=enabled)
                data = _samples(8)
                reference(data)[0].backward()
                with _cpu_engine_identity():
                    route_deepspeed_training_loss(engine)
                for index in range(8):
                    micro = {name: values[2 * index:2 * (index + 1)] for name, values in data.items()}
                    engine.backward(engine.training_loss(micro)[0])
                torch.testing.assert_close(engine.module.weight.grad, reference.weight.grad, rtol=1e-10, atol=1e-12)

    def test_helper_rejects_unprepared_or_incompatible_models(self):
        with _cpu_engine_identity():
            with self.assertRaisesRegex(TypeError, "DeepSpeedEngine"):
                route_deepspeed_training_loss(_TinyObjective())
            without_forward = torch.nn.Module()
            without_forward.training_loss = lambda sample: None
            with self.assertRaisesRegex(TypeError, "forward"):
                route_deepspeed_training_loss(_CPUEngine(without_forward, 8))
            with self.assertRaisesRegex(TypeError, "training_loss"):
                route_deepspeed_training_loss(_CPUEngine(torch.nn.Linear(2, 1), 8))

    def test_object_model_forward_delegates_without_loss_or_logging_changes(self):
        from vpp2_obj.model import ObjectVideoModel

        model = object.__new__(ObjectVideoModel)
        torch.nn.Module.__init__(model)
        expected = (torch.tensor(3.0, requires_grad=True), {"loss_video": 3.0})
        model.training_loss = Mock(return_value=expected)
        sample = {"sentinel": "the unchanged batch"}
        actual = model(sample, tiled=True)
        self.assertIs(actual, expected)
        model.training_loss.assert_called_once_with(sample, tiled=True)


if __name__ == "__main__":
    unittest.main()

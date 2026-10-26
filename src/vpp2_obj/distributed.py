"""Route the upstream loss API through DeepSpeed's normal forward lifecycle."""

import logging
from types import MethodType

logger = logging.getLogger(__name__)


def _training_loss_via_engine(engine, *args, **kwargs):
    # __call__ runs the engine/module hooks. Calling module.training_loss or
    # engine.forward directly would bypass part of the normal module lifecycle.
    return engine(*args, **kwargs)


def route_deepspeed_training_loss(engine):
    """Adapt VideoTrainer's training_loss call after accelerator.prepare().

    DeepSpeed 0.18.5 registers gradient-accumulation scaling on forward outputs.
    Its automatic delegation of engine.training_loss to module.training_loss
    bypasses those hooks. Bind only the prepared engine instance: its module's
    training_loss stays unchanged and is reached via ObjectVideoModel.forward.
    No manual division by gradient_accumulation_steps is performed here.
    """
    # Keep heavyweight imports out of the training CLI's CPU metadata dry-run.
    from deepspeed.runtime.engine import DeepSpeedEngine
    from torch import nn

    if not isinstance(engine, DeepSpeedEngine):
        raise TypeError("Loss routing requires the prepared DeepSpeedEngine.")
    if getattr(engine, "_vpp2_loss_uses_engine_forward", False):
        return engine
    if type(engine.module).forward is nn.Module.forward:
        raise TypeError("The wrapped video model must implement forward(sample).")
    if not callable(getattr(engine.module, "training_loss", None)):
        raise TypeError("The wrapped video model must retain its training_loss API.")
    engine.training_loss = MethodType(_training_loss_via_engine, engine)
    engine._vpp2_loss_uses_engine_forward = True
    logger.info(
        "Training loss routed through DeepSpeed forward; accumulation scaling "
        "and backward hooks are active (gradient_accumulation_steps=%s).",
        engine.gradient_accumulation_steps(),
    )
    return engine

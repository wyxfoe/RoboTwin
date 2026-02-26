"""Robot model factory — uses the official AudioX package (audiox).

AudioX is a 'diffusion_cond' model. Our JSON configs use
model_type='robot_diffusion' as a marker; this function remaps it to the
AudioX-native 'diffusion_cond' type before calling the factory.

The official AudioX conditioner factory does not know about
'trajectory' conditioners, so we intercept the conditioning config
and build it with our patched factory instead.
"""

import copy

from audiox.models.factory import create_model_from_config
from audiox.models.diffusion import (
    ConditionedDiffusionModelWrapper,
    DiTWrapper,
)
from .conditioners import create_robot_conditioner_from_config


def create_robot_model_from_config(config):
    """Create an AudioX diffusion model from a robotics config dict.

    AudioX's create_diffusion_cond_from_config reads:
        - config["model"]["io_channels"]
        - config["model"]["diffusion"]["type"]  / ["config"]
        - config["model"]["conditioning"]
        - config["sample_rate"]
        - config["sample_size"]

    Our robotics JSON stores io_channels inside model.diffusion.config,
    so this function copies it up to model.io_channels where the factory
    expects it.

    If the conditioning config contains robot-specific types (e.g.
    "trajectory"), we build the conditioner ourselves and assemble the
    ConditionedDiffusionModelWrapper manually, bypassing AudioX's factory
    for the conditioning part only.
    """
    factory_config = copy.deepcopy(config)

    # model_type: 'robot_diffusion' -> AudioX's 'diffusion_cond'
    if factory_config.get("model_type") in ("robot_diffusion", "robot_finetune"):
        factory_config["model_type"] = "diffusion_cond"

    # AudioX factory reads io_channels from config["model"]
    model_cfg = factory_config.setdefault("model", {})
    diff_cfg = model_cfg.get("diffusion", {}).get("config", {})
    if "io_channels" not in model_cfg:
        model_cfg["io_channels"] = diff_cfg.get("io_channels", config.get("action_dim"))

    # Audio-specific fields at top level (robotics defaults)
    factory_config.setdefault("sample_rate", 1)
    factory_config.setdefault("sample_size", config.get("action_chunk_size", 50))

    # Check whether we have robot-specific conditioner types
    cond_config = model_cfg.get("conditioning", None)
    has_robot_cond = False
    if cond_config:
        for c in cond_config.get("configs", []):
            if c["type"] == "trajectory":
                has_robot_cond = True
                break

    if has_robot_cond:
        # Build manually: DiT backbone via AudioX, conditioner via our factory
        model = _build_robot_diffusion_model(factory_config)
    else:
        # Pure AudioX — no robot-specific conditioners
        model = create_model_from_config(factory_config)

    model._config = config
    return model


def _build_robot_diffusion_model(config):
    """Assemble a ConditionedDiffusionModelWrapper with robot conditioners."""
    import numpy as np

    model_config = config["model"]
    diffusion_config = model_config["diffusion"]
    diffusion_model_config = copy.deepcopy(diffusion_config["config"])

    # Remove keys that are ours, not DiT's
    diffusion_model_config.pop("video_fps", None)

    io_channels = model_config["io_channels"]
    sample_rate = config.get("sample_rate", 1)
    diffusion_objective = diffusion_config.get("diffusion_objective", "v")

    # Build DiT backbone
    diffusion_model_type = diffusion_config["type"]
    if diffusion_model_type == "dit":
        diffusion_model = DiTWrapper(**diffusion_model_config)
    else:
        raise NotImplementedError(f"Unsupported diffusion type: {diffusion_model_type}")

    min_input_length = diffusion_model.model.patch_size

    # Build conditioner with our extended factory
    conditioning_config = model_config.get("conditioning", None)
    conditioner = None
    if conditioning_config is not None:
        conditioner = create_robot_conditioner_from_config(conditioning_config)

    cross_attention_ids = diffusion_config.get("cross_attention_cond_ids", [])
    global_cond_ids = diffusion_config.get("global_cond_ids", [])
    input_concat_ids = diffusion_config.get("input_concat_ids", [])
    prepend_cond_ids = diffusion_config.get("prepend_cond_ids", [])

    return ConditionedDiffusionModelWrapper(
        diffusion_model,
        conditioner,
        min_input_length=min_input_length,
        sample_rate=sample_rate,
        cross_attn_cond_ids=cross_attention_ids,
        global_cond_ids=global_cond_ids,
        input_concat_ids=input_concat_ids,
        prepend_cond_ids=prepend_cond_ids,
        pretransform=None,
        io_channels=io_channels,
        diffusion_objective=diffusion_objective,
    )

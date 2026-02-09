"""Robot model factory — thin wrapper around stable_audio_tools model creation."""

from stable_audio_tools.models.factory import create_model_from_config


def create_robot_model_from_config(config):
    """Create a diffusion model from a robot config dict.

    This is a thin wrapper around stable_audio_tools' create_model_from_config.
    The config must follow the AudioX model config schema with model_type,
    action_dim, model.diffusion, and model.conditioning sections.

    Args:
        config: dict loaded from a robotx_*.json config file.

    Returns:
        nn.Module: The complete AudioX diffusion model (DiT + MultiConditioner).
    """
    model = create_model_from_config(config)

    # Attach config for later reference (e.g., sample_size, sample_rate)
    model._config = config

    return model

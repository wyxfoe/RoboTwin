"""Robot model factory — uses AudioX's create_model_from_config directly.

AudioX is a 'diffusion_cond' model in stable_audio_tools.  Our JSON configs
use model_type='robot_diffusion' as a marker; this function remaps it to the
AudioX-native 'diffusion_cond' type before calling the factory.
"""

import os
import sys

# Reuse the same AUDIOX_PATH resolution logic as audiox_model.py so that
# the *local* AudioX source (not the pip package) is found first.
_policy_dir = os.path.dirname(os.path.abspath(__file__))
_audiox_search_paths = [
    os.environ.get("AUDIOX_PATH", ""),
    os.path.join(_policy_dir, "../../../../AudioX-"),
    os.path.join(_policy_dir, "../../../AudioX-"),
    os.path.join(_policy_dir, "../../AudioX-"),
]
for _p in _audiox_search_paths:
    if not _p:
        continue
    _p = os.path.abspath(_p)
    if os.path.isdir(os.path.join(_p, "stable_audio_tools")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
        break

from stable_audio_tools.models.factory import create_model_from_config


def create_robot_model_from_config(config):
    """Create an AudioX diffusion model from a robotics config dict.

    AudioX factory expects certain fields at the top level (io_channels,
    sample_rate, sample_size) that our robotics config stores inside
    model.diffusion.config.  This function copies them up.
    """
    import copy
    factory_config = copy.deepcopy(config)

    # model_type: 'robot_diffusion' -> AudioX's 'diffusion_cond'
    if factory_config.get("model_type") == "robot_diffusion":
        factory_config["model_type"] = "diffusion_cond"

    # AudioX factory reads io_channels from the top level
    diff_cfg = factory_config.get("model", {}).get("diffusion", {}).get("config", {})
    factory_config.setdefault("io_channels", diff_cfg.get("io_channels"))

    # Audio-specific fields (robotics defaults)
    factory_config.setdefault("sample_rate", 1)
    factory_config.setdefault("sample_size", config.get("action_chunk_size", 50))

    model = create_model_from_config(factory_config)
    model._config = config
    return model

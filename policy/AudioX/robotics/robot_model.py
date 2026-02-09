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

    Our JSON uses a nested layout for readability::

        model.diffusion.type / model.diffusion.config / model.conditioning

    AudioX's factory expects a flat layout::

        model_type = "diffusion_cond"
        model.type / model.config / model.conditioning

    This function bridges the two.
    """
    import copy
    factory_config = copy.deepcopy(config)

    # 1. model_type: 'robot_diffusion' -> AudioX's 'diffusion_cond'
    if factory_config.get("model_type") == "robot_diffusion":
        factory_config["model_type"] = "diffusion_cond"

    # 2. Flatten model.diffusion.* up into model.*
    model_section = factory_config.get("model", {})
    diffusion_section = model_section.pop("diffusion", None)
    if diffusion_section is not None:
        model_section.update(diffusion_section)
        factory_config["model"] = model_section

    # 3. Add AudioX audio fields if missing
    chunk_size = config.get("action_chunk_size", 50)
    factory_config.setdefault("sample_rate", 1)
    factory_config.setdefault("sample_size", chunk_size)

    model = create_model_from_config(factory_config)
    model._config = config
    return model

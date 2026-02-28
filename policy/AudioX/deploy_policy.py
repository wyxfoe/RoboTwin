"""
AudioX Policy Adapter for RoboTwin Evaluation.

This module adapts the fine-tuned AudioX (Diffusion Transformer) model
to the RoboTwin evaluation interface. It provides the three required
functions: get_model(), eval(), and reset_model().

AudioX is originally an audio generation DiT model that has been
fine-tuned to generate robot action sequences conditioned on
camera observations, language instructions, and robot state.
"""

import os
import sys
import numpy as np

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from audiox_model import AudioXRobot


def _resolve_path(path):
    """Resolve a path relative to the policy directory if not absolute."""
    if path is not None and not os.path.isabs(path):
        return os.path.join(parent_directory, path)
    return path


def encode_obs(observation):
    """
    Post-process RoboTwin observation into format for AudioX.

    Extracts camera RGB images and robot joint state from the
    RoboTwin observation dictionary.

    Camera order MUST match training: [front_camera, head_camera, left_camera, right_camera].
    If front_camera is not available (e.g. some env configs), use head_camera as fallback.

    Args:
        observation: RoboTwin observation dict with structure:
            {
                "observation": {
                    "front_camera": {"rgb": np.ndarray},  # optional
                    "head_camera": {"rgb": np.ndarray},
                    "left_camera": {"rgb": np.ndarray},
                    "right_camera": {"rgb": np.ndarray},
                },
                "joint_action": {
                    "left_arm": [N floats],
                    "left_gripper": float,
                    "right_arm": [N floats],
                    "right_gripper": float,
                    "vector": [action_dim floats],
                }
            }

    Returns:
        input_rgb_arr: List of 4 camera images [front, head, left, right].
        input_state: Joint state vector.
    """
    obs = observation["observation"]
    front_cam = obs.get("front_camera", {}).get("rgb")
    if front_cam is None:
        front_cam = obs["head_camera"]["rgb"]  # fallback
    input_rgb_arr = [
        front_cam,
        obs["head_camera"]["rgb"],
        obs["left_camera"]["rgb"],
        obs["right_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state


def get_model(usr_args):
    """
    Create and return the AudioX robot model.

    Args:
        usr_args: Dict from deploy_policy.yml merged with CLI overrides.
            Required keys:
                - model_config: Path to AudioX model config JSON.
                - ckpt_path: Path to fine-tuned checkpoint.
            Optional keys:
                - pretrained_name: HuggingFace pretrained name (alternative).
                - action_dim: Action dimension (default 16).
                - action_chunk_size: Steps per inference (default 50).
                - left_arm_dim: Left arm DOFs (default 6, set by eval framework).
                - right_arm_dim: Right arm DOFs (default 6, set by eval framework).
                - img_size: Input image size (default 224).
                - use_half: Use fp16 (default False).
                - action_stats_path: Path to action normalization stats.
                - action_head_path: Path to pre-trained ActionHead weights.

    Returns:
        AudioXRobot model instance.
    """
    model_config = _resolve_path(usr_args.get("model_config", None))
    ckpt_path = _resolve_path(usr_args.get("ckpt_path", None))
    pretrained_name = usr_args.get("pretrained_name", None)
    action_stats_path = _resolve_path(usr_args.get("action_stats_path", None))
    action_head_path = _resolve_path(usr_args.get("action_head_path", None))

    action_dim = usr_args.get("action_dim", 16)
    action_chunk_size = usr_args.get("action_chunk_size", 50)

    # Arm dimensions from the eval framework (eval_policy.py sets these from embodiment config)
    left_arm_dim = usr_args.get("left_arm_dim", 6)
    right_arm_dim = usr_args.get("right_arm_dim", 6)

    img_size_val = usr_args.get("img_size", 224)
    if isinstance(img_size_val, int):
        img_size = (img_size_val, img_size_val)
    else:
        img_size = tuple(img_size_val)

    use_half = usr_args.get("use_half", False)

    model = AudioXRobot(
        model_config_path=model_config,
        ckpt_path=ckpt_path,
        action_dim=action_dim,
        action_chunk_size=action_chunk_size,
        left_arm_dim=left_arm_dim,
        right_arm_dim=right_arm_dim,
        img_size=img_size,
        device="cuda",
        use_half=use_half,
        pretrained_name=pretrained_name,
        action_stats_path=action_stats_path,
        action_head_path=action_head_path,
    )

    return model


def eval(TASK_ENV, model, observation):
    """
    Run one evaluation step: observe -> predict actions -> execute.

    This function is called in a loop by the RoboTwin evaluation framework.
    Each call processes the current observation, generates an action chunk
    via the AudioX diffusion model, and executes each action step.

    Args:
        TASK_ENV: RoboTwin task environment instance.
        model: AudioXRobot model instance.
        observation: Current observation from TASK_ENV.get_obs().
    """
    # Set language instruction on first call (observation_window is None)
    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    # Encode observation
    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    # Generate action chunk via diffusion
    actions = model.get_action()[:model.action_chunk_size]

    # Execute each action step
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)


def reset_model(model):
    """
    Reset model state between evaluation episodes.

    Clears observation window, instruction cache, and CUDA memory.
    Called at the start of each new evaluation episode.

    Args:
        model: AudioXRobot model instance.
    """
    model.reset()

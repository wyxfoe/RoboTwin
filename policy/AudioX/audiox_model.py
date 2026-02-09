"""
AudioX Model Wrapper for RoboTwin Evaluation.

Wraps the fine-tuned AudioX (Diffusion Transformer) model for robot action generation.
AudioX has been adapted from its original audio generation architecture to output
robot joint actions conditioned on camera images, language instructions, and robot state.
"""

import os
import sys
import json
import numpy as np
import torch
import cv2
from PIL import Image

# Add AudioX- directory to Python path to find stable_audio_tools
current_file_path = os.path.abspath(__file__)
policy_dir = os.path.dirname(current_file_path)
# AudioX- is typically at the same level as RoboTwin
# Try multiple possible paths
possible_audiox_paths = [
    os.path.join(policy_dir, "../../../AudioX-"),  # From RoboTwin/policy/AudioX to AudioX-
    os.path.join(policy_dir, "../../AudioX-"),    # Alternative path
    "/home/wyx/Workspace/iros2026/AudioX-",        # Absolute path
]

for audiox_path in possible_audiox_paths:
    audiox_path = os.path.abspath(audiox_path)
    if os.path.exists(audiox_path) and os.path.exists(os.path.join(audiox_path, "stable_audio_tools")):
        if audiox_path not in sys.path:
            sys.path.insert(0, audiox_path)
        break
else:
    raise ImportError(
        f"Could not find AudioX- directory. Tried paths: {possible_audiox_paths}\n"
        "Please ensure AudioX- is installed or add its path to PYTHONPATH."
    )

from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.pretrained import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond


class AudioXRobot:
    """
    AudioX model adapted for robot action generation in RoboTwin.

    The model uses a Diffusion Transformer (DiT) backbone to generate
    robot action sequences conditioned on:
      - Camera images (head, left wrist, right wrist) via CLIP vision encoder
      - Language instructions via T5 text encoder
      - Robot joint state as additional conditioning

    Action output: [left_arm(7) + left_gripper(1) + right_arm(7) + right_gripper(1)] = 16 dims
    """

    def __init__(
        self,
        model_config_path,
        ckpt_path,
        action_dim=16,
        action_chunk_size=50,
        img_size=(224, 224),
        device="cuda",
        use_half=False,
        pretrained_name=None,
        action_stats_path=None,
    ):
        """
        Args:
            model_config_path: Path to AudioX model config JSON file.
            ckpt_path: Path to fine-tuned checkpoint (.ckpt or .safetensors).
            action_dim: Robot action dimension (default 16 for dual-arm).
            action_chunk_size: Number of action steps to predict per inference.
            img_size: Input image size for vision encoder.
            device: Device to run inference on.
            use_half: Whether to use fp16 inference.
            pretrained_name: HuggingFace pretrained model name (alternative to config+ckpt).
            action_stats_path: Path to action normalization stats (.pt file from training).
        """
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.img_size = img_size
        self.device = device
        self.use_half = use_half

        # Load model
        if pretrained_name is not None:
            self.model, self.model_config = get_pretrained_model(pretrained_name)
        else:
            with open(model_config_path, "r") as f:
                self.model_config = json.load(f)
            self.model = create_model_from_config(self.model_config)
            if ckpt_path is not None:
                self._load_checkpoint(ckpt_path)

        self.model = self.model.to(self.device)
        if self.use_half:
            self.model = self.model.half()
        self.model.eval()

        # Observation state
        self.observation_window = None
        self.instruction = None

        # Extract sample rate and sample size from config if available
        self.sample_rate = self.model_config.get("sample_rate", 1)
        self.sample_size = self.model_config.get("sample_size", self.action_chunk_size * self.action_dim)

        # Load action normalization statistics for denormalization
        self.action_stats = None
        if action_stats_path is not None and os.path.exists(action_stats_path):
            self.action_stats = torch.load(action_stats_path, map_location="cpu")
            print(f"[AudioX] Loaded action stats from {action_stats_path}")

        print(f"[AudioX] Model loaded successfully on {self.device}")
        print(f"[AudioX] Action dim: {self.action_dim}, Chunk size: {self.action_chunk_size}")

    def _load_checkpoint(self, ckpt_path):
        """Load model weights from checkpoint file."""
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(ckpt_path)
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]

        self.model.load_state_dict(state_dict, strict=False)
        print(f"[AudioX] Checkpoint loaded from {ckpt_path}")

    def set_language(self, instruction):
        """Set language instruction for current episode."""
        self.instruction = instruction

    def update_observation_window(self, img_arr, state):
        """
        Update the observation window with current camera images and robot state.

        Args:
            img_arr: List of 3 camera images [head, right, left] as numpy arrays (H, W, 3).
            state: Robot joint state vector (16-dim).
        """
        img_head = img_arr[0]
        img_right = img_arr[1]
        img_left = img_arr[2]

        # Resize images to target size
        img_head = self._preprocess_image(img_head)
        img_right = self._preprocess_image(img_right)
        img_left = self._preprocess_image(img_left)

        self.observation_window = {
            "images": {
                "head": img_head,
                "right_wrist": img_right,
                "left_wrist": img_left,
            },
            "state": np.array(state, dtype=np.float32),
            "instruction": self.instruction,
        }

    def _preprocess_image(self, img):
        """Preprocess a single image: resize and normalize."""
        if img.shape[:2] != self.img_size:
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
        return img

    def _prepare_video_conditioning(self, images_dict):
        """
        Prepare video conditioning input for AudioX's CLIP conditioner.

        Converts camera images to video tensor format expected by AudioX.
        The CLIP conditioner expects video frames as tensors.
        """
        frames = []
        for key in ["head", "left_wrist", "right_wrist"]:
            img = images_dict[key]
            # Convert BGR to RGB if needed, then to tensor
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
            frames.append(img_tensor)

        # Stack frames as a video sequence: (num_frames, C, H, W)
        video_tensor = torch.stack(frames, dim=0)
        return video_tensor

    def _prepare_conditioning(self):
        """
        Prepare all conditioning inputs for the AudioX model.

        Returns conditioning dict compatible with AudioX's MultiConditioner.
        """
        assert self.observation_window is not None, "Must call update_observation_window first!"

        conditioning = {}

        # Text conditioning (language instruction)
        if self.instruction is not None:
            conditioning["prompt"] = self.instruction

        # Video conditioning (camera images as video frames)
        video_tensor = self._prepare_video_conditioning(self.observation_window["images"])
        conditioning["video"] = video_tensor

        # State conditioning (robot joint state)
        state = self.observation_window["state"]
        conditioning["state"] = state

        return conditioning

    @torch.no_grad()
    def get_action(self):
        """
        Generate robot actions using AudioX diffusion inference.

        Returns:
            actions: numpy array of shape (action_chunk_size, action_dim)
        """
        assert self.observation_window is not None, "Must call update_observation_window first!"

        conditioning = self._prepare_conditioning()

        # Prepare conditioning for AudioX's generate_diffusion_cond
        conditioning_list = [conditioning]

        # Generate actions through diffusion process
        output = generate_diffusion_cond(
            self.model,
            conditioning=conditioning_list,
            sample_size=self.sample_size,
            sample_rate=self.sample_rate,
            device=self.device,
            seed=None,
        )

        # Post-process: reshape output to action sequence
        # Output from diffusion: (batch, channels, length)
        if isinstance(output, torch.Tensor):
            actions = output.squeeze(0).cpu().numpy()
        else:
            actions = np.array(output)

        # Reshape to (chunk_size, action_dim)
        if actions.ndim == 1:
            actions = actions.reshape(-1, self.action_dim)
        elif actions.ndim == 2 and actions.shape[0] != self.action_chunk_size:
            # Transpose if needed (channels, length) -> (length, channels)
            if actions.shape[0] == self.action_dim:
                actions = actions.T
            actions = actions[:self.action_chunk_size]

        actions = actions[:self.action_chunk_size]

        # Denormalize actions from normalized space back to real joint angles
        if self.action_stats is not None:
            actions = self._denormalize_actions(actions)

        return actions

    def _denormalize_actions(self, actions):
        """
        Denormalize actions from [-1, 1] back to real joint angle space.

        Supports two normalization conventions:
          - min/max stats: action = (normalized + 1) / 2 * (max - min) + min
          - mean/std stats: action = normalized * std + mean

        Args:
            actions: numpy array of shape (chunk_size, action_dim) in normalized space.

        Returns:
            Denormalized actions in real joint angle space.
        """
        stats = self.action_stats
        if "action_min" in stats and "action_max" in stats:
            a_min = stats["action_min"].numpy()
            a_max = stats["action_max"].numpy()
            # Ensure broadcastable to (chunk_size, action_dim)
            if a_min.ndim == 1:
                a_min = a_min[:self.action_dim]
                a_max = a_max[:self.action_dim]
            actions = (actions + 1.0) / 2.0 * (a_max - a_min) + a_min
        elif "action_mean" in stats and "action_std" in stats:
            a_mean = stats["action_mean"].numpy()
            a_std = stats["action_std"].numpy()
            if a_mean.ndim == 1:
                a_mean = a_mean[:self.action_dim]
                a_std = a_std[:self.action_dim]
            actions = actions * a_std + a_mean
        return actions

    def reset(self):
        """Reset model state between evaluation episodes."""
        self.observation_window = None
        self.instruction = None
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[AudioX] Model state reset")

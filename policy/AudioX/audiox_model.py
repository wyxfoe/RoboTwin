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
import torch.nn as nn
import cv2
from PIL import Image

# Add AudioX source directory to Python path to find stable_audio_tools.
# Resolution order:
#   1. AUDIOX_PATH environment variable (explicit override)
#   2. ../../../AudioX-  (sibling to RoboTwin project root)
#   3. ../../AudioX-     (alternative layout)
current_file_path = os.path.abspath(__file__)
policy_dir = os.path.dirname(current_file_path)

# Ensure local robotics package is importable
if policy_dir not in sys.path:
    sys.path.insert(0, policy_dir)

_audiox_search_paths = [
    os.environ.get("AUDIOX_PATH", ""),
    os.path.join(policy_dir, "../../../AudioX-"),
    os.path.join(policy_dir, "../../AudioX-"),
]

for _p in _audiox_search_paths:
    if not _p:
        continue
    _p = os.path.abspath(_p)
    if os.path.isdir(os.path.join(_p, "stable_audio_tools")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
        break
else:
    raise ImportError(
        "Cannot find AudioX source (stable_audio_tools).\n"
        "Set the AUDIOX_PATH environment variable or place AudioX- beside the RoboTwin directory.\n"
        f"Searched: {[os.path.abspath(p) for p in _audiox_search_paths if p]}"
    )

from stable_audio_tools.models.pretrained import get_pretrained_model
from stable_audio_tools.inference.sampling import sample

# Use our robotics adapter that remaps config keys for AudioX factory
from robotics.robot_model import create_robot_model_from_config


# CLIP-ViT-B/32 normalization constants
CLIP_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]


class ActionHead(nn.Module):
    """
    Output head for projecting diffusion output to robot action space.

    Mirrors the FinalLayer design in RDT (RmsNorm -> MLP) and DexVLA
    (AdaLN -> Linear). Uses residual connection with zero-initialized
    output layer so the head acts as identity when not fine-tuned.
    """

    def __init__(self, action_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = action_dim * 4
        self.norm = nn.LayerNorm(action_dim)
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, action_dim),
        )
        # Zero-initialize the last layer so residual output = identity at init
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        """
        Args:
            x: (chunk_size, action_dim) raw action predictions

        Returns:
            Refined action predictions of same shape.
        """
        return x + self.mlp(self.norm(x))


class AudioXRobot:
    """
    AudioX model adapted for robot action generation in RoboTwin.

    The model uses a Diffusion Transformer (DiT) backbone to generate
    robot action sequences conditioned on:
      - Camera images (head, left wrist, right wrist) via CLIP vision encoder
      - Language instructions via T5 text encoder
      - Robot joint state as additional conditioning

    Action output: [left_arm(N) + left_gripper(1) + right_arm(N) + right_gripper(1)]
    """

    def __init__(
        self,
        model_config_path,
        ckpt_path,
        action_dim=16,
        action_chunk_size=50,
        left_arm_dim=6,
        right_arm_dim=6,
        img_size=(224, 224),
        device="cuda",
        use_half=False,
        pretrained_name=None,
        action_stats_path=None,
        action_head_path=None,
    ):
        """
        Args:
            model_config_path: Path to AudioX model config JSON file.
            ckpt_path: Path to fine-tuned checkpoint (.ckpt or .safetensors).
            action_dim: Robot action dimension (default 16 for dual-arm).
            action_chunk_size: Number of action steps to predict per inference.
            left_arm_dim: Number of joint DOFs for the left arm (excluding gripper).
            right_arm_dim: Number of joint DOFs for the right arm (excluding gripper).
            img_size: Input image size for vision encoder.
            device: Device to run inference on.
            use_half: Whether to use fp16 inference.
            pretrained_name: HuggingFace pretrained model name (alternative to config+ckpt).
            action_stats_path: Path to action normalization stats (.pt file from training).
            action_head_path: Path to pre-trained ActionHead weights (.pt).
        """
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.left_arm_dim = left_arm_dim
        self.right_arm_dim = right_arm_dim
        self.img_size = img_size
        self.device = device
        self.use_half = use_half

        # Gripper dimension indices within the action vector
        # Layout: [left_arm(N), left_gripper(1), right_arm(N), right_gripper(1)]
        self.left_gripper_idx = left_arm_dim
        self.right_gripper_idx = left_arm_dim + 1 + right_arm_dim

        # Load model
        if pretrained_name is not None:
            self.model, self.model_config = get_pretrained_model(pretrained_name)
        else:
            with open(model_config_path, "r") as f:
                self.model_config = json.load(f)
            self.model = create_robot_model_from_config(self.model_config)
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
        # sample_size = chunk_size (NOT chunk_size * action_dim)
        # AudioX generates (batch, io_channels, sample_size) where io_channels = action_dim
        self.sample_rate = self.model_config.get("sample_rate", 1)
        self.sample_size = self.model_config.get("sample_size", self.action_chunk_size)

        # Load action normalization statistics for denormalization
        self.action_stats = None
        if action_stats_path is not None and os.path.exists(action_stats_path):
            self.action_stats = torch.load(action_stats_path, map_location="cpu")
            print(f"[AudioX] Loaded action stats from {action_stats_path}")

        # Initialize ActionHead (LayerNorm + MLP) for output projection
        self.action_head = ActionHead(action_dim).to(self.device)
        if action_head_path is not None and os.path.exists(action_head_path):
            head_state = torch.load(action_head_path, map_location=self.device)
            self.action_head.load_state_dict(head_state)
            print(f"[AudioX] Loaded action head from {action_head_path}")
        else:
            print("[AudioX] ActionHead initialized (identity mode, no pre-trained weights)")
        self.action_head.eval()

        # Precompute CLIP normalization tensors
        self._clip_mean = torch.tensor(CLIP_IMAGE_MEAN).view(3, 1, 1)
        self._clip_std = torch.tensor(CLIP_IMAGE_STD).view(3, 1, 1)

        print(f"[AudioX] Model loaded successfully on {self.device}")
        print(f"[AudioX] Action dim: {self.action_dim}, Chunk size: {self.action_chunk_size}")
        print(f"[AudioX] Arm config: left={self.left_arm_dim}, right={self.right_arm_dim}")

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
            state: Robot joint state vector.
        """
        img_head = self._preprocess_image(img_arr[0])
        img_right = self._preprocess_image(img_arr[1])
        img_left = self._preprocess_image(img_arr[2])

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
        """Preprocess a single image: resize to target size."""
        if img.shape[:2] != self.img_size:
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
        return img

    def _preprocess_camera_image(self, img):
        """Convert a raw camera image (H,W,3 uint8 or float) to CLIP-normalized (C,H,W) tensor."""
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
        img_tensor = (img_tensor - self._clip_mean) / self._clip_std
        return img_tensor

    @torch.no_grad()
    def _encode_images_clip(self, images_dict):
        """Encode camera images through CLIP independently, concatenate features.

        RDT-style: each camera view is processed through CLIP individually,
        then all patch token sequences are concatenated along dim=1.
        No temporal transformer — preserves full per-view spatial information.

        Returns:
            [concat_features, mask] matching AudioX conditioner output format.
        """
        clip_cond = self.model.conditioner.conditioners["video"]
        clip_model = clip_cond.visual_encoder_model

        all_features = []
        for key in ["head", "left_wrist", "right_wrist"]:
            img = images_dict[key]
            img_tensor = self._preprocess_camera_image(img)
            # (1, C, H, W) for single image batch
            img_batch = img_tensor.unsqueeze(0).to(self.device)

            outputs = clip_model(pixel_values=img_batch)
            # last_hidden_state: (1, num_patches+1, hidden_dim)
            all_features.append(outputs.last_hidden_state)

        # Concatenate all cameras: (1, num_cams * num_tokens, hidden_dim)
        concat_features = torch.cat(all_features, dim=1)
        mask = torch.ones(1, concat_features.shape[1]).to(self.device)

        return [concat_features, mask]

    @torch.no_grad()
    def _build_conditioning(self):
        """Build conditioning with RDT-style multi-view concatenation.

        - T5, trajectory: processed via AudioX conditioners as usual
        - Video: each camera independently through CLIP, then token concatenation
        """
        assert self.observation_window is not None, "Must call update_observation_window first!"

        conditioner = self.model.conditioner
        conditioning = {}

        # Process non-video conditioners normally
        metadata = [{
            "prompt": self.instruction or "",
            "proprio": self.observation_window["state"],
        }]
        for key, cond_module in conditioner.conditioners.items():
            if key == "video":
                continue
            inputs = [metadata[0][key]]
            conditioning[key] = cond_module(inputs, self.device)

        # RDT-style: independent CLIP per camera, concatenate tokens
        conditioning["video"] = self._encode_images_clip(self.observation_window["images"])

        return conditioning

    @torch.no_grad()
    def get_action(self):
        """
        Generate robot actions using AudioX diffusion inference.

        Pipeline:
          1. Build conditioning (RDT-style multi-view CLIP + T5 + proprio)
          2. Run diffusion sampling → raw output (batch, action_dim, chunk_size)
          3. Transpose to robot convention (chunk_size, action_dim)
          4. Apply ActionHead (LayerNorm + MLP) for action-space projection
          5. Denormalize from training normalization back to real joint angles
          6. Clamp gripper values to valid range [0, 1]

        Returns:
            actions: numpy array of shape (action_chunk_size, action_dim)
        """
        assert self.observation_window is not None, "Must call update_observation_window first!"

        # Step 1: Build conditioning (bypasses CLIPConditioner temporal transformer)
        conditioning = self._build_conditioning()
        cond_inputs = self.model.get_conditioning_inputs(conditioning)

        # Step 2: Diffusion sampling
        noise = torch.randn(1, self.action_dim, self.action_chunk_size).to(self.device)

        if self.model.diffusion_objective == "v":
            output = sample(
                self.model.model, noise, steps=50, starting_t=0,
                **cond_inputs, cfg_scale=3.0, batch_cfg=True,
            )
        else:
            from stable_audio_tools.inference.sampling import sample_discrete_euler
            output = sample_discrete_euler(
                self.model.model, noise, steps=50,
                **cond_inputs, cfg_scale=3.0, batch_cfg=True,
            )

        # Step 3: Transpose (action_dim, chunk_size) → (chunk_size, action_dim)
        actions = output.squeeze(0).T  # (chunk_size, action_dim)
        actions = actions[:self.action_chunk_size]

        # Step 4: Apply ActionHead
        actions = actions.to(self.device)
        actions = self.action_head(actions)

        # Step 5: Denormalize
        if self.action_stats is not None:
            actions = self._denormalize_actions(actions)

        # Step 6: Clamp gripper values
        actions = self._clamp_gripper(actions)

        return actions.cpu().numpy()

    def _denormalize_actions(self, actions):
        """
        Denormalize actions from [-1, 1] back to real joint angle space.

        Supports two normalization conventions:
          - min/max stats: action = (normalized + 1) / 2 * (max - min) + min
          - mean/std stats: action = normalized * std + mean

        Args:
            actions: tensor of shape (chunk_size, action_dim) in normalized space.

        Returns:
            Denormalized actions tensor in real joint angle space.
        """
        stats = self.action_stats
        if "action_min" in stats and "action_max" in stats:
            a_min = stats["action_min"][:self.action_dim].to(actions.device)
            a_max = stats["action_max"][:self.action_dim].to(actions.device)
            actions = (actions + 1.0) / 2.0 * (a_max - a_min) + a_min
        elif "action_mean" in stats and "action_std" in stats:
            a_mean = stats["action_mean"][:self.action_dim].to(actions.device)
            a_std = stats["action_std"][:self.action_dim].to(actions.device)
            actions = actions * a_std + a_mean
        return actions

    def _clamp_gripper(self, actions):
        """
        Clamp gripper dimensions to valid range [0, 1].

        Joint angles are unconstrained, but gripper open/close values
        must be within [0, 1] for the robot hardware.

        Args:
            actions: tensor of shape (chunk_size, action_dim)

        Returns:
            Actions with gripper values clamped.
        """
        actions[:, self.left_gripper_idx] = actions[:, self.left_gripper_idx].clamp(0.0, 1.0)
        actions[:, self.right_gripper_idx] = actions[:, self.right_gripper_idx].clamp(0.0, 1.0)
        return actions

    def reset(self):
        """Reset model state between evaluation episodes."""
        self.observation_window = None
        self.instruction = None
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[AudioX] Model state reset")

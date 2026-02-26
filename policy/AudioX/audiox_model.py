"""
AudioX Model Wrapper for RoboTwin Evaluation.

Wraps the fine-tuned AudioX (Diffusion Transformer) model for robot action generation.
AudioX has been adapted from its original audio generation architecture to output
robot joint actions conditioned on camera images, language instructions, and robot state.

Supports two modes:
  1. Direct mode (original): io_channels == action_dim, DiT operates in action space
  2. Fine-tune mode (1.2B):  io_channels == 64, DiT operates in latent space,
     adapter layers (input_proj / output_proj) bridge action ↔ latent spaces
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image

# Ensure local robotics package is importable
current_file_path = os.path.abspath(__file__)
policy_dir = os.path.dirname(current_file_path)
if policy_dir not in sys.path:
    sys.path.insert(0, policy_dir)

# Official AudioX package (pip install -e /path/to/AudioX  or  pip install git+https://github.com/ZeyueT/AudioX.git)
from audiox.models.pretrained import get_pretrained_model
from audiox.inference.sampling import sample, sample_discrete_euler

# Our robotics adapters
from robotics.robot_model import create_robot_model_from_config
from robotics.finetune_model import AudioXFineTuneModel, create_finetune_model


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

        # Load model — detect fine-tune mode from config
        if pretrained_name is not None:
            self.model, self.model_config = get_pretrained_model(pretrained_name)
            self.is_finetune = False
        else:
            with open(model_config_path, "r") as f:
                self.model_config = json.load(f)

            self.is_finetune = self.model_config.get("model_type") == "robot_finetune"

            if self.is_finetune:
                # Fine-tune mode: build AudioXFineTuneModel with adapter layers
                self.model = create_finetune_model(self.model_config, audiox_ckpt_path=None)
                self.latent_dim = self.model_config.get("latent_dim", 64)
                print(f"[AudioX] Fine-tune mode: action_dim={action_dim} → latent_dim={self.latent_dim}")
            else:
                # Direct mode: io_channels == action_dim
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
        self.sample_rate = self.model_config.get("sample_rate", 1)
        self.sample_size = self.model_config.get("sample_size", self.action_chunk_size)

        # Load action normalization statistics for denormalization
        self.action_stats = None
        if action_stats_path is not None and os.path.exists(action_stats_path):
            self.action_stats = torch.load(action_stats_path, map_location="cpu")
            print(f"[AudioX] Loaded action stats from {action_stats_path}")

        # Initialize ActionHead (LayerNorm + MLP) for output projection
        # In fine-tune mode the output_proj inside the model handles latent→action,
        # so ActionHead provides an optional additional refinement layer.
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
        print(f"[AudioX] Mode: {'fine-tune (latent)' if self.is_finetune else 'direct'}")
        print(f"[AudioX] Action dim: {self.action_dim}, Chunk size: {self.action_chunk_size}")
        print(f"[AudioX] Arm config: left={self.left_arm_dim}, right={self.right_arm_dim}")

    def _load_checkpoint(self, ckpt_path):
        """Load model weights from checkpoint file.

        Handles both direct-mode and fine-tune-mode checkpoints.
        Fine-tune checkpoints may have keys prefixed with 'diffusion.' from
        the Lightning training wrapper — these are stripped automatically.
        """
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(ckpt_path)
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]

        # Strip Lightning wrapper prefix if present
        # (RobotFineTuneTrainingWrapper stores model as self.diffusion)
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("diffusion."):
                cleaned[k[len("diffusion."):]] = v
            else:
                cleaned[k] = v

        self.model.load_state_dict(cleaned, strict=False)
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

        Note: AudioX's TrajectoryConditioner may not call proj_out internally,
        so we apply it here if the output dim doesn't match output_dim.
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

            # Fix for conditioners that don't call proj_out internally
            # (e.g. TrajectoryConditioner outputs dim instead of output_dim)
            features, mask = conditioning[key]
            if (hasattr(cond_module, 'proj_out') and
                    hasattr(cond_module, 'output_dim') and
                    features.shape[-1] != cond_module.output_dim):
                features = cond_module.proj_out(features)
                conditioning[key] = [features, mask]

        # RDT-style: independent CLIP per camera, concatenate tokens
        conditioning["video"] = self._encode_images_clip(self.observation_window["images"])

        return conditioning

    @torch.no_grad()
    def get_action(self):
        """
        Generate robot actions using AudioX diffusion inference.

        Direct mode pipeline:
          1. Build conditioning
          2. Diffusion sampling in action space → (B, action_dim, T)
          3. Transpose → (T, action_dim), apply ActionHead
          4. Denormalize + clamp gripper

        Fine-tune mode pipeline:
          1. Build conditioning
          2. Diffusion sampling in 64-dim latent space → (B, 64, T)
          3. output_proj: (B, 64, T) → (B, 14, T)
          4. Transpose → (T, 14), apply ActionHead
          5. Denormalize + clamp gripper

        Returns:
            actions: numpy array of shape (action_chunk_size, action_dim)
        """
        assert self.observation_window is not None, "Must call update_observation_window first!"

        # Step 1: Build conditioning
        conditioning = self._build_conditioning()
        cond_inputs = self.model.get_conditioning_inputs(conditioning)

        # Step 2: Diffusion sampling
        if self.is_finetune:
            # Sample in 64-dim latent space
            noise = torch.randn(1, self.latent_dim, self.action_chunk_size).to(self.device)
        else:
            # Sample directly in action space
            noise = torch.randn(1, self.action_dim, self.action_chunk_size).to(self.device)

        if self.model.diffusion_objective == "v":
            output = sample(
                self.model.model, noise, steps=50, eta=0,
                **cond_inputs, cfg_scale=3.0,
            )
        else:
            output = sample_discrete_euler(
                self.model.model, noise, steps=50,
                **cond_inputs, cfg_scale=3.0,
            )

        # Step 3: Decode latent → actions (fine-tune mode only)
        if self.is_finetune:
            output = self.model.decode_latent(output)  # (B, 64, T) → (B, 14, T)

        # Step 4: Transpose (action_dim, chunk_size) → (chunk_size, action_dim)
        actions = output.squeeze(0).T  # (chunk_size, action_dim)
        actions = actions[:self.action_chunk_size]

        # Step 5: Apply ActionHead
        actions = actions.to(self.device)
        actions = self.action_head(actions)

        # Step 6: Denormalize
        if self.action_stats is not None:
            actions = self._denormalize_actions(actions)

        # Step 7: Clamp gripper values
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

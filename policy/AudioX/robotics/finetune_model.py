"""AudioX 1.2B fine-tuning adapter for robot action generation.

Replaces AudioX's audio-specific interface layers while preserving the
pretrained DiT backbone (1536 embed, 24 depth, 24 heads):

    AudioX original (1.2B)                     Robot fine-tune
    ├── pretransform (VAE): wave→64d latent     → ActionInputProj:  Linear(14, 64)
    ├── DiT: 1536e, 24d, 24h                   → DiT: load pretrained weights
    ├── T5 conditioner                          → T5: load pretrained weights
    ├── CLIP conditioner                        → CLIP: load pretrained weights
    └── Audio conditioner                       → TrajectoryConditioner (MLP)

The diffusion process operates in the 64-dim latent space that AudioX was
pretrained on, so the DiT's learned generative prior transfers directly.
New adapter layers (input/output projections + trajectory conditioner)
are trained with normal LR while the DiT backbone uses a lower LR.
"""

import copy
import os
import sys

import torch
import torch.nn as nn

# Ensure AudioX source is importable
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


# ---------------------------------------------------------------------------
# Adapter layers
# ---------------------------------------------------------------------------

class ActionInputProjection(nn.Module):
    """Project robot actions (14-dim) into AudioX latent space (64-dim).

    Replaces AudioX's VAE pretransform.  Uses a linear layer followed by
    LayerNorm so the latent distribution roughly matches what the pretrained
    DiT expects (zero-mean, unit-variance).
    """

    def __init__(self, action_dim: int = 14, latent_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(action_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        """
        Args:
            x: (B, action_dim, T) — action sequence in AudioX channel-first format.
        Returns:
            (B, latent_dim, T) — projected into 64-dim latent space.
        """
        # Transpose to (B, T, action_dim), project, then back to (B, latent_dim, T)
        x = x.transpose(1, 2)              # (B, T, action_dim)
        x = self.norm(self.proj(x))         # (B, T, latent_dim)
        return x.transpose(1, 2)            # (B, latent_dim, T)


class ActionOutputProjection(nn.Module):
    """Project from AudioX latent space (64-dim) back to robot actions (14-dim).

    Applied after diffusion sampling to decode latent trajectories into
    joint angle commands.
    """

    def __init__(self, latent_dim: int = 64, action_dim: int = 14):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.proj = nn.Linear(latent_dim, action_dim)

    def forward(self, x):
        """
        Args:
            x: (B, latent_dim, T) — latent trajectory from diffusion.
        Returns:
            (B, action_dim, T) — decoded robot actions.
        """
        x = x.transpose(1, 2)              # (B, T, latent_dim)
        x = self.proj(self.norm(x))         # (B, T, action_dim)
        return x.transpose(1, 2)            # (B, action_dim, T)


# ---------------------------------------------------------------------------
# Fine-tune model wrapper
# ---------------------------------------------------------------------------

class AudioXFineTuneModel(nn.Module):
    """Wraps an AudioX diffusion_cond model with action-space adapter layers.

    Structure:
        action (B, 14, T)
          → input_proj  (B, 64, T)          [new, trainable]
          → DiT diffusion backbone           [pretrained, low LR]
          → output_proj (B, 14, T)          [new, trainable]

    Conditioning:
        - T5 text:          kept from AudioX pretrained
        - CLIP vision:      kept from AudioX pretrained
        - TrajectoryConditioner: new, trainable (replaces Audio conditioner)
    """

    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model
        self._config = config

        action_dim = config.get("action_dim", 14)
        latent_dim = config.get("latent_dim", 64)

        self.input_proj = ActionInputProjection(action_dim, latent_dim)
        self.output_proj = ActionOutputProjection(latent_dim, action_dim)

    # --- Delegate key attributes to base_model for compatibility ----------

    @property
    def model(self):
        return self.base_model.model

    @property
    def conditioner(self):
        return self.base_model.conditioner

    @property
    def diffusion_objective(self):
        return self.base_model.diffusion_objective

    @property
    def sample_rate(self):
        return getattr(self.base_model, "sample_rate", 1)

    @property
    def sample_size(self):
        return getattr(self.base_model, "sample_size", 50)

    def get_conditioning_inputs(self, conditioning):
        return self.base_model.get_conditioning_inputs(conditioning)

    # --- Action-space interface -------------------------------------------

    def encode_actions(self, actions):
        """Map (B, action_dim, T) to (B, latent_dim, T)."""
        return self.input_proj(actions)

    def decode_latent(self, latent):
        """Map (B, latent_dim, T) to (B, action_dim, T)."""
        return self.output_proj(latent)

    # --- Parameter groups for differential LR -----------------------------

    def adapter_parameters(self):
        """Parameters of newly added layers (trained at full LR)."""
        params = list(self.input_proj.parameters()) + list(self.output_proj.parameters())
        # Include the trajectory conditioner if it exists
        if hasattr(self.base_model, "conditioner"):
            conds = self.base_model.conditioner.conditioners
            if "proprio" in conds:
                params += list(conds["proprio"].parameters())
        return params

    def backbone_parameters(self):
        """DiT backbone parameters (trained at low LR or frozen)."""
        return list(self.base_model.model.parameters())

    def frozen_parameters(self):
        """T5 + CLIP conditioner parameters (frozen during fine-tuning)."""
        params = []
        if hasattr(self.base_model, "conditioner"):
            conds = self.base_model.conditioner.conditioners
            for key, cond in conds.items():
                if key in ("prompt", "video"):
                    params += list(cond.parameters())
        return params


# ---------------------------------------------------------------------------
# Weight loading utilities
# ---------------------------------------------------------------------------

def _filter_pretrained_weights(pretrained_state, model_state, skip_prefixes=None):
    """Filter pretrained weights to only load compatible keys.

    Skips keys that:
      - Don't exist in the target model
      - Have a shape mismatch (e.g., io_channels changed)
      - Match any of the skip_prefixes (e.g., pretransform, audio conditioner)

    Returns:
        Dict of loadable weights and a list of skipped key names.
    """
    skip_prefixes = skip_prefixes or []
    loadable = {}
    skipped = []

    for key, value in pretrained_state.items():
        # Skip explicitly excluded prefixes
        if any(key.startswith(p) for p in skip_prefixes):
            skipped.append(key)
            continue

        if key not in model_state:
            skipped.append(key)
            continue

        if value.shape != model_state[key].shape:
            skipped.append(key)
            continue

        loadable[key] = value

    return loadable, skipped


def load_audiox_pretrained(model, ckpt_path, skip_pretransform=True):
    """Load AudioX pretrained weights into the base model.

    Loads DiT backbone + T5 + CLIP conditioner weights from the AudioX
    checkpoint. Skips:
      - pretransform (VAE) — replaced by ActionInputProjection
      - Audio conditioner — replaced by TrajectoryConditioner
      - Keys with shape mismatches (e.g., if io_channels differs)

    Args:
        model: AudioXFineTuneModel instance.
        ckpt_path: Path to AudioX pretrained checkpoint (.ckpt, .pt, .safetensors).
        skip_pretransform: Whether to skip pretransform/VAE weights.

    Returns:
        (num_loaded, num_skipped) tuple.
    """
    # Load checkpoint
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        pretrained_state = load_file(ckpt_path, device="cpu")
    else:
        raw = torch.load(ckpt_path, map_location="cpu")
        if isinstance(raw, dict):
            if "state_dict" in raw:
                pretrained_state = raw["state_dict"]
            elif "model" in raw:
                pretrained_state = raw["model"]
            else:
                pretrained_state = raw
        else:
            pretrained_state = raw.state_dict()

    # Determine which prefixes to skip
    skip_prefixes = []
    if skip_pretransform:
        skip_prefixes.append("pretransform.")
        skip_prefixes.append("pretransform_")

    # Also skip the audio conditioner (replaced by trajectory)
    # AudioX audio conditioner keys typically look like:
    #   conditioner.conditioners.audio_input.*
    #   conditioner.conditioners.audio.*
    skip_prefixes.append("conditioner.conditioners.audio")

    model_state = model.base_model.state_dict()
    loadable, skipped = _filter_pretrained_weights(
        pretrained_state, model_state, skip_prefixes
    )

    # Load compatible weights
    result = model.base_model.load_state_dict(loadable, strict=False)
    missing = set(result.missing_keys) - set(skipped)

    print(f"[FineTune] Loaded {len(loadable)} pretrained weight tensors")
    print(f"[FineTune] Skipped {len(skipped)} tensors (pretransform/audio/mismatch)")
    if missing:
        print(f"[FineTune] {len(missing)} keys in model have no pretrained match")

    return len(loadable), len(skipped)


def freeze_for_finetune(model, finetune_config):
    """Apply freeze / LR configuration for fine-tuning.

    - T5 + CLIP conditioners: always frozen
    - DiT backbone: frozen if finetune_config["freeze_dit"] is True
    - Adapter layers (input_proj, output_proj, trajectory): always trainable

    Args:
        model: AudioXFineTuneModel instance.
        finetune_config: Dict with keys freeze_dit, dit_lr_scale, etc.
    """
    # Freeze T5 + CLIP conditioners
    for param in model.frozen_parameters():
        param.requires_grad = False

    # DiT backbone
    freeze_dit = finetune_config.get("freeze_dit", False)
    if freeze_dit:
        for param in model.backbone_parameters():
            param.requires_grad = False
        print("[FineTune] DiT backbone: FROZEN")
    else:
        for param in model.backbone_parameters():
            param.requires_grad = True
        lr_scale = finetune_config.get("dit_lr_scale", 0.1)
        print(f"[FineTune] DiT backbone: trainable (LR scale={lr_scale}x)")

    # Adapter layers always trainable
    for param in model.adapter_parameters():
        param.requires_grad = True
    print("[FineTune] Adapter layers (input/output proj + trajectory): trainable")


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_finetune_model(config, audiox_ckpt_path=None):
    """Create an AudioXFineTuneModel from config, optionally loading pretrained weights.

    Steps:
      1. Build AudioX diffusion_cond model from config (with io_channels=64)
      2. Wrap with ActionInputProjection + ActionOutputProjection
      3. Load AudioX pretrained weights (DiT + T5 + CLIP)
      4. Apply freeze configuration
      5. Return ready-to-train model

    Args:
        config: Full config dict (from robotx_finetune_1_2b.json).
        audiox_ckpt_path: Path to AudioX pretrained checkpoint.
            If None, reads from config["audiox_pretrained"]["ckpt_path"].

    Returns:
        AudioXFineTuneModel instance.
    """
    # 1. Build the base AudioX model with 64 io_channels
    factory_config = copy.deepcopy(config)
    factory_config["model_type"] = "diffusion_cond"

    model_cfg = factory_config.setdefault("model", {})
    diff_cfg = model_cfg.get("diffusion", {}).get("config", {})
    if "io_channels" not in model_cfg:
        model_cfg["io_channels"] = diff_cfg.get("io_channels", config.get("latent_dim", 64))

    factory_config.setdefault("sample_rate", 1)
    factory_config.setdefault("sample_size", config.get("action_chunk_size", 50))

    base_model = create_model_from_config(factory_config)
    base_model._config = config

    # 2. Wrap with adapter layers
    model = AudioXFineTuneModel(base_model, config)

    # 3. Load pretrained weights if available
    if audiox_ckpt_path is None:
        pretrained_cfg = config.get("audiox_pretrained", {})
        audiox_ckpt_path = pretrained_cfg.get("ckpt_path")

    if audiox_ckpt_path and os.path.exists(audiox_ckpt_path):
        load_audiox_pretrained(model, audiox_ckpt_path)
    else:
        if audiox_ckpt_path:
            print(f"[FineTune] WARNING: pretrained checkpoint not found: {audiox_ckpt_path}")
        print("[FineTune] Starting from random initialization (no pretrained weights)")

    # 4. Apply freeze configuration
    finetune_config = config.get("finetune", {})
    freeze_for_finetune(model, finetune_config)

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[FineTune] Total params: {total / 1e6:.1f}M, Trainable: {trainable / 1e6:.1f}M")

    return model

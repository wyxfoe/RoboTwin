"""PyTorch Lightning training wrapper for AudioX robot diffusion model.

Implements v-prediction diffusion training with cosine noise schedule,
classifier-free guidance dropout, and optional EMA.
"""

import copy
import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_alphas_sigmas(t):
    """Cosine noise schedule: alpha(t)=cos(pi/2*t), sigma(t)=sin(pi/2*t).

    Compatible with stable_audio_tools.inference.sampling.get_alphas_sigmas.
    """
    alphas = torch.cos(t * math.pi / 2)
    sigmas = torch.sin(t * math.pi / 2)
    return alphas, sigmas


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    def apply(self, model):
        """Replace model parameters with EMA values (for inference / export)."""
        model.load_state_dict(self.shadow, strict=False)


class RobotDiffusionTrainingWrapper(pl.LightningModule):
    """Lightning module wrapping AudioX model for robot trajectory training.

    Training loop:
        1. Sample timestep t ~ logit_normal or uniform
        2. Add noise: x_t = alpha_t * x_0 + sigma_t * noise
        3. Compute v-prediction target: v = alpha_t * noise - sigma_t * x_0
        4. Forward DiT: predicted_v = model(x_t, t, **cond)
        5. Loss = MSE(predicted_v, v)
    """

    def __init__(
        self,
        model,
        lr=1e-4,
        use_ema=True,
        log_loss_info=True,
        cfg_dropout_prob=0.1,
        timestep_sampler="logit_normal",
        optimizer_configs=None,
    ):
        super().__init__()
        self.diffusion = model            # full AudioX model
        self.lr = lr
        self.use_ema = use_ema
        self.log_loss_info = log_loss_info
        self.cfg_dropout_prob = cfg_dropout_prob
        self.timestep_sampler = timestep_sampler
        self.optimizer_configs = optimizer_configs or {}

        # EMA
        self.ema = None
        if use_ema:
            self.ema = EMA(model, decay=0.999)

        # Optional ActionHead (trainable output projection)
        self.action_head = None

    def _sample_timesteps(self, batch_size):
        """Sample diffusion timesteps in [0, 1]."""
        if self.timestep_sampler == "logit_normal":
            # Logit-normal: concentrate around mid-range noise levels
            u = torch.randn(batch_size, device=self.device)
            t = torch.sigmoid(-0.4 + 1.2 * u)
        else:
            t = torch.rand(batch_size, device=self.device)
        return t

    @torch.no_grad()
    def _encode_images_clip(self, metadata):
        """Encode camera images through CLIP independently, concatenate features.

        RDT-style: each camera view is processed through CLIP individually,
        then all patch token sequences are concatenated along dim=1.
        No temporal transformer — preserves full per-view spatial information.

        Returns:
            [concat_features, mask] matching AudioX conditioner output format.
        """
        clip_cond = self.diffusion.conditioner.conditioners["video"]
        clip_model = clip_cond.visual_encoder_model

        batch_size = len(metadata)
        num_cams = len(metadata[0]["camera_images"])
        all_features = []

        for cam_idx in range(num_cams):
            # Stack this camera across batch: (B, C, H, W)
            cam_batch = torch.stack(
                [m["camera_images"][cam_idx] for m in metadata]
            ).to(self.device)

            outputs = clip_model(pixel_values=cam_batch)
            # last_hidden_state: (B, num_patches+1, hidden_dim)
            all_features.append(outputs.last_hidden_state)

        # Concatenate all cameras: (B, num_cams * num_tokens, hidden_dim)
        concat_features = torch.cat(all_features, dim=1)
        # Mask must match token count for cross-attention
        mask = torch.ones(batch_size, concat_features.shape[1]).to(self.device)

        return [concat_features, mask]

    def _build_conditioning(self, metadata):
        """Build conditioning with RDT-style multi-view concatenation.

        - T5, trajectory: processed via AudioX conditioners as usual
        - Video: each camera independently through CLIP, then token concatenation
        """
        conditioner = self.diffusion.conditioner
        conditioning = {}

        # Process non-video conditioners normally
        for key, cond_module in conditioner.conditioners.items():
            if key == "video":
                continue
            inputs = [m[key] for m in metadata]
            conditioning[key] = cond_module(inputs, self.device)

        # RDT-style: independent CLIP per camera, concatenate tokens
        conditioning["video"] = self._encode_images_clip(metadata)

        return conditioning

    def training_step(self, batch, batch_idx):
        actions, metadata = batch  # actions: (B, action_dim, chunk_size)
        batch_size = actions.shape[0]

        # --- Conditioning (RDT-style multi-view) ---
        conditioning = self._build_conditioning(metadata)

        # Classifier-free guidance dropout: randomly mask conditioning
        if self.cfg_dropout_prob > 0 and self.training:
            mask = torch.rand(batch_size, device=self.device) < self.cfg_dropout_prob
            if mask.any():
                null_meta = [{
                    "prompt": "",
                    "camera_images": [img * 0 for img in m["camera_images"]],
                    "proprio": m["proprio"] * 0,
                } for m in metadata]
                null_cond = self._build_conditioning(
                    [null_meta[i] if mask[i] else metadata[i] for i in range(batch_size)]
                )
                conditioning = null_cond

        cond_inputs = self.diffusion.get_conditioning_inputs(conditioning)

        # --- Diffusion noise ---
        t = self._sample_timesteps(batch_size)
        alphas, sigmas = _get_alphas_sigmas(t)
        # Reshape for broadcasting: (B,) -> (B, 1, 1)
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]

        noise = torch.randn_like(actions)
        noised_actions = actions * alphas + noise * sigmas

        # v-prediction target
        targets = noise * alphas - actions * sigmas

        # --- Forward ---
        output = self.diffusion.model(
            noised_actions,
            t,
            **cond_inputs,
        )

        loss = F.mse_loss(output, targets)

        # Logging
        self.log("train/loss", loss, prog_bar=True)
        if self.log_loss_info:
            with torch.no_grad():
                self.log("train/mean_t", t.mean())

        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        if self.ema is not None:
            self.ema.update(self.diffusion)

    def configure_optimizers(self):
        # Extract optimizer config for the diffusion component
        diff_cfg = self.optimizer_configs.get("diffusion", {})
        opt_cfg = diff_cfg.get("optimizer", {}).get("config", {})
        sched_cfg = diff_cfg.get("scheduler", {})

        lr = opt_cfg.get("lr", self.lr)
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        weight_decay = opt_cfg.get("weight_decay", 0.01)

        optimizer = torch.optim.AdamW(
            self.diffusion.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

        result = {"optimizer": optimizer}

        # Scheduler
        sched_type = sched_cfg.get("type", "")
        sched_params = sched_cfg.get("config", {})
        if sched_type == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=sched_params.get("T_max", 50000),
                eta_min=sched_params.get("eta_min", 1e-6),
            )
            result["lr_scheduler"] = {"scheduler": scheduler, "interval": "step"}

        return result

    def export_model(self, path, use_safetensors=False):
        """Export model weights (with EMA if available)."""
        model_to_save = copy.deepcopy(self.diffusion)
        if self.ema is not None:
            self.ema.apply(model_to_save)

        if use_safetensors:
            from safetensors.torch import save_file
            save_file(model_to_save.state_dict(), path)
        else:
            torch.save(model_to_save.state_dict(), path)
        print(f"[Training] Exported model to {path}")


class RobotDemoCallback(pl.callbacks.Callback):
    """Periodically generates demo trajectories during training for logging."""

    def __init__(self, demo_every=2000, num_demos=4, demo_steps=50, demo_cfg_scales=None):
        super().__init__()
        self.demo_every = demo_every
        self.num_demos = num_demos
        self.demo_steps = demo_steps
        self.demo_cfg_scales = demo_cfg_scales or [1.0, 3.0]

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step > 0 and trainer.global_step % self.demo_every == 0:
            self._generate_demos(trainer, pl_module, batch)

    @torch.no_grad()
    def _generate_demos(self, trainer, pl_module, batch):
        """Generate and log demo trajectories."""
        actions, metadata = batch
        n = min(self.num_demos, actions.shape[0])

        try:
            from stable_audio_tools.inference.generation import generate_diffusion_cond
        except ImportError:
            return  # Skip if inference module unavailable

        model = pl_module.diffusion
        model_config = getattr(model, "_config", {})
        sample_size = model_config.get("sample_size", actions.shape[2])
        sample_rate = model_config.get("sample_rate", 1)

        for cfg_scale in self.demo_cfg_scales:
            demo_cond = metadata[:n]
            preds = generate_diffusion_cond(
                model,
                conditioning=demo_cond,
                sample_size=sample_size,
                sample_rate=sample_rate,
                device=pl_module.device,
                steps=self.demo_steps,
                cfg_scale=cfg_scale,
            )

            # Compute MSE against ground truth
            gt = actions[:n].to(pl_module.device)
            mse = F.mse_loss(preds[:n], gt).item()
            pl_module.log(f"demo/mse_cfg{cfg_scale}", mse)

        print(f"[Demo] Step {trainer.global_step}: generated {n} demos")

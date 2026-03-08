"""PyTorch Lightning training wrapper for AudioX robot diffusion model.

Implements v-prediction diffusion training with cosine noise schedule,
classifier-free guidance dropout, and optional EMA.

Includes both the original from-scratch wrapper (RobotDiffusionTrainingWrapper)
and the fine-tuning wrapper (RobotFineTuneTrainingWrapper) that supports
differential learning rates and AudioX pretrained weight reuse.
"""

import copy
import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_alphas_sigmas(t):
    """Cosine noise schedule: alpha(t)=cos(pi/2*t), sigma(t)=sin(pi/2*t).

    Compatible with audiox.inference.sampling.get_alphas_sigmas.
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
                if self.shadow[k].device != v.device:
                    self.shadow[k] = self.shadow[k].to(v.device)
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

        Note: AudioX's TrajectoryConditioner may not call proj_out internally,
        so we apply it here if the output dim doesn't match output_dim.
        """
        conditioner = self.diffusion.conditioner
        conditioning = {}

        # Process non-video conditioners normally
        for key, cond_module in conditioner.conditioners.items():
            if key == "video":
                continue
            inputs = [m[key] for m in metadata]
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


class RobotFineTuneTrainingWrapper(pl.LightningModule):
    """Lightning module for fine-tuning AudioX 1.2B on robot trajectories.

    Differences from RobotDiffusionTrainingWrapper:
      - Operates in 64-dim latent space (not 14-dim action space directly)
      - Uses input_proj to map actions → latent before adding noise
      - Adds reconstruction loss to train input_proj + output_proj jointly
      - Supports differential learning rates:
          * adapter_lr   for input_proj, output_proj, trajectory conditioner
          * adapter_lr * dit_lr_scale   for DiT backbone
          * T5, CLIP conditioners are frozen

    Loss function:
        L = L_diffusion + λ * L_reconstruction

        L_diffusion:       MSE(v_predicted, v_target) in 64-dim latent space
                           where v_target = α·ε − σ·x₀ (v-prediction objective)
        L_reconstruction:  MSE(output_proj(input_proj(actions)), actions)
                           ensures the adapter pair forms a coherent round-trip
        λ:                 finetune_config["reconstruction_weight"] (default 0.1)
    """

    def __init__(
        self,
        model,
        finetune_config=None,
        use_ema=True,
        log_loss_info=True,
        cfg_dropout_prob=0.1,
        timestep_sampler="logit_normal",
        optimizer_configs=None,
    ):
        super().__init__()
        self.diffusion = model  # AudioXFineTuneModel
        self.finetune_config = finetune_config or {}
        self.use_ema = use_ema
        self.log_loss_info = log_loss_info
        self.cfg_dropout_prob = cfg_dropout_prob
        self.timestep_sampler = timestep_sampler
        self.optimizer_configs = optimizer_configs or {}

        self.recon_weight = self.finetune_config.get("reconstruction_weight", 0.1)

        # EMA
        self.ema = None
        if use_ema:
            self.ema = EMA(model, decay=0.999)

    def _sample_timesteps(self, batch_size):
        if self.timestep_sampler == "logit_normal":
            u = torch.randn(batch_size, device=self.device)
            t = torch.sigmoid(-0.4 + 1.2 * u)
        else:
            t = torch.rand(batch_size, device=self.device)
        return t

    @torch.no_grad()
    def _encode_images_clip(self, metadata):
        """RDT-style: each camera through CLIP independently, concatenate tokens."""
        clip_cond = self.diffusion.conditioner.conditioners["video"]
        clip_model = clip_cond.visual_encoder_model

        batch_size = len(metadata)
        num_cams = len(metadata[0]["camera_images"])
        all_features = []

        for cam_idx in range(num_cams):
            cam_batch = torch.stack(
                [m["camera_images"][cam_idx] for m in metadata]
            ).to(self.device)
            outputs = clip_model(pixel_values=cam_batch)
            all_features.append(outputs.last_hidden_state)

        concat_features = torch.cat(all_features, dim=1)
        mask = torch.ones(batch_size, concat_features.shape[1]).to(self.device)
        return [concat_features, mask]

    def _build_conditioning(self, metadata):
        """Build conditioning from metadata (same as parent class)."""
        conditioner = self.diffusion.conditioner
        conditioning = {}

        for key, cond_module in conditioner.conditioners.items():
            if key == "video":
                continue
            inputs = [m[key] for m in metadata]
            conditioning[key] = cond_module(inputs, self.device)

            features, mask = conditioning[key]
            if (hasattr(cond_module, 'proj_out') and
                    hasattr(cond_module, 'output_dim') and
                    features.shape[-1] != cond_module.output_dim):
                features = cond_module.proj_out(features)
                conditioning[key] = [features, mask]

        conditioning["video"] = self._encode_images_clip(metadata)
        return conditioning

    def training_step(self, batch, batch_idx):
        actions, metadata = batch  # actions: (B, action_dim=14, chunk_size)
        batch_size = actions.shape[0]

        # --- Project actions to latent space ---
        latent_actions = self.diffusion.encode_actions(actions)  # (B, 64, T)

        # --- Reconstruction loss (train adapter round-trip) ---
        recon_actions = self.diffusion.decode_latent(latent_actions)  # (B, 14, T)
        loss_recon = F.mse_loss(recon_actions, actions)

        # --- Conditioning ---
        conditioning = self._build_conditioning(metadata)

        if self.cfg_dropout_prob > 0 and self.training:
            drop_mask = torch.rand(batch_size, device=self.device) < self.cfg_dropout_prob
            if drop_mask.any():
                null_meta = [{
                    "prompt": "",
                    "camera_images": [img * 0 for img in m["camera_images"]],
                    "proprio": m["proprio"] * 0,
                } for m in metadata]
                mixed_meta = [
                    null_meta[i] if drop_mask[i] else metadata[i]
                    for i in range(batch_size)
                ]
                conditioning = self._build_conditioning(mixed_meta)

        cond_inputs = self.diffusion.get_conditioning_inputs(conditioning)

        # --- Diffusion in latent space ---
        t = self._sample_timesteps(batch_size)
        alphas, sigmas = _get_alphas_sigmas(t)
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]

        noise = torch.randn_like(latent_actions)
        noised_latent = latent_actions * alphas + noise * sigmas

        # v-prediction target in latent space
        targets = noise * alphas - latent_actions * sigmas

        # Forward through DiT backbone
        output = self.diffusion.model(
            noised_latent,
            t,
            **cond_inputs,
        )

        loss_diffusion = F.mse_loss(output, targets)

        # --- Combined loss ---
        loss = loss_diffusion + self.recon_weight * loss_recon

        # Logging
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/loss_diffusion", loss_diffusion, prog_bar=True)
        self.log("train/loss_recon", loss_recon)
        if self.log_loss_info:
            with torch.no_grad():
                self.log("train/mean_t", t.mean())

        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        if self.ema is not None:
            self.ema.update(self.diffusion)

    def configure_optimizers(self):
        """Build optimizer with differential learning rates.

        Parameter groups:
          1. Adapter (input_proj, output_proj, trajectory conditioner): adapter_lr
          2. DiT backbone: adapter_lr * dit_lr_scale
          T5/CLIP are frozen (requires_grad=False) so excluded automatically.
        """
        diff_cfg = self.optimizer_configs.get("diffusion", {})
        opt_cfg = diff_cfg.get("optimizer", {}).get("config", {})
        sched_cfg = diff_cfg.get("scheduler", {})

        adapter_lr = self.finetune_config.get("adapter_lr", opt_cfg.get("lr", 1e-4))
        dit_lr_scale = self.finetune_config.get("dit_lr_scale", 0.1)
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        weight_decay = opt_cfg.get("weight_decay", 0.01)

        # Collect parameter ID sets for each group
        adapter_ids = {id(p) for p in self.diffusion.adapter_parameters() if p.requires_grad}
        backbone_ids = {id(p) for p in self.diffusion.backbone_parameters() if p.requires_grad}

        adapter_params = [p for p in self.diffusion.adapter_parameters() if p.requires_grad]
        backbone_params = [p for p in self.diffusion.backbone_parameters() if p.requires_grad]

        param_groups = []
        if adapter_params:
            param_groups.append({
                "params": adapter_params,
                "lr": adapter_lr,
                "name": "adapter",
            })
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": adapter_lr * dit_lr_scale,
                "name": "backbone",
            })

        if not param_groups:
            raise ValueError("No trainable parameters found!")

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=betas,
            weight_decay=weight_decay,
        )

        n_adapter = sum(p.numel() for p in adapter_params)
        n_backbone = sum(p.numel() for p in backbone_params)
        print(f"[FineTune Optimizer] adapter: {n_adapter/1e6:.1f}M params @ lr={adapter_lr}")
        print(f"[FineTune Optimizer] backbone: {n_backbone/1e6:.1f}M params @ lr={adapter_lr * dit_lr_scale}")

        result = {"optimizer": optimizer}

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
        """Export full fine-tuned model (base_model + adapters, with EMA)."""
        model_to_save = copy.deepcopy(self.diffusion)
        if self.ema is not None:
            self.ema.apply(model_to_save)

        if use_safetensors:
            from safetensors.torch import save_file
            save_file(model_to_save.state_dict(), path)
        else:
            torch.save(model_to_save.state_dict(), path)
        print(f"[FineTune] Exported model to {path}")


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
            try:
                self._generate_demos(trainer, pl_module, batch)
            except Exception as e:
                print(f"[Demo] Step {trainer.global_step}: demo generation failed ({type(e).__name__}: {e}), skipping")

    @torch.no_grad()
    def _generate_demos(self, trainer, pl_module, batch):
        """Generate and log demo trajectories.

        Uses generate_diffusion_cond with batch_size=n to match the
        pre-built conditioning_tensors batch dimension.  The DIT's CFG
        branch correctly handles global_embed=None and prepend_cond=None
        (falls back to timestep_embed), so no fake conds are needed.
        """
        actions, metadata = batch
        n = min(self.num_demos, actions.shape[0])

        from audiox.inference.generation import generate_diffusion_cond

        model = pl_module.diffusion
        model_config = getattr(model, "_config", {})
        sample_size = model_config.get("sample_size", actions.shape[2])

        # Pre-compute conditioning tensors via the training wrapper's
        # _build_conditioning (which correctly maps camera_images → video).
        demo_meta = metadata[:n]
        conditioning_tensors = pl_module._build_conditioning(demo_meta)

        for cfg_scale in self.demo_cfg_scales:
            preds = generate_diffusion_cond(
                model,
                conditioning_tensors=conditioning_tensors,
                batch_size=n,
                sample_size=sample_size,
                device=pl_module.device,
                steps=self.demo_steps,
                cfg_scale=cfg_scale,
            )

            # Decode from 64-dim latent space to 14-dim action space
            preds = model.decode_latent(preds)

            # Compute MSE against ground truth
            gt = actions[:n].to(pl_module.device)
            mse = F.mse_loss(preds[:n], gt).item()
            pl_module.log(f"demo/mse_cfg{cfg_scale}", mse)

        print(f"[Demo] Step {trainer.global_step}: generated {n} demos")

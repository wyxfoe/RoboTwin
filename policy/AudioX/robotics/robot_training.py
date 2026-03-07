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
      - Pre-caches frozen CLIP/T5 features to avoid redundant computation

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

        # Conditioning feature cache (populated in on_fit_start)
        # Maps sample_key -> {"video": [features, mask], "prompt": [features, mask]}
        self._cond_cache = {}
        self._cond_cache_enabled = False

    def _sample_timesteps(self, batch_size):
        if self.timestep_sampler == "logit_normal":
            u = torch.randn(batch_size, device=self.device)
            t = torch.sigmoid(-0.4 + 1.2 * u)
        else:
            t = torch.rand(batch_size, device=self.device)
        return t

    def on_fit_start(self):
        """Pre-compute and cache frozen CLIP/T5 features for all training samples."""
        trainer = self.trainer
        dataloader = trainer.train_dataloader
        if dataloader is None:
            return

        dataset = dataloader.dataset
        num_samples = len(dataset)
        print(f"\n[FineTune] Pre-computing CLIP + T5 features for {num_samples} samples...")

        clip_cond = self.diffusion.conditioner.conditioners["video"]
        clip_model = clip_cond.visual_encoder_model

        # Pre-compute T5 embedding for the task description (single string, shared)
        prompt_cond = self.diffusion.conditioner.conditioners.get("prompt")
        cached_prompt = None
        if prompt_cond is not None:
            task_desc = dataset.task_description
            cached_prompt = prompt_cond([task_desc], self.device)
            # Also cache null prompt for CFG dropout
            self._null_prompt = prompt_cond([""], self.device)
        self._cached_prompt = cached_prompt

        # Pre-compute CLIP features per unique (episode, timestep)
        # Key: (ep_idx, start_t) -> concatenated CLIP features (1, num_cams*tokens, dim)
        clip_cache = {}
        num_cams = len(dataset.camera_names)

        batch_size = 64  # process images in batches for efficiency
        sample_keys = []
        image_batches = {cam_idx: [] for cam_idx in range(num_cams)}

        for sample_idx in range(num_samples):
            ep_idx, start_t = dataset.samples[sample_idx]
            key = (ep_idx, start_t)
            if key in clip_cache:
                continue
            sample_keys.append(key)
            # Get images from dataset cache or load
            for cam_idx in range(num_cams):
                if dataset.cache_in_memory:
                    cam_data = dataset._ep_images[ep_idx].get(cam_idx, {})
                    if start_t in cam_data:
                        image_batches[cam_idx].append(torch.from_numpy(cam_data[start_t]))
                    else:
                        image_batches[cam_idx].append(
                            torch.zeros(3, dataset.image_size, dataset.image_size))
                else:
                    # Fallback: load from disk
                    _, metadata = dataset[sample_idx]
                    for ci in range(num_cams):
                        image_batches[ci].append(metadata["camera_images"][ci])
                    break

            # Process in batches
            if len(sample_keys) == batch_size or sample_idx == num_samples - 1:
                if not sample_keys:
                    continue
                n = len(sample_keys)
                all_cam_features = []
                with torch.no_grad():
                    for cam_idx in range(num_cams):
                        imgs = torch.stack(image_batches[cam_idx][:n]).to(self.device)
                        outputs = clip_model(pixel_values=imgs)
                        all_cam_features.append(outputs.last_hidden_state.cpu())

                # concat cameras: (n, num_cams * tokens, dim)
                concat = torch.cat(all_cam_features, dim=1)
                for i, key in enumerate(sample_keys):
                    clip_cache[key] = concat[i:i+1]  # (1, num_cams*tokens, dim)

                sample_keys = []
                image_batches = {cam_idx: [] for cam_idx in range(num_cams)}

        # Also cache null (zero) image features for CFG dropout
        with torch.no_grad():
            null_features = []
            zero_img = torch.zeros(1, 3, dataset.image_size, dataset.image_size).to(self.device)
            for _ in range(num_cams):
                outputs = clip_model(pixel_values=zero_img)
                null_features.append(outputs.last_hidden_state.cpu())
            self._null_clip = torch.cat(null_features, dim=1)  # (1, num_cams*tokens, dim)

        self._clip_cache = clip_cache
        self._cond_cache_enabled = True
        print(f"[FineTune] Cached {len(clip_cache)} unique CLIP features. "
              f"Training will skip CLIP/T5 encoding.")

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

    def _get_cached_clip(self, metadata):
        """Look up pre-computed CLIP features from cache."""
        batch_features = []
        for m in metadata:
            key = m.get("_sample_key")
            if key is not None and key in self._clip_cache:
                batch_features.append(self._clip_cache[key])
            elif key is None:
                # Null sample from CFG dropout
                batch_features.append(self._null_clip)
            else:
                # Unknown key, fallback to on-the-fly computation
                return None
        concat = torch.cat(batch_features, dim=0).to(self.device)
        mask = torch.ones(concat.shape[0], concat.shape[1]).to(self.device)
        return [concat, mask]

    def _build_conditioning(self, metadata):
        """Build conditioning, using cached CLIP/T5 features when available."""
        conditioner = self.diffusion.conditioner
        conditioning = {}

        for key, cond_module in conditioner.conditioners.items():
            if key == "video":
                continue

            if key == "prompt" and self._cond_cache_enabled and self._cached_prompt is not None:
                # Use cached T5 features — check if any sample has empty prompt (CFG null)
                has_null = any(m.get("prompt", "") == "" for m in metadata)
                has_real = any(m.get("prompt", "") != "" for m in metadata)
                if not has_null:
                    # All real prompts (same text) — tile cached features
                    features, mask = self._cached_prompt
                    bs = len(metadata)
                    conditioning[key] = [
                        features.expand(bs, -1, -1),
                        mask.expand(bs, -1),
                    ]
                elif not has_real:
                    # All null prompts
                    features, mask = self._null_prompt
                    bs = len(metadata)
                    conditioning[key] = [
                        features.expand(bs, -1, -1),
                        mask.expand(bs, -1),
                    ]
                else:
                    # Mixed: build per-sample
                    real_f, real_m = self._cached_prompt
                    null_f, null_m = self._null_prompt
                    f_list = []
                    m_list = []
                    for m in metadata:
                        if m.get("prompt", "") == "":
                            f_list.append(null_f)
                            m_list.append(null_m)
                        else:
                            f_list.append(real_f)
                            m_list.append(real_m)
                    conditioning[key] = [
                        torch.cat(f_list, dim=0),
                        torch.cat(m_list, dim=0),
                    ]
                continue

            inputs = [m[key] for m in metadata]
            conditioning[key] = cond_module(inputs, self.device)

            features, mask = conditioning[key]
            if (hasattr(cond_module, 'proj_out') and
                    hasattr(cond_module, 'output_dim') and
                    features.shape[-1] != cond_module.output_dim):
                features = cond_module.proj_out(features)
                conditioning[key] = [features, mask]

        # CLIP features: use cache if available
        if self._cond_cache_enabled:
            cached = self._get_cached_clip(metadata)
            if cached is not None:
                conditioning["video"] = cached
            else:
                conditioning["video"] = self._encode_images_clip(metadata)
        else:
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
                    "_sample_key": None,  # signals null for cache lookup
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
            self._generate_demos(trainer, pl_module, batch)

    @torch.no_grad()
    def _generate_demos(self, trainer, pl_module, batch):
        """Generate and log demo trajectories."""
        actions, metadata = batch
        n = min(self.num_demos, actions.shape[0])

        try:
            from audiox.inference.generation import generate_diffusion_cond
        except ImportError:
            return  # Skip if inference module unavailable

        model = pl_module.diffusion
        model_config = getattr(model, "_config", {})
        sample_size = model_config.get("sample_size", actions.shape[2])
        sample_rate = model_config.get("sample_rate", 1)

        # Pre-compute conditioning tensors (maps camera_images → video, etc.)
        demo_meta = metadata[:n]
        conditioning_tensors = pl_module._build_conditioning(demo_meta)

        for cfg_scale in self.demo_cfg_scales:
            preds = generate_diffusion_cond(
                model,
                conditioning_tensors=conditioning_tensors,
                sample_size=sample_size,
                sample_rate=sample_rate,
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

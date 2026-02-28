"""
AudioX 1.2B Fine-Tuning Script for Robot Trajectories.

Fine-tunes the pretrained AudioX 1.2B diffusion transformer on robot
action data by replacing the audio interface layers with action-space
adapters while preserving the DiT backbone's generative prior.

Architecture:
    action (14-dim) → input_proj → [64-dim latent] → DiT (1.2B pretrained)
                                                    → [64-dim latent] → output_proj → action (14-dim)

Loss:
    L = L_diffusion (v-prediction MSE in 64-dim latent)
      + λ · L_reconstruction (MSE round-trip in 14-dim action space)

Usage:
    # Fine-tune from AudioX pretrained checkpoint:
    python train_finetune.py \
        --config configs/robotx_finetune_1_2b.json \
        --audiox_ckpt /path/to/audiox_1.2b.ckpt \
        --data_dir /path/to/data \
        --batch_size 8

    # Resume from fine-tuning checkpoint:
    python train_finetune.py \
        --config configs/robotx_finetune_1_2b.json \
        --ckpt_path checkpoints/finetune/robotx-00010000.ckpt

    # Multi-GPU fine-tuning:
    python train_finetune.py \
        --config configs/robotx_finetune_1_2b.json \
        --audiox_ckpt /path/to/audiox_1.2b.ckpt \
        --data_dir /path/to/data \
        --num_gpus 4 --batch_size 4

    # Freeze DiT backbone (only train adapter layers):
    python train_finetune.py \
        --config configs/robotx_finetune_1_2b.json \
        --audiox_ckpt /path/to/audiox_1.2b.ckpt \
        --freeze_dit
"""

import argparse
import json
import os
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robotics.finetune_model import create_finetune_model
from robotics.robot_training import (
    RobotFineTuneTrainingWrapper,
    RobotDemoCallback,
)
from robotics.robotwin_dataset import create_robotwin_dataloader


def main():
    parser = argparse.ArgumentParser(description="Fine-tune AudioX 1.2B for robot trajectories")

    # Config
    parser.add_argument("--config", type=str, required=True,
                        help="Path to fine-tune config JSON (e.g. configs/robotx_finetune_1_2b.json)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data directory from config")

    # AudioX pretrained model
    parser.add_argument("--audiox_ckpt", type=str, default=None,
                        help="Path to AudioX 1.2B pretrained checkpoint")

    # Fine-tune control
    parser.add_argument("--freeze_dit", action="store_true",
                        help="Freeze DiT backbone (only train adapter layers)")
    parser.add_argument("--dit_lr_scale", type=float, default=None,
                        help="Override DiT backbone learning rate scale (default 0.1)")
    parser.add_argument("--adapter_lr", type=float, default=None,
                        help="Override adapter learning rate (default 1e-4)")
    parser.add_argument("--recon_weight", type=float, default=None,
                        help="Override reconstruction loss weight (default 0.1)")

    # Training
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size per GPU")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                        help="bf16-mixed recommended for 1.2B model")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50000,
        help="Maximum number of training steps. "
             "If --max_epochs is set, this will be ignored."
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Maximum number of training epochs. "
             "If set, overrides --max_steps."
    )
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # Checkpointing
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Resume from fine-tuning checkpoint")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/finetune")
    parser.add_argument("--save_every", type=int, default=2000,
                        help="Save checkpoint every N steps (if using --max_steps)")
    parser.add_argument("--save_every_epoch", type=int, default=None,
                        help="Save checkpoint every N epochs (if using --max_epochs). "
                             "If set, overrides --save_every behavior.")

    # Logging
    parser.add_argument("--project_name", type=str, default="robotx-finetune")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--demo_every", type=int, default=2000)

    # Task
    parser.add_argument("--task_description", type=str, default=None)
    parser.add_argument("--max_episodes", type=int, default=None)

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # Load config
    print(f"Loading config from {args.config}")
    with open(args.config, "r") as f:
        config = json.load(f)

    # Apply CLI overrides to finetune config
    finetune_config = config.setdefault("finetune", {})
    if args.freeze_dit:
        finetune_config["freeze_dit"] = True
    if args.dit_lr_scale is not None:
        finetune_config["dit_lr_scale"] = args.dit_lr_scale
    if args.adapter_lr is not None:
        finetune_config["adapter_lr"] = args.adapter_lr
    if args.recon_weight is not None:
        finetune_config["reconstruction_weight"] = args.recon_weight

    # Apply CLI overrides to dataset config
    dataset_config = config.get("dataset", {})
    if args.data_dir:
        dataset_config["data_dir"] = args.data_dir
    if args.batch_size:
        dataset_config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        dataset_config["num_workers"] = args.num_workers
    if args.task_description:
        dataset_config["task_description"] = args.task_description

    # Create fine-tune model
    print("\n" + "=" * 60)
    print("Creating AudioX Fine-Tune Model")
    print("=" * 60)
    print(f"  action_dim:  {config['action_dim']}")
    print(f"  latent_dim:  {config.get('latent_dim', 64)}")
    print(f"  chunk_size:  {config['action_chunk_size']}")
    print(f"  DiT:         embed={config['model']['diffusion']['config']['embed_dim']}, "
          f"depth={config['model']['diffusion']['config']['depth']}, "
          f"heads={config['model']['diffusion']['config']['num_heads']}")
    print(f"  freeze_dit:  {finetune_config.get('freeze_dit', False)}")
    print(f"  dit_lr_scale: {finetune_config.get('dit_lr_scale', 0.1)}")

    model = create_finetune_model(config, audiox_ckpt_path=args.audiox_ckpt)

    # Create training wrapper
    training_config = config.get("training", {})
    training_wrapper = RobotFineTuneTrainingWrapper(
        model=model,
        finetune_config=finetune_config,
        use_ema=training_config.get("use_ema", True),
        log_loss_info=training_config.get("log_loss_info", True),
        cfg_dropout_prob=training_config.get("cfg_dropout_prob", 0.1),
        timestep_sampler=training_config.get("timestep_sampler", "logit_normal"),
        optimizer_configs=training_config.get("optimizer_configs", None),
    )

    # Create data loader
    print("\nCreating data loader...")
    data_dir = dataset_config.get("data_dir")
    if not data_dir:
        raise ValueError("data_dir must be specified in config or via --data_dir")

    per_gpu_batch_size = dataset_config.get("batch_size", 8)
    total_batch_size = per_gpu_batch_size * args.num_gpus * args.num_nodes
    num_workers_per_gpu = dataset_config.get("num_workers", 4)

    print(f"  Data directory: {data_dir}")
    print(f"  Batch size per GPU: {per_gpu_batch_size}")
    print(f"  Total batch size: {total_batch_size}")

    train_dataloader, train_dataset = create_robotwin_dataloader(
        data_dir=data_dir,
        batch_size=per_gpu_batch_size,
        action_chunk_size=config.get("action_chunk_size", 50),
        image_size=dataset_config.get("image_size", 224),
        camera_names=dataset_config.get("camera_names", ["front_camera", "head_camera"]),
        task_description=dataset_config.get("task_description", "complete the task"),
        num_workers=num_workers_per_gpu,
        max_episodes=args.max_episodes,
        normalize=dataset_config.get("normalize", True),
        augment=dataset_config.get("augment", True),
        shuffle=True,
    )

    print(f"  Loaded {len(train_dataset)} samples from {len(train_dataset.episodes)} episodes")

    # Save action stats
    os.makedirs(args.save_dir, exist_ok=True)
    if hasattr(train_dataset, "action_stats") and train_dataset.action_stats:
        stats_path = os.path.join(args.save_dir, "action_stats.pt")
        torch.save(train_dataset.action_stats, stats_path)
        print(f"  Saved action statistics to {stats_path}")

    # Callbacks - determine save strategy based on training mode
    if args.max_epochs is not None:
        # Epoch-based training: save by epoch
        save_every_epoch = args.save_every_epoch if args.save_every_epoch is not None else 1
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.save_dir,
            filename="robotx-finetune-epoch{epoch:03d}",
            every_n_epochs=save_every_epoch,
            save_top_k=-1,
        )
        print(f"  Checkpoint saving: every {save_every_epoch} epoch(s)")
    else:
        # Step-based training: save by step
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.save_dir,
            filename="robotx-finetune-{step:08d}",
            every_n_train_steps=args.save_every,
            save_top_k=-1,
        )
        print(f"  Checkpoint saving: every {args.save_every} steps")
    
    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="step"),
    ]

    demo_config = training_config.get("demo", {})
    callbacks.append(RobotDemoCallback(
        demo_every=demo_config.get("demo_every", args.demo_every),
        num_demos=demo_config.get("num_demos", 4),
        demo_steps=demo_config.get("demo_steps", 50),
        demo_cfg_scales=demo_config.get("demo_cfg_scales", [1.0, 3.0]),
    ))

    # Logger
    if args.wandb:
        logger = WandbLogger(
            project=args.project_name,
            name=args.run_name,
            save_dir=args.save_dir,
        )
    else:
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(
            save_dir=args.save_dir,
            name="tensorboard",
        )

    # Trainer
    if args.num_gpus > 1 or args.num_nodes > 1:
        strategy = "ddp_find_unused_parameters_true" if args.strategy == "auto" else args.strategy
    else:
        strategy = "auto"

    trainer_kwargs = dict(
        devices=args.num_gpus if args.num_gpus > 0 else "auto",
        num_nodes=args.num_nodes,
        accelerator="gpu" if torch.cuda.is_available() and args.num_gpus > 0 else "cpu",
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=None,
        limit_val_batches=0,
        enable_checkpointing=True,
        sync_batchnorm=args.num_gpus > 1,
    )

    # Prefer epoch-based stopping if max_epochs is provided
    if args.max_epochs is not None:
        trainer_kwargs["max_epochs"] = args.max_epochs
        # Disable step-based early stopping when using epochs
        trainer_kwargs["max_steps"] = -1
    else:
        trainer_kwargs["max_steps"] = args.max_steps

    trainer = pl.Trainer(**trainer_kwargs)

    # Train
    print("\n" + "=" * 60)
    print("Starting fine-tuning...")
    print(f"  Loss = L_diffusion + {finetune_config.get('reconstruction_weight', 0.1)} * L_reconstruction")
    print("=" * 60)

    trainer.fit(
        training_wrapper,
        train_dataloaders=train_dataloader,
        ckpt_path=args.ckpt_path,
    )

    # Export
    final_path = os.path.join(args.save_dir, "robotx_finetune_final.pt")
    training_wrapper.export_model(final_path, use_safetensors=False)
    print(f"\nExported final fine-tuned model to {final_path}")


if __name__ == "__main__":
    main()

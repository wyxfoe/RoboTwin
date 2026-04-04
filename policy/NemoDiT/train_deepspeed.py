"""
NemoDiT DeepSpeed Distributed Training Script.

支持 DeepSpeed ZeRO Stage 1/2/3 + Data Parallel 多卡训练加速。
可通过 deepspeed launcher 或 torchrun 启动。

Usage:
    # 单机多卡 (推荐)
    deepspeed train_deepspeed.py --deepspeed_config ds_config.json --data_path <path> [options]

    # 指定 GPU
    deepspeed --include localhost:0,1,2,3 train_deepspeed.py --deepspeed_config ds_config.json ...

    # 多机多卡
    deepspeed --hostfile hostfile train_deepspeed.py --deepspeed_config ds_config.json ...

Features:
    - DeepSpeed ZeRO optimizer (Stage 1/2/3) for memory-efficient training
    - Data Parallel (DP) across multiple GPUs
    - FP16 / BF16 mixed precision via DeepSpeed
    - Gradient accumulation
    - torch.profiler integration for bottleneck analysis
    - EMA model support
    - WandB logging (rank 0 only)
    - Checkpoint save/resume with DeepSpeed engine
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
from torchvision import transforms
from tqdm import tqdm
import os
import argparse
import math
import json
from pathlib import Path

import deepspeed
from deepspeed import comm as dist

from model.action_model.action_model import ActionModel
from dataloader import RobotDataset
from utils.wandb_utils import WandbLogger
from utils.ema_model import EMAModel


def parse_args():
    parser = argparse.ArgumentParser(description='DeepSpeed Distributed Training for NemoDiT')

    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to robot dataset directory containing .hdf5 files')
    parser.add_argument('--num_cameras', type=int, default=4,
                        help='Number of camera views (default: 4)')
    parser.add_argument('--use_both_arms', action='store_true', default=True,
                        help='Use both arms data (default: True)')
    parser.add_argument('--action_type', type=str, default='joint',
                        choices=['endpose', 'joint'],
                        help='Action type: endpose (ee pose) or joint (joint angles)')
    parser.add_argument('--quat_convention', type=str, default='wxyz',
                        choices=['wxyz', 'xyzw'],
                        help='Quaternion convention in HDF5 data (default: wxyz, only for endpose)')

    # Model arguments
    parser.add_argument('--model_type', type=str, default='DiT-S',
                        choices=['DiT-S', 'DiT-B', 'DiT-L', 'DiT-XL'],
                        help='DiT model size (default: DiT-B)')
    parser.add_argument('--dropout_prob', type=float, default=0.1,
                        help='Class dropout probability for classifier-free guidance (default: 0.1)')
    parser.add_argument('--action_dim', type=int, default=7,
                        help='Action dimension (auto-computed from action_type and use_both_arms)')
    parser.add_argument('--future_action_window', type=int, default=13,
                        help='Number of future action steps to predict (default: 13)')
    parser.add_argument('--past_action_window', type=int, default=0,
                        help='Number of past action steps as context (default: 0)')
    parser.add_argument('--token_size', type=int, default=2048,
                        help='Token size for conditioning (default: 2048)')

    # Chunk 机制参数
    parser.add_argument('--n_obs_steps', type=int, default=3,
                        help='Number of observation steps for visual conditioning (default: 3)')
    parser.add_argument('--temporal_agg', type=str, default='concat',
                        choices=['last', 'mean', 'concat'],
                        help='Temporal aggregation method for multi-frame observations (default: concat)')
    parser.add_argument('--n_action_steps', type=int, default=8,
                        help='Number of action steps to execute during inference (default: 8)')

    # Vision arguments
    parser.add_argument('--vision_backbone', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'vit_b_16', 'vit_b_32'],
                        help='Vision backbone type (default: resnet50)')
    parser.add_argument('--vision_pretrained', action='store_true', default=True,
                        help='Use pretrained vision backbone')
    parser.add_argument('--freeze_vision', action='store_true', default=False,
                        help='Freeze vision backbone weights')
    parser.add_argument('--adapter_type', type=str, default='mlp',
                        choices=['linear', 'mlp', 'attention_pooling'],
                        help='Feature adapter type')

    # Diffusion arguments
    parser.add_argument('--diffusion_steps', type=int, default=500,
                        help='Number of diffusion steps (default: 500)')
    parser.add_argument('--noise_schedule', type=str, default='squaredcos_cap_v2',
                        help='Noise schedule type (default: squaredcos_cap_v2)')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size per GPU (default: 16)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs (default: 500)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for L2 regularization (default: 0.01)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max norm (default: 1.0)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Number of warmup epochs (default: 0, no warmup)')
    parser.add_argument('--warmup_type', type=str, default='linear',
                        choices=['linear', 'cosine'],
                        help='Warmup schedule type (default: linear)')

    # EMA arguments
    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='Use Exponential Moving Average for model weights (default: False)')
    parser.add_argument('--ema_inv_gamma', type=float, default=1.0,
                        help='EMA inverse gamma (default: 1.0)')
    parser.add_argument('--ema_power', type=float, default=0.6667,
                        help='EMA power (default: 2/3)')
    parser.add_argument('--ema_max_value', type=float, default=0.9999,
                        help='Maximum EMA decay rate (default: 0.9999)')

    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--save_every', type=int, default=50,
                        help='Save checkpoint every N epochs (default: 50)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to DeepSpeed checkpoint directory to resume from')

    # Profiler arguments (torch.profiler)
    parser.add_argument('--use_profiler', action='store_true', default=False,
                        help='Enable torch.profiler for performance analysis (default: False)')
    parser.add_argument('--profiler_dir', type=str, default='profiler_logs',
                        help='Directory to save profiler traces (default: profiler_logs)')
    parser.add_argument('--profiler_wait', type=int, default=2,
                        help='Profiler schedule: wait steps (default: 2)')
    parser.add_argument('--profiler_warmup', type=int, default=2,
                        help='Profiler schedule: warmup steps (default: 2)')
    parser.add_argument('--profiler_active', type=int, default=6,
                        help='Profiler schedule: active tracing steps (default: 6)')
    parser.add_argument('--profiler_repeat', type=int, default=1,
                        help='Profiler schedule: repeat cycles (default: 1)')
    parser.add_argument('--profiler_record_shapes', action='store_true', default=True,
                        help='Record tensor shapes in profiler (default: True)')
    parser.add_argument('--profiler_with_stack', action='store_true', default=False,
                        help='Record stack traces in profiler (default: False)')
    parser.add_argument('--profiler_with_flops', action='store_true', default=True,
                        help='Estimate FLOPs for operators (default: True)')

    # torch.compile 优化
    parser.add_argument('--use_compile', action='store_true', default=False,
                        help='Use torch.compile to optimize the model (default: False)')
    parser.add_argument('--compile_mode', type=str, default='default',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode (default: default)')

    # WandB arguments
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='robotwin_nemodit',
                        help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='WandB entity (username or team name)')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='WandB run name')

    # Image preprocessing
    parser.add_argument('--no_resize', action='store_true', default=True,
                        help='Skip image resizing, use original size')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size for vision backbone (default: 224)')

    # DeepSpeed adds its own args (--local_rank, --deepspeed_config, etc.)
    parser = deepspeed.add_config_arguments(parser)

    args, _ = parser.parse_known_args()
    return args


def prepare_dataloader(args, rank, world_size):
    """Prepare distributed dataloader with DistributedSampler."""

    if args.no_resize:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    dataset = RobotDataset(
        data_path=args.data_path,
        future_action_window=args.future_action_window,
        past_action_window=args.past_action_window,
        transform=transform,
        num_cameras=args.num_cameras,
        use_both_arms=args.use_both_arms,
        action_type=args.action_type,
        quat_convention=args.quat_convention,
        n_obs_steps=args.n_obs_steps,
    )

    # DistributedSampler 确保每张卡看到不同的数据子集
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader, dataset, sampler


def create_model(args):
    """Create ActionModel with vision conditioning."""
    model = ActionModel(
        token_size=args.token_size,
        model_type=args.model_type,
        in_channels=args.action_dim,
        future_action_window_size=args.future_action_window,
        past_action_window_size=args.past_action_window,
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        use_vision_condition=True,
        vision_backbone_type=args.vision_backbone,
        vision_pretrained=args.vision_pretrained,
        num_cameras=args.num_cameras,
        freeze_vision_backbone=args.freeze_vision,
        adapter_type=args.adapter_type,
        class_dropout_prob=args.dropout_prob,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        temporal_agg=args.temporal_agg,
    )
    return model


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch,
                                warmup_type='linear'):
    """
    创建带有 warmup 的 cosine annealing 学习率调度器 (per-step)。

    DeepSpeed 的 scheduler 按 step 更新，因此这里转换为 step-based。
    """
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            if warmup_type == 'linear':
                return (current_step + 1) / warmup_steps
            else:  # cosine warmup
                return 0.5 * (1 - math.cos(math.pi * (current_step + 1) / warmup_steps))
        else:
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint_deepspeed(engine, epoch, global_step, args, ema_model=None, tag=None):
    """Save DeepSpeed checkpoint (handles ZeRO states, optimizer, etc.)."""
    if tag is None:
        tag = f"epoch_{epoch}"

    # DeepSpeed 会自动在 checkpoint_dir 下创建 tag 子目录
    client_state = {
        'epoch': epoch,
        'global_step': global_step,
        'args': vars(args),
    }
    if ema_model is not None:
        client_state['ema_state_dict'] = ema_model.state_dict()

    engine.save_checkpoint(args.checkpoint_dir, tag=tag, client_state=client_state)

    if engine.local_rank == 0:
        print(f"DeepSpeed checkpoint saved: {args.checkpoint_dir}/{tag}")


def load_checkpoint_deepspeed(engine, args, ema_model=None):
    """Load DeepSpeed checkpoint and return (epoch, global_step)."""
    load_dir = args.resume
    _, client_state = engine.load_checkpoint(load_dir)

    epoch = client_state.get('epoch', 0)
    global_step = client_state.get('global_step', 0)

    if ema_model is not None and 'ema_state_dict' in client_state:
        ema_model.load_state_dict(client_state['ema_state_dict'])
        if engine.local_rank == 0:
            print(f"EMA model restored (decay={ema_model.decay:.6f}, step={ema_model.optimization_step})")

    if engine.local_rank == 0:
        print(f"Resumed from epoch {epoch}, global step {global_step}")

    return epoch, global_step


def save_inference_checkpoint(model, epoch, global_step, args, ema_model=None, filename=None):
    """
    保存用于推理的标准 PyTorch checkpoint (仅 rank 0)。

    与原始 train.py 的 checkpoint 格式兼容，可直接用于 eval.py 和 deploy。
    """
    if filename is None:
        filename = f'{epoch}.pt'

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, filename)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'args': vars(args),
    }
    if ema_model is not None:
        checkpoint['ema_state_dict'] = ema_model.state_dict()

    torch.save(checkpoint, checkpoint_path)
    print(f"Inference checkpoint saved: {checkpoint_path}")


def train():
    """Main DeepSpeed distributed training loop."""

    args = parse_args()

    # Auto-set action_dim
    if args.action_type == 'endpose':
        args.action_dim = 20 if args.use_both_arms else 10
    else:
        args.action_dim = 14 if args.use_both_arms else 7

    # DeepSpeed initializes distributed backend automatically
    deepspeed.init_distributed()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    is_main = (rank == 0)

    if is_main:
        print(f"DeepSpeed distributed training: {world_size} GPUs")
        print(f"Action type: {args.action_type}, Action dim: {args.action_dim} "
              f"({'dual arm' if args.use_both_arms else 'single arm'})")
        print(f"Temporal: n_obs_steps={args.n_obs_steps}, n_action_steps={args.n_action_steps}, "
              f"future_action_window={args.future_action_window}, temporal_agg={args.temporal_agg}")
        print(f"Batch size per GPU: {args.batch_size}, "
              f"Effective batch size: {args.batch_size * world_size}")

    # WandB (rank 0 only)
    wandb_logger = None
    if args.use_wandb and is_main:
        run_name = args.wandb_name or f"{os.path.basename(args.checkpoint_dir)}_{args.model_type}_ds"
        wandb_logger = WandbLogger(
            project_name=args.wandb_project,
            run_name=run_name,
            config={**vars(args), 'world_size': world_size},
            entity=args.wandb_entity,
        )

    # Dataloader with DistributedSampler
    if is_main:
        print("Loading dataset...")
    dataloader, dataset, sampler = prepare_dataloader(args, rank, world_size)
    if is_main:
        print(f"Dataset size: {len(dataset)}, Batches per GPU: {len(dataloader)}")

    # Create model
    model = create_model(args)

    # torch.compile optimization (before DeepSpeed wrapping)
    if args.use_compile:
        if is_main:
            print(f"Applying torch.compile (mode={args.compile_mode})...")
        model = torch.compile(model, mode=args.compile_mode)
        if is_main:
            print("torch.compile applied successfully")

    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer (will be wrapped by DeepSpeed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    # LR scheduler (step-based for DeepSpeed)
    steps_per_epoch = len(dataloader)
    if args.warmup_epochs > 0:
        scheduler = get_warmup_cosine_scheduler(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            warmup_type=args.warmup_type,
        )
        if is_main:
            print(f"Using warmup: {args.warmup_epochs} epochs ({args.warmup_type})")
    else:
        total_steps = args.epochs * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps,
        )

    # Load DeepSpeed config
    ds_config_path = args.deepspeed_config
    if os.path.exists(ds_config_path):
        with open(ds_config_path, 'r') as f:
            ds_config = json.load(f)
        if is_main:
            print(f"Loaded DeepSpeed config from: {ds_config_path}")
    else:
        # 使用默认 DeepSpeed 配置
        ds_config = {
            "train_batch_size": args.batch_size * world_size,
            "train_micro_batch_size_per_gpu": args.batch_size,
            "gradient_clipping": args.grad_clip,
            "zero_optimization": {
                "stage": 1,
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            "zero_allow_untested_optimizer": True,
        }
        if is_main:
            print(f"Warning: {ds_config_path} not found, using default DeepSpeed config (ZeRO-1 + FP16)")

    # Initialize DeepSpeed engine
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_config,
    )

    if is_main:
        print(f"DeepSpeed engine initialized (ZeRO stage: "
              f"{ds_config.get('zero_optimization', {}).get('stage', 0)})")

    # EMA model (maintained on rank 0; uses the underlying module, not the engine)
    ema_model = None
    if args.use_ema:
        ema_model = EMAModel(
            engine.module,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            max_value=args.ema_max_value,
        )
        if is_main:
            print(f"Using EMA (inv_gamma={args.ema_inv_gamma}, power={args.ema_power}, "
                  f"max_value={args.ema_max_value})")

    # Resume
    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        start_epoch, global_step = load_checkpoint_deepspeed(engine, args, ema_model)

    # Setup profiler (rank 0 only, to avoid redundant traces)
    profiler = None
    if args.use_profiler and is_main:
        profiler_log_dir = os.path.join(
            args.profiler_dir, f"{os.path.basename(args.checkpoint_dir)}_rank{rank}"
        )
        os.makedirs(profiler_log_dir, exist_ok=True)
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=args.profiler_wait,
                warmup=args.profiler_warmup,
                active=args.profiler_active,
                repeat=args.profiler_repeat,
            ),
            on_trace_ready=tensorboard_trace_handler(profiler_log_dir),
            record_shapes=args.profiler_record_shapes,
            profile_memory=True,
            with_stack=args.profiler_with_stack,
            with_flops=args.profiler_with_flops,
        )
        print(f"Profiler enabled: traces -> {profiler_log_dir}")
        print(f"  View with: tensorboard --logdir={profiler_log_dir}")

    # Training loop
    if is_main:
        print("Starting DeepSpeed training...")

    if profiler is not None:
        profiler.start()

    for epoch in range(start_epoch, args.epochs):
        # 设置 sampler 的 epoch 以确保每个 epoch 的 shuffle 不同
        sampler.set_epoch(epoch)
        engine.train()

        epoch_loss = 0.0
        if is_main:
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        else:
            progress_bar = dataloader

        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            with record_function("data_transfer"):
                images = batch['images'].to(device)
                state = batch['state'].to(device)
                actions = batch['actions'].to(device)

            # Forward pass — DeepSpeed handles FP16/BF16 casting internally
            with record_function("forward"):
                loss = engine.module.loss(x=actions, images=images, state=state)

            # Backward pass — DeepSpeed handles gradient scaling and allreduce
            with record_function("backward"):
                engine.backward(loss)

            # Optimizer step — DeepSpeed handles gradient clipping and ZeRO updates
            with record_function("optimizer_step"):
                engine.step()

            # EMA update (on the underlying module)
            if ema_model is not None:
                with record_function("ema_update"):
                    ema_model.step(engine.module)

            epoch_loss += loss.item()
            global_step += 1

            # WandB logging (rank 0)
            if wandb_logger:
                log_dict = {
                    'train/loss': loss.item(),
                    'train/lr': engine.get_lr()[0] if hasattr(engine, 'get_lr') else optimizer.param_groups[0]["lr"],
                    'train/epoch': epoch + 1,
                    'train/global_step': global_step,
                }
                if ema_model is not None:
                    log_dict['train/ema_decay'] = ema_model.decay
                wandb_logger.log(log_dict, step=global_step)

            if is_main:
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}',
                    'lr': f'{engine.get_lr()[0]:.6f}' if hasattr(engine, 'get_lr') else 'N/A',
                })

            # Profiler step
            if profiler is not None:
                profiler.step()

        # Epoch summary
        avg_epoch_loss = epoch_loss / len(dataloader)
        if is_main:
            print(f"Epoch {epoch + 1} finished. Avg Loss: {avg_epoch_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint_deepspeed(engine, epoch + 1, global_step, args, ema_model)
            # 同时保存标准 PyTorch checkpoint 用于推理 (仅 rank 0)
            if is_main:
                save_inference_checkpoint(
                    engine.module, epoch + 1, global_step, args, ema_model,
                )

    # Stop profiler
    if profiler is not None:
        profiler.stop()
        print("\n" + "=" * 80)
        print("PROFILER SUMMARY — CUDA time")
        print("=" * 80)
        print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=30))
        if args.profiler_with_flops:
            print("\n" + "=" * 80)
            print("PROFILER SUMMARY — FLOPs")
            print("=" * 80)
            print(profiler.key_averages().table(sort_by="flops", row_limit=20))

    # Final save
    if is_main:
        print("Training completed!")
    save_checkpoint_deepspeed(engine, args.epochs, global_step, args, ema_model, tag="final")
    if is_main:
        save_inference_checkpoint(
            engine.module, args.epochs, global_step, args, ema_model, filename='final.pt',
        )

    if wandb_logger:
        wandb_logger.finish()


if __name__ == '__main__':
    train()

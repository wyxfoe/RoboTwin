#!/bin/bash
# ============================================================================
# NemoDiT DeepSpeed Distributed Training Script for RoboTwin
# ============================================================================
#
# Usage:
#   bash train_deepspeed.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${gpu_ids}
#
# Examples:
#   # 4卡训练
#   bash train_deepspeed.sh beat_block_hammer default 50 0 "0,1,2,3"
#
#   # 2卡训练
#   bash train_deepspeed.sh beat_block_hammer default 50 0 "0,1"
#
#   # 单卡训练 (仍使用 DeepSpeed 的 ZeRO 优化)
#   bash train_deepspeed.sh beat_block_hammer default 50 0 "0"
#
# Arguments:
#   task_name       - Name of the task to train on
#   task_config     - Task configuration (e.g., default)
#   expert_data_num - Number of expert demonstrations to use
#   seed            - Random seed
#   gpu_ids         - Comma-separated GPU IDs (e.g., "0,1,2,3")
#
# ============================================================================

# Parse command line arguments
task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_ids=${5:-"0,1,2,3"}

# ============================================================================
# Training Configuration
# ============================================================================

# Model configuration
model_type="DiT-B"
vision_backbone="resnet50"
adapter_type="mlp"

# Action configuration
action_type="joint"
use_both_arms="--use_both_arms"
quat_convention="wxyz"

# Temporal configuration
n_obs_steps=2
n_action_steps=8
future_action_window=12
past_action_window=0
temporal_agg="concat"

# Diffusion configuration
diffusion_steps=100
noise_schedule="squaredcos_cap_v2"

# Training configuration
epochs=500
batch_size=32                 # Per-GPU batch size
lr=1e-4
weight_decay=0.01
grad_clip=1.0
num_workers=4
save_every=50

# Camera configuration
num_cameras=4

# DeepSpeed configuration
ds_config="ds_config.json"   # DeepSpeed JSON config file

# Profiler configuration
use_profiler=""               # Set to "--use_profiler" to enable
profiler_dir="profiler_logs"

# ============================================================================
# WandB Configuration
# ============================================================================
use_wandb="--use_wandb"
wandb_project="robotwin_nemodit"
wandb_entity=""

# ============================================================================
# Setup
# ============================================================================

# Count number of GPUs
IFS=',' read -ra GPU_ARRAY <<< "${gpu_ids}"
num_gpus=${#GPU_ARRAY[@]}

# Set visible GPUs
export CUDA_VISIBLE_DEVICES=${gpu_ids}

# Data path
data_path="../../data/${task_name}/demo_clean_${expert_data_num}/data"
# Checkpoint directory
checkpoint_dir="checkpoints/${task_name}-${task_config}-${expert_data_num}-${seed}-ds"

# Print training info
echo -e "\033[33m============================================\033[0m"
echo -e "\033[33m[NemoDiT] DeepSpeed Distributed Training\033[0m"
echo -e "\033[33m============================================\033[0m"
echo -e "\033[33m[NemoDiT] GPUs: ${gpu_ids} (${num_gpus} GPUs)\033[0m"
echo -e "\033[33m[NemoDiT] Task: ${task_name}\033[0m"
echo -e "\033[33m[NemoDiT] Config: ${task_config}\033[0m"
echo -e "\033[33m[NemoDiT] Expert Data: ${expert_data_num}\033[0m"
echo -e "\033[33m[NemoDiT] Seed: ${seed}\033[0m"
echo -e "\033[33m[NemoDiT] Model: ${model_type}\033[0m"
echo -e "\033[33m[NemoDiT] Action Type: ${action_type}\033[0m"
echo -e "\033[33m[NemoDiT] Batch/GPU: ${batch_size}, Effective: $((batch_size * num_gpus))\033[0m"
echo -e "\033[33m[NemoDiT] DeepSpeed Config: ${ds_config}\033[0m"
echo -e "\033[33m============================================\033[0m"

# ============================================================================
# Run DeepSpeed Training
# ============================================================================

deepspeed --num_gpus ${num_gpus} train_deepspeed.py \
    --deepspeed_config ${ds_config} \
    --data_path ${data_path} \
    --num_cameras ${num_cameras} \
    ${use_both_arms} \
    --action_type ${action_type} \
    --quat_convention ${quat_convention} \
    --model_type ${model_type} \
    --vision_backbone ${vision_backbone} \
    --vision_pretrained \
    --adapter_type ${adapter_type} \
    --n_obs_steps ${n_obs_steps} \
    --n_action_steps ${n_action_steps} \
    --future_action_window ${future_action_window} \
    --past_action_window ${past_action_window} \
    --temporal_agg ${temporal_agg} \
    --diffusion_steps ${diffusion_steps} \
    --noise_schedule ${noise_schedule} \
    --epochs ${epochs} \
    --batch_size ${batch_size} \
    --lr ${lr} \
    --weight_decay ${weight_decay} \
    --grad_clip ${grad_clip} \
    ${use_wandb} \
    --wandb_project ${wandb_project} \
    --wandb_entity "${wandb_entity}" \
    --num_workers ${num_workers} \
    --checkpoint_dir ${checkpoint_dir} \
    --save_every ${save_every} \
    ${use_profiler} \
    --profiler_dir ${profiler_dir}

echo -e "\033[32m[NemoDiT] DeepSpeed training completed!\033[0m"
echo -e "\033[32m[NemoDiT] Checkpoints saved to: ${checkpoint_dir}\033[0m"

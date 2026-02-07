#!/bin/bash

# AudioX Evaluation Script for RoboTwin
#
# Usage:
#   bash policy/AudioX/eval.sh <task_name> <task_config> <model_config> <ckpt_path> <seed> <gpu_id>
#
# Example:
#   bash policy/AudioX/eval.sh place_object_scale demo_clean configs/audiox_robot.json checkpoints/best.ckpt 0 0
#
# Arguments:
#   task_name:    Name of the task to evaluate (e.g., place_object_scale)
#   task_config:  Task configuration file name without .yml (e.g., demo_clean)
#   model_config: Path to AudioX model config JSON
#   ckpt_path:    Path to fine-tuned checkpoint
#   seed:         Random seed for evaluation
#   gpu_id:       GPU device ID

policy_name=AudioX
task_name=${1}
task_config=${2}
model_config=${3}
ckpt_path=${4}
seed=${5}
gpu_id=${6}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..  # move to RoboTwin root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --model_config ${model_config} \
    --ckpt_path ${ckpt_path} \
    --ckpt_setting ${ckpt_path} \
    --seed ${seed} \
    --policy_name ${policy_name}

#!/bin/bash
# ============================================================
# AudioX Evaluation Script for RoboTwin
# ============================================================
#
# Usage:
#   bash policy/AudioX/eval.sh <task_name> <task_config> <seed> <gpu_id>
#
# Example (single task):
#   bash policy/AudioX/eval.sh beat_block_hammer demo_clean 0 0
#
# Example (batch evaluation, all tasks):
#   for task in beat_block_hammer place_object_scale pick_cube_bottle; do
#       bash policy/AudioX/eval.sh $task demo_clean 0 0
#   done
#
# Prerequisites:
#   1. Fine-tune AudioX:  python policy/AudioX/train_robotx.py --config ...
#   2. Set paths in deploy_policy.yml:
#        ckpt_path:         checkpoints/robotx/robotx_final.pt
#        action_stats_path: checkpoints/robotx/action_stats.pt
#        action_head_path:  checkpoints/robotx/action_head.pt  (optional)
#   3. Ensure AUDIOX_PATH env var points to the AudioX- source directory
#      (or place AudioX- as a sibling to the RoboTwin project root)
# ============================================================

# == keep unchanged ==
policy_name=AudioX
task_name=${1}
task_config=${2}
seed=${3}
gpu_id=${4}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..  # move to RoboTwin root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting audiox \
    --seed ${seed} \
    --policy_name ${policy_name}

#!/bin/bash
# =============================================================================
# HDF5 vs Zarr vs LeRobot v2.1 Data Format Benchmark for Diffusion Policy
#
# DP's native pipeline is:
#   raw HDF5 episodes  ->  process_data.py  ->  single merged .zarr  ->  training
#
# This script runs the conversion once (zarr -> hdf5 + zarr -> lerobot) then
# benchmarks all three formats under both lazy and eager loading strategies.
#
# Usage:
#   bash run_benchmark.sh <task_name> <task_config> <expert_data_num> [gpu_id]
#
# Example:
#   bash run_benchmark.sh block_hammer_beat ActivePick 50 0
# =============================================================================

set -e

task_name=${1:?"Usage: $0 <task_name> <task_config> <expert_data_num> [gpu_id]"}
task_config=${2:?"Missing task_config"}
expert_data_num=${3:?"Missing expert_data_num"}
gpu_id=${4:-0}

export CUDA_VISIBLE_DEVICES=${gpu_id}

ZARR_PATH="./data/${task_name}-${task_config}-${expert_data_num}.zarr"
HDF5_PATH="./data/${task_name}-${task_config}-${expert_data_num}.hdf5"
LEROBOT_ROOT="./processed_data_lerobot"
LEROBOT_REPO_ID="robotwin/dp_${task_name}_${task_config}_${expert_data_num}"
LEROBOT_MODE="${LEROBOT_MODE:-video}"
LEROBOT_FPS="${LEROBOT_FPS:-25}"

echo "============================================="
echo "  DP Data Format Benchmark"
echo "============================================="
echo "  Task:      ${task_name}"
echo "  Config:    ${task_config}"
echo "  Episodes:  ${expert_data_num}"
echo "  Zarr:      ${ZARR_PATH}"
echo "  HDF5:      ${HDF5_PATH}"
echo "  LeRobot:   ${LEROBOT_ROOT}/${LEROBOT_REPO_ID} (${LEROBOT_MODE})"
echo "============================================="

# --- Step 1: Ensure DP zarr exists ---
if [ ! -d "${ZARR_PATH}" ]; then
    echo ""
    echo "[Step 1] Building DP zarr via process_data.py ..."
    python3 process_data.py ${task_name} ${task_config} ${expert_data_num}
else
    echo ""
    echo "[Step 1] DP zarr already exists, skipping."
fi

# --- Step 2: Convert to HDF5 ---
if [ ! -f "${HDF5_PATH}" ]; then
    echo ""
    echo "[Step 2a] Converting zarr -> HDF5 ..."
    python3 convert_dp_data.py \
        --zarr_path "${ZARR_PATH}" \
        --format hdf5 \
        --out_path "${HDF5_PATH}"
else
    echo "[Step 2a] HDF5 already exists, skipping."
fi

# --- Step 2b: Convert to LeRobot v2.1 ---
if [ ! -d "${LEROBOT_ROOT}/${LEROBOT_REPO_ID}" ]; then
    echo ""
    echo "[Step 2b] Converting zarr -> LeRobot v2.1 (${LEROBOT_MODE}) ..."
    python3 convert_dp_data.py \
        --zarr_path "${ZARR_PATH}" \
        --format lerobot \
        --repo_id "${LEROBOT_REPO_ID}" \
        --root "${LEROBOT_ROOT}" \
        --mode "${LEROBOT_MODE}" \
        --fps "${LEROBOT_FPS}"
else
    echo "[Step 2b] LeRobot dataset already exists, skipping."
fi

# --- Step 3: Run benchmark ---
echo ""
echo "[Step 3] Running 3x2 benchmark ..."
python3 benchmark_data_formats.py \
    --zarr_path "${ZARR_PATH}" \
    --hdf5_path "${HDF5_PATH}" \
    --lerobot_repo_id "${LEROBOT_REPO_ID}" \
    --lerobot_root "${LEROBOT_ROOT}" \
    --lerobot_fps "${LEROBOT_FPS}" \
    --horizon 8 \
    --batch_size 128 \
    --num_warmup 5 \
    --num_iters 50 \
    --output "benchmark_results_dp_3x2.json"

echo ""
echo "============================================="
echo "  Done. Results: benchmark_results_dp_3x2.json"
echo "============================================="

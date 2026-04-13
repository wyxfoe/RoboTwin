#!/bin/bash
# =============================================================================
# HDF5 vs Zarr vs LeRobot v2.1 Data Format Benchmark for ACT Policy
#
# Tests three Zarr configurations and one LeRobot v2.1 configuration against HDF5:
#   1. Zarr (no compression)     — raw I/O baseline
#   2. Zarr (DP preset)          — Blosc(zstd, clevel=3, shuffle=SHUFFLE),
#                                  same as Diffusion Policy process_data.py
#   3. Zarr (DP-memory preset)   — Blosc(lz4, clevel=5, shuffle=NOSHUFFLE),
#                                  same as DP ReplayBuffer in-memory default
#   4. LeRobot v2.1 (video mode) — MP4-encoded frames + parquet state/action
#
# Each configuration is evaluated under both lazy and eager loading strategies
# via the 3x2 matrix in benchmark_data_formats.py.
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

HDF5_DIR="./processed_data/sim-${task_name}/${task_config}-${expert_data_num}"
ZARR_BASE="./processed_data_zarr/sim-${task_name}/${task_config}-${expert_data_num}"
ZARR_DIR_NONE="${ZARR_BASE}_none"
ZARR_DIR_DP="${ZARR_BASE}_dp"
ZARR_DIR_DP_MEM="${ZARR_BASE}_dp_memory"

# LeRobot uses its own home directory; we keep a local root to avoid polluting $HOME.
LEROBOT_ROOT="./processed_data_lerobot"
LEROBOT_REPO_ID="robotwin/act_${task_name}_${task_config}_${expert_data_num}"
LEROBOT_MODE="${LEROBOT_MODE:-video}"
LEROBOT_FPS="${LEROBOT_FPS:-25}"

echo "============================================="
echo "  Data Format Benchmark: HDF5 vs Zarr"
echo "============================================="
echo "  Task:       ${task_name}"
echo "  Config:     ${task_config}"
echo "  Episodes:   ${expert_data_num}"
echo "  GPU:        ${gpu_id}"
echo "  HDF5 dir:   ${HDF5_DIR}"
echo "============================================="

# --- Step 1: Ensure HDF5 data exists ---
if [ ! -d "${HDF5_DIR}" ]; then
    echo ""
    echo "[Step 1] Processing raw data to HDF5 ..."
    python3 process_data.py ${task_name} ${task_config} ${expert_data_num}
else
    echo ""
    echo "[Step 1] HDF5 data already exists at ${HDF5_DIR}, skipping."
fi

# --- Step 2: Convert to Zarr variants ---
convert_if_missing() {
    local label=$1
    local dir=$2
    local compressor=$3
    if [ ! -d "${dir}" ]; then
        echo ""
        echo "[Convert] ${label} ..."
        python3 convert_hdf5_to_zarr.py \
            --dataset_dir "${HDF5_DIR}" \
            --output_dir "${dir}" \
            --num_episodes ${expert_data_num} \
            --compressor ${compressor} \
            --layout per_episode
    else
        echo "[Convert] ${label} already exists, skipping."
    fi
}

convert_if_missing "Zarr (no compression)"     "${ZARR_DIR_NONE}"   "none"
convert_if_missing "Zarr (DP: zstd clevel=3)"  "${ZARR_DIR_DP}"     "dp"
convert_if_missing "Zarr (DP-memory: lz4)"     "${ZARR_DIR_DP_MEM}" "dp-memory"

# --- Step 2b: Convert to LeRobot v2.1 ---
if [ ! -d "${LEROBOT_ROOT}/${LEROBOT_REPO_ID}" ]; then
    echo ""
    echo "[Convert] LeRobot v2.1 (${LEROBOT_MODE}) ..."
    python3 convert_hdf5_to_lerobot.py \
        --dataset_dir "${HDF5_DIR}" \
        --repo_id "${LEROBOT_REPO_ID}" \
        --num_episodes ${expert_data_num} \
        --mode "${LEROBOT_MODE}" \
        --fps "${LEROBOT_FPS}" \
        --root "${LEROBOT_ROOT}"
else
    echo "[Convert] LeRobot v2.1 already exists, skipping."
fi

# --- Step 3: Run benchmarks ---
#
# Each invocation pairs HDF5 against one Zarr variant; LeRobot v2.1 is included
# in every run so the 3x2 matrix is complete end-to-end.
run_benchmark() {
    local label=$1
    local zarr_dir=$2
    local output=$3
    echo ""
    echo "[Benchmark] HDF5 vs ${label} vs LeRobot ..."
    python3 benchmark_data_formats.py \
        --hdf5_dir "${HDF5_DIR}" \
        --zarr_dir "${zarr_dir}" \
        --lerobot_repo_id "${LEROBOT_REPO_ID}" \
        --lerobot_root "${LEROBOT_ROOT}" \
        --num_episodes ${expert_data_num} \
        --batch_size 8 \
        --num_warmup 5 \
        --num_iters 50 \
        --output "${output}"
}

run_benchmark "Zarr (no compression)"     "${ZARR_DIR_NONE}"   "benchmark_results_none.json"
run_benchmark "Zarr (DP: zstd clevel=3)"  "${ZARR_DIR_DP}"     "benchmark_results_dp.json"
run_benchmark "Zarr (DP-memory: lz4)"     "${ZARR_DIR_DP_MEM}" "benchmark_results_dp_memory.json"

echo ""
echo "============================================="
echo "  Benchmark complete! Results saved to:"
echo "    benchmark_results_none.json"
echo "    benchmark_results_dp.json"
echo "    benchmark_results_dp_memory.json"
echo "============================================="

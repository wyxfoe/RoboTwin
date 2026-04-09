#!/bin/bash
# =============================================================================
# HDF5 vs Zarr Data Format Benchmark for ACT Policy
#
# Tests three Zarr configurations against HDF5:
#   1. Zarr (no compression)   — raw I/O baseline
#   2. Zarr (DP preset)        — Blosc(zstd, clevel=3, shuffle=SHUFFLE),
#                                same as Diffusion Policy process_data.py
#   3. Zarr (DP-memory preset) — Blosc(lz4, clevel=5, shuffle=NOSHUFFLE),
#                                same as DP ReplayBuffer in-memory default
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

# --- Step 3: Run benchmarks ---
run_benchmark() {
    local label=$1
    local zarr_dir=$2
    local output=$3
    echo ""
    echo "[Benchmark] HDF5 vs ${label} ..."
    python3 benchmark_data_formats.py \
        --hdf5_dir "${HDF5_DIR}" \
        --zarr_dir "${zarr_dir}" \
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

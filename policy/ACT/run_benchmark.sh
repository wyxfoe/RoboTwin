#!/bin/bash
# =============================================================================
# HDF5 vs Zarr Data Format Benchmark for ACT Policy
#
# Usage:
#   bash run_benchmark.sh <task_name> <task_config> <expert_data_num> [gpu_id]
#
# Example:
#   bash run_benchmark.sh block_hammer_beat ActivePick 50 0
#
# Steps:
#   1. Process raw data into HDF5 (if not already done)
#   2. Convert HDF5 to Zarr (with multiple compression options)
#   3. Run the benchmark suite
# =============================================================================

set -e

task_name=${1:?"Usage: $0 <task_name> <task_config> <expert_data_num> [gpu_id]"}
task_config=${2:?"Missing task_config"}
expert_data_num=${3:?"Missing expert_data_num"}
gpu_id=${4:-0}

export CUDA_VISIBLE_DEVICES=${gpu_id}

HDF5_DIR="./processed_data/sim-${task_name}/${task_config}-${expert_data_num}"
ZARR_DIR_NONE="./processed_data_zarr/sim-${task_name}/${task_config}-${expert_data_num}_no_compress"
ZARR_DIR_LZ4="./processed_data_zarr/sim-${task_name}/${task_config}-${expert_data_num}_lz4"

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

# --- Step 2: Convert to Zarr (no compression) ---
if [ ! -d "${ZARR_DIR_NONE}" ]; then
    echo ""
    echo "[Step 2a] Converting HDF5 -> Zarr (no compression) ..."
    python3 convert_hdf5_to_zarr.py \
        --dataset_dir "${HDF5_DIR}" \
        --output_dir "${ZARR_DIR_NONE}" \
        --num_episodes ${expert_data_num} \
        --compressor none
else
    echo ""
    echo "[Step 2a] Zarr (no compress) already exists, skipping."
fi

# --- Step 2b: Convert to Zarr (LZ4) ---
if [ ! -d "${ZARR_DIR_LZ4}" ]; then
    echo ""
    echo "[Step 2b] Converting HDF5 -> Zarr (LZ4) ..."
    python3 convert_hdf5_to_zarr.py \
        --dataset_dir "${HDF5_DIR}" \
        --output_dir "${ZARR_DIR_LZ4}" \
        --num_episodes ${expert_data_num} \
        --compressor lz4
else
    echo ""
    echo "[Step 2b] Zarr (LZ4) already exists, skipping."
fi

# --- Step 3: Run benchmarks ---
echo ""
echo "[Step 3] Running benchmark: HDF5 vs Zarr (no compression) ..."
python3 benchmark_data_formats.py \
    --hdf5_dir "${HDF5_DIR}" \
    --zarr_dir "${ZARR_DIR_NONE}" \
    --num_episodes ${expert_data_num} \
    --batch_size 8 \
    --num_warmup 5 \
    --num_iters 50 \
    --output "benchmark_results_no_compress.json"

echo ""
echo "[Step 4] Running benchmark: HDF5 vs Zarr (LZ4) ..."
python3 benchmark_data_formats.py \
    --hdf5_dir "${HDF5_DIR}" \
    --zarr_dir "${ZARR_DIR_LZ4}" \
    --num_episodes ${expert_data_num} \
    --batch_size 8 \
    --num_warmup 5 \
    --num_iters 50 \
    --output "benchmark_results_lz4.json"

echo ""
echo "============================================="
echo "  Benchmark complete!"
echo "  Results saved to:"
echo "    benchmark_results_no_compress.json"
echo "    benchmark_results_lz4.json"
echo "============================================="

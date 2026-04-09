"""
Convert HDF5 episode data to Zarr format for benchmarking comparison.

Supports two storage layouts:
  --layout per_episode   (default) Each episode is a separate .zarr store,
                         mirroring the per-file HDF5 layout used by ACT.
  --layout merged        All episodes concatenated into a single .zarr store
                         with a meta/episode_ends array, matching the format
                         used by Diffusion Policy (DP).

Compressor presets mirror real usage in the RoboTwin codebase:
  "dp"        — Blosc(zstd, clevel=3, shuffle=SHUFFLE), same as DP process_data.py
  "dp-memory" — Blosc(lz4,  clevel=5, shuffle=NOSHUFFLE), DP ReplayBuffer default
  "none"      — no compression (raw I/O baseline)
  "lz4"       — Blosc(lz4,  clevel=5, shuffle=BITSHUFFLE)
  "zstd"      — Blosc(zstd, clevel=3, shuffle=BITSHUFFLE)

Usage:
    # Per-episode layout (for ACT-style random-episode access)
    python convert_hdf5_to_zarr.py --dataset_dir <hdf5_dir> --output_dir <zarr_dir> \\
        --num_episodes 50 --compressor dp --layout per_episode

    # Merged layout (DP-style single Zarr store)
    python convert_hdf5_to_zarr.py --dataset_dir <hdf5_dir> --output_dir <zarr_dir> \\
        --num_episodes 50 --compressor dp --layout merged
"""

import os
import argparse
import time
import math

import h5py
import numpy as np
import zarr
from numcodecs import Blosc


# ---------------------------------------------------------------------------
# Compressor presets
# ---------------------------------------------------------------------------

COMPRESSOR_PRESETS = {
    "dp":        Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),      # DP process_data.py
    "dp-memory": Blosc(cname="lz4",  clevel=5, shuffle=Blosc.NOSHUFFLE),    # DP ReplayBuffer default
    "none":      None,
    "lz4":       Blosc(cname="lz4",  clevel=5, shuffle=Blosc.BITSHUFFLE),
    "zstd":      Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
}


def resolve_compressor(name):
    if name in COMPRESSOR_PRESETS:
        return COMPRESSOR_PRESETS[name]
    raise ValueError(f"Unknown compressor preset: {name}. "
                     f"Choose from {list(COMPRESSOR_PRESETS.keys())}")


def get_optimal_chunks(shape, dtype, target_chunk_bytes=2e6):
    """Compute chunk sizes targeting ~2 MB per chunk (same heuristic as DP ReplayBuffer)."""
    itemsize = np.dtype(dtype).itemsize
    rshape = list(shape[::-1])
    split_idx = len(shape) - 1
    for i in range(len(shape) - 1):
        this_bytes = itemsize * np.prod(rshape[:i])
        next_bytes = itemsize * np.prod(rshape[:i + 1])
        if this_bytes <= target_chunk_bytes < next_bytes:
            split_idx = i
    rchunks = list(rshape[:split_idx])
    item_bytes = itemsize * max(np.prod(rshape[:split_idx]), 1)
    this_max = rshape[split_idx]
    next_len = min(this_max, math.ceil(target_chunk_bytes / item_bytes))
    rchunks.append(next_len)
    rchunks.extend([1] * (len(shape) - len(rchunks)))
    return tuple(rchunks[::-1])


# ---------------------------------------------------------------------------
# Per-episode layout (ACT style)
# ---------------------------------------------------------------------------

def convert_episode(hdf5_path, zarr_path, compressor):
    """Convert a single HDF5 episode to its own Zarr store."""
    with h5py.File(hdf5_path, "r") as f:
        store = zarr.DirectoryStore(zarr_path)
        root = zarr.group(store, overwrite=True)

        action = f["/action"][()]
        root.create_dataset("action", data=action, chunks=action.shape, compressor=compressor)

        obs = root.create_group("observations")
        qpos = f["/observations/qpos"][()]
        obs.create_dataset("qpos", data=qpos, chunks=qpos.shape, compressor=compressor)

        images = obs.create_group("images")
        for cam_name in ["cam_high", "cam_right_wrist", "cam_left_wrist"]:
            key = f"/observations/images/{cam_name}"
            if key in f:
                img = f[key][()]  # (T, H, W, C)
                chunks = (1,) + img.shape[1:]  # one frame per chunk for random access
                images.create_dataset(cam_name, data=img, chunks=chunks, compressor=compressor)

    return os.path.getsize(hdf5_path)


def convert_per_episode(args, compressor):
    """Convert each HDF5 episode to a separate .zarr store."""
    os.makedirs(args.output_dir, exist_ok=True)
    total_hdf5_size = 0
    for i in range(args.num_episodes):
        hdf5_path = os.path.join(args.dataset_dir, f"episode_{i}.hdf5")
        zarr_path = os.path.join(args.output_dir, f"episode_{i}.zarr")
        if not os.path.exists(hdf5_path):
            print(f"Warning: {hdf5_path} not found, skipping")
            continue
        hdf5_size = convert_episode(hdf5_path, zarr_path, compressor)
        total_hdf5_size += hdf5_size
        print(f"Converted episode {i} ({hdf5_size / 1e6:.1f} MB)")
    return total_hdf5_size


# ---------------------------------------------------------------------------
# Merged layout (DP style)
# ---------------------------------------------------------------------------

def convert_merged(args, compressor):
    """
    Concatenate all episodes into a single Zarr store with structure:
        data/
            action      (total_T, action_dim)
            qpos        (total_T, state_dim)
            cam_high    (total_T, C, H, W)   — NCHW like DP
            cam_right_wrist ...
            cam_left_wrist  ...
        meta/
            episode_ends (num_episodes,)
    """
    all_action, all_qpos = [], []
    all_images = {c: [] for c in ["cam_high", "cam_right_wrist", "cam_left_wrist"]}
    episode_ends = []
    total_hdf5_size = 0
    total_t = 0

    for i in range(args.num_episodes):
        hdf5_path = os.path.join(args.dataset_dir, f"episode_{i}.hdf5")
        if not os.path.exists(hdf5_path):
            print(f"Warning: {hdf5_path} not found, skipping")
            continue
        total_hdf5_size += os.path.getsize(hdf5_path)

        with h5py.File(hdf5_path, "r") as f:
            action = f["/action"][()]
            qpos = f["/observations/qpos"][()]
            ep_len = action.shape[0]
            all_action.append(action)
            all_qpos.append(qpos)
            for cam_name in all_images:
                img = f[f"/observations/images/{cam_name}"][()]
                # HWC -> CHW to match DP convention
                img = np.moveaxis(img, -1, 1)
                all_images[cam_name].append(img)
            total_t += ep_len
            episode_ends.append(total_t)
        print(f"Read episode {i} ({ep_len} steps)", end="\r")

    print()

    # Concatenate
    action_arr = np.concatenate(all_action, axis=0)
    qpos_arr = np.concatenate(all_qpos, axis=0).astype(np.float32)
    episode_ends_arr = np.array(episode_ends, dtype=np.int64)

    # Write merged Zarr store
    os.makedirs(args.output_dir, exist_ok=True)
    zarr_root = zarr.group(args.output_dir, overwrite=True)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    # Chunk sizes: 100 along time dimension, matching DP process_data.py
    chunk_t = 100

    zarr_data.create_dataset("action", data=action_arr, dtype="float32",
                             chunks=(chunk_t, action_arr.shape[1]),
                             compressor=compressor, overwrite=True)
    zarr_data.create_dataset("state", data=qpos_arr, dtype="float32",
                             chunks=(chunk_t, qpos_arr.shape[1]),
                             compressor=compressor, overwrite=True)

    for cam_name in all_images:
        img_arr = np.concatenate(all_images[cam_name], axis=0)
        chunks = (chunk_t,) + img_arr.shape[1:]
        zarr_data.create_dataset(cam_name, data=img_arr, chunks=chunks,
                                 compressor=compressor, overwrite=True)

    zarr_meta.create_dataset("episode_ends", data=episode_ends_arr,
                             dtype="int64", compressor=compressor, overwrite=True)

    return total_hdf5_size


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 episodes to Zarr format")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Directory containing episode_*.hdf5 files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for Zarr store(s)")
    parser.add_argument("--num_episodes", type=int, required=True,
                        help="Number of episodes to convert")
    parser.add_argument("--compressor", type=str, default="dp",
                        choices=list(COMPRESSOR_PRESETS.keys()),
                        help="Compression preset (default: dp)")
    parser.add_argument("--layout", type=str, default="per_episode",
                        choices=["per_episode", "merged"],
                        help="per_episode = one .zarr per episode (ACT style); "
                             "merged = single .zarr with episode_ends (DP style)")
    args = parser.parse_args()

    compressor = resolve_compressor(args.compressor)
    print(f"Layout:     {args.layout}")
    print(f"Compressor: {args.compressor} -> {compressor}")

    t0 = time.time()
    if args.layout == "per_episode":
        total_hdf5_size = convert_per_episode(args, compressor)
    else:
        total_hdf5_size = convert_merged(args, compressor)
    elapsed = time.time() - t0

    # Compute Zarr total size
    total_zarr_size = 0
    for dirpath, _, filenames in os.walk(args.output_dir):
        for fname in filenames:
            total_zarr_size += os.path.getsize(os.path.join(dirpath, fname))

    print(f"\n--- Conversion Summary ---")
    print(f"Layout:             {args.layout}")
    print(f"Compressor:         {args.compressor}")
    print(f"Episodes converted: {args.num_episodes}")
    print(f"HDF5 total size:    {total_hdf5_size / 1e6:.1f} MB")
    print(f"Zarr total size:    {total_zarr_size / 1e6:.1f} MB")
    print(f"Compression ratio:  {total_hdf5_size / max(total_zarr_size, 1):.2f}x")
    print(f"Conversion time:    {elapsed:.1f}s")


if __name__ == "__main__":
    main()

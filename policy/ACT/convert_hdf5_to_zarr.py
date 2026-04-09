"""
Convert HDF5 episode data to Zarr format for benchmarking comparison.

Usage:
    python convert_hdf5_to_zarr.py --dataset_dir <hdf5_dir> --output_dir <zarr_dir> --num_episodes <N>

The Zarr store mirrors the HDF5 structure:
    episode_{i}.zarr/
        action                      (T, action_dim)
        observations/
            qpos                    (T, state_dim)
            images/
                cam_high            (T, H, W, C)
                cam_right_wrist     (T, H, W, C)
                cam_left_wrist      (T, H, W, C)
"""

import os
import argparse
import time

import h5py
import numpy as np
import zarr
from numcodecs import Blosc


def convert_episode(hdf5_path, zarr_path, compressor=None):
    """Convert a single HDF5 episode to Zarr format."""
    with h5py.File(hdf5_path, "r") as f:
        store = zarr.DirectoryStore(zarr_path)
        root = zarr.group(store, overwrite=True)

        # Action data
        action = f["/action"][()]
        root.create_dataset("action", data=action, chunks=action.shape, compressor=compressor)

        # Observations group
        obs = root.create_group("observations")
        qpos = f["/observations/qpos"][()]
        obs.create_dataset("qpos", data=qpos, chunks=qpos.shape, compressor=compressor)

        # Image datasets — chunk per frame for random-access reads
        images = obs.create_group("images")
        for cam_name in ["cam_high", "cam_right_wrist", "cam_left_wrist"]:
            key = f"/observations/images/{cam_name}"
            if key in f:
                img = f[key][()]  # (T, H, W, C)
                # Chunk: one frame at a time for efficient single-timestep reads
                chunks = (1,) + img.shape[1:]
                images.create_dataset(cam_name, data=img, chunks=chunks, compressor=compressor)

    return os.path.getsize(hdf5_path)


def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 episodes to Zarr format")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing episode_*.hdf5 files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for Zarr stores")
    parser.add_argument("--num_episodes", type=int, required=True, help="Number of episodes to convert")
    parser.add_argument("--compressor", type=str, default="default",
                        choices=["default", "none", "lz4", "zstd", "blosc-lz4"],
                        help="Compression algorithm for Zarr datasets")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.compressor == "none":
        compressor = None
    elif args.compressor == "lz4":
        compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
    elif args.compressor == "zstd":
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    elif args.compressor == "blosc-lz4":
        compressor = Blosc(cname="lz4", clevel=1, shuffle=Blosc.NOSHUFFLE)
    else:
        compressor = None  # zarr default

    total_hdf5_size = 0
    t0 = time.time()

    for i in range(args.num_episodes):
        hdf5_path = os.path.join(args.dataset_dir, f"episode_{i}.hdf5")
        zarr_path = os.path.join(args.output_dir, f"episode_{i}.zarr")
        if not os.path.exists(hdf5_path):
            print(f"Warning: {hdf5_path} not found, skipping")
            continue
        hdf5_size = convert_episode(hdf5_path, zarr_path, compressor=compressor)
        total_hdf5_size += hdf5_size
        print(f"Converted episode {i} ({hdf5_size / 1e6:.1f} MB)")

    elapsed = time.time() - t0

    # Compute Zarr total size
    total_zarr_size = 0
    for dirpath, _, filenames in os.walk(args.output_dir):
        for fname in filenames:
            total_zarr_size += os.path.getsize(os.path.join(dirpath, fname))

    print(f"\n--- Conversion Summary ---")
    print(f"Episodes converted: {args.num_episodes}")
    print(f"HDF5 total size:    {total_hdf5_size / 1e6:.1f} MB")
    print(f"Zarr total size:    {total_zarr_size / 1e6:.1f} MB")
    print(f"Compression ratio:  {total_hdf5_size / max(total_zarr_size, 1):.2f}x")
    print(f"Conversion time:    {elapsed:.1f}s")


if __name__ == "__main__":
    main()

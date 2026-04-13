"""
Convert HDF5 episode data to LeRobot v2.1 dataset format.

LeRobot v2.1 stores episodes as:
  {HF_LEROBOT_HOME or --root}/{repo_id}/
      meta/info.json                    schema + fps + task list
      meta/stats.json                   per-key mean/std/min/max
      meta/episodes.jsonl               per-episode length + task ids
      data/chunk-000/episode_*.parquet  state/action/frame_index
      videos/chunk-000/observation.images.{cam}/episode_*.mp4   (if use_videos)
      images/chunk-000/observation.images.{cam}/episode_*/*.png (if not use_videos)

Two storage modes:
  --mode video    (default) MP4 videos via pyav; heavy decode, small on disk
  --mode image    PNG image folders; cheaper decode, much larger on disk

Usage:
    python convert_hdf5_to_lerobot.py \\
        --dataset_dir <hdf5_dir> \\
        --repo_id robotwin/act_benchmark \\
        --num_episodes 50 \\
        --mode video \\
        --fps 25
"""

import argparse
import os
import shutil
import time
from pathlib import Path

import h5py
import numpy as np


# LeRobot v2.1 treats each camera as a feature named "observation.images.<cam>".
# We keep ACT's three ALOHA cameras by default.
DEFAULT_CAMERAS = ["cam_high", "cam_right_wrist", "cam_left_wrist"]


def _build_features(state_dim, action_dim, cameras, img_hw, mode):
    """Return the LeRobot `features` dict describing every column we'll write."""
    feats = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"j{i}" for i in range(state_dim)],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"j{i}" for i in range(action_dim)],
        },
    }
    h, w = img_hw
    for cam in cameras:
        feats[f"observation.images.{cam}"] = {
            "dtype": mode,  # "video" or "image"
            "shape": (3, h, w),
            "names": ["channels", "height", "width"],
        }
    return feats


def _peek_shapes(hdf5_path, cameras):
    """Return (state_dim, action_dim, (H, W)) from the first episode."""
    with h5py.File(hdf5_path, "r") as f:
        state_dim = f["/observations/qpos"].shape[1]
        action_dim = f["/action"].shape[1]
        first_cam = cameras[0]
        h, w = f[f"/observations/images/{first_cam}"].shape[1:3]
    return state_dim, action_dim, (h, w)


def convert(args):
    # LeRobot imports live here so the module is importable even when lerobot
    # is absent (helpful for partial environments running only HDF5/Zarr cells).
    from lerobot.common.datasets.lerobot_dataset import (
        HF_LEROBOT_HOME,
        LeRobotDataset,
    )

    cameras = args.cameras.split(",") if args.cameras else DEFAULT_CAMERAS
    print(f"LeRobot conversion: mode={args.mode}, fps={args.fps}, cameras={cameras}")

    # Inspect the first episode to size features.
    first_path = os.path.join(args.dataset_dir, f"episode_0.hdf5")
    if not os.path.exists(first_path):
        raise FileNotFoundError(first_path)
    state_dim, action_dim, img_hw = _peek_shapes(first_path, cameras)
    features = _build_features(state_dim, action_dim, cameras, img_hw, args.mode)

    # Resolve output directory.
    root = Path(args.root) if args.root else HF_LEROBOT_HOME
    target = root / args.repo_id
    if target.exists():
        print(f"Overwriting existing dataset at {target}")
        shutil.rmtree(target)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
        features=features,
        root=root if args.root else None,
        use_videos=(args.mode == "video"),
        tolerance_s=args.tolerance_s,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        video_backend=args.video_backend,
    )

    total_hdf5_size = 0
    t0 = time.perf_counter()
    for ep_id in range(args.num_episodes):
        hdf5_path = os.path.join(args.dataset_dir, f"episode_{ep_id}.hdf5")
        if not os.path.exists(hdf5_path):
            print(f"Warning: {hdf5_path} not found, skipping")
            continue
        total_hdf5_size += os.path.getsize(hdf5_path)

        with h5py.File(hdf5_path, "r") as f:
            state = f["/observations/qpos"][()].astype(np.float32)
            action = f["/action"][()].astype(np.float32)
            imgs = {cam: f[f"/observations/images/{cam}"][()] for cam in cameras}
            T = state.shape[0]

        for t in range(T):
            frame = {
                "observation.state": state[t],
                "action": action[t],
            }
            for cam in cameras:
                # LeRobot expects HWC uint8 for both video and image modes.
                img = imgs[cam][t]
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                frame[f"observation.images.{cam}"] = img
            dataset.add_frame(frame, task=args.task)
        dataset.save_episode()
        print(f"  converted episode {ep_id} ({T} steps)")

    elapsed = time.perf_counter() - t0

    # Report disk usage.
    total_out = 0
    for dirpath, _, filenames in os.walk(target):
        for fname in filenames:
            total_out += os.path.getsize(os.path.join(dirpath, fname))

    print("\n--- LeRobot Conversion Summary ---")
    print(f"Mode:              {args.mode}")
    print(f"Episodes:          {args.num_episodes}")
    print(f"HDF5 total size:   {total_hdf5_size / 1e6:.1f} MB")
    print(f"LeRobot size:      {total_out / 1e6:.1f} MB")
    print(f"Compression ratio: {total_hdf5_size / max(total_out, 1):.2f}x")
    print(f"Conversion time:   {elapsed:.1f}s")
    print(f"Output:            {target}")


def main():
    p = argparse.ArgumentParser(description="Convert HDF5 episodes to LeRobot v2.1")
    p.add_argument("--dataset_dir", type=str, required=True,
                   help="Directory containing episode_*.hdf5 files")
    p.add_argument("--repo_id", type=str, required=True,
                   help="LeRobot repo_id, e.g. 'robotwin/act_bench'")
    p.add_argument("--num_episodes", type=int, required=True)
    p.add_argument("--root", type=str, default=None,
                   help="Root dir; defaults to $HF_LEROBOT_HOME")
    p.add_argument("--mode", type=str, default="video",
                   choices=["video", "image"],
                   help="video = MP4 via pyav (default), image = PNG frames")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--robot_type", type=str, default="aloha")
    p.add_argument("--task", type=str, default="robotwin_benchmark",
                   help="Task instruction string attached to every frame")
    p.add_argument("--cameras", type=str, default=None,
                   help="Comma-separated camera names (default: cam_high,cam_right_wrist,cam_left_wrist)")
    p.add_argument("--tolerance_s", type=float, default=1e-4)
    p.add_argument("--image_writer_processes", type=int, default=4)
    p.add_argument("--image_writer_threads", type=int, default=4)
    p.add_argument("--video_backend", type=str, default=None)
    args = p.parse_args()
    convert(args)


if __name__ == "__main__":
    main()

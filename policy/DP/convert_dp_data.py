"""
Convert Diffusion Policy's single-zarr dataset to HDF5 and LeRobot v2.1 so we
can benchmark all three formats on equal footing.

Source  : ./data/<task>-<config>-<num>.zarr  (produced by DP's process_data.py)
    data/head_camera  (N, 3, H, W) uint8   NCHW, zstd(clevel=3, shuffle=SHUFFLE)
    data/state        (N, state_dim) float32
    data/action       (N, action_dim) float32
    meta/episode_ends (E,) int64

Targets :
    --format hdf5     => <out>.hdf5 with the same top-level structure
    --format lerobot  => LeRobot v2.1 dataset with one episode per recording,
                         a single `head_camera` feature

Usage:
    python convert_dp_data.py \\
        --zarr_path ./data/block_hammer_beat-ActivePick-50.zarr \\
        --format hdf5 \\
        --out_path ./data/block_hammer_beat-ActivePick-50.hdf5

    python convert_dp_data.py \\
        --zarr_path ./data/block_hammer_beat-ActivePick-50.zarr \\
        --format lerobot --mode video --fps 25 \\
        --repo_id robotwin/dp_block_hammer --root ./processed_data_lerobot
"""

import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np
import zarr


def load_dp_zarr(zarr_path):
    """Return (head_camera_nchw, state, action, episode_ends)."""
    root = zarr.open(zarr_path, mode="r")
    head_camera = root["data/head_camera"][:]  # (N, 3, H, W) NCHW uint8
    state = root["data/state"][:].astype(np.float32)
    action = root["data/action"][:].astype(np.float32)
    episode_ends = root["meta/episode_ends"][:].astype(np.int64)
    return head_camera, state, action, episode_ends


# ---------------------------------------------------------------------------
# Target 1: HDF5 (merged layout, same structure as source zarr)
# ---------------------------------------------------------------------------

def to_hdf5(args, head_camera, state, action, episode_ends):
    import h5py

    out_path = args.out_path
    if os.path.exists(out_path):
        os.remove(out_path)

    # Chunk along time; keep per-image as one chunk for single-window reads.
    img_chunk = (1,) + head_camera.shape[1:]
    state_chunk = (min(100, state.shape[0]), state.shape[1])
    action_chunk = (min(100, action.shape[0]), action.shape[1])

    with h5py.File(out_path, "w") as f:
        data = f.create_group("data")
        data.create_dataset(
            "head_camera", data=head_camera, chunks=img_chunk,
            compression="gzip", compression_opts=4,
        )
        data.create_dataset(
            "state", data=state, chunks=state_chunk,
            compression="gzip", compression_opts=4,
        )
        data.create_dataset(
            "action", data=action, chunks=action_chunk,
            compression="gzip", compression_opts=4,
        )
        meta = f.create_group("meta")
        meta.create_dataset("episode_ends", data=episode_ends)

    return out_path


# ---------------------------------------------------------------------------
# Target 2: LeRobot v2.1
# ---------------------------------------------------------------------------

def to_lerobot(args, head_camera, state, action, episode_ends):
    from lerobot.common.datasets.lerobot_dataset import (
        HF_LEROBOT_HOME,
        LeRobotDataset,
    )

    # NCHW -> NHWC for LeRobot (it wants HWC uint8 when adding frames).
    head_camera_hwc = np.moveaxis(head_camera, 1, -1)
    h, w = head_camera_hwc.shape[1], head_camera_hwc.shape[2]

    state_dim = state.shape[1]
    action_dim = action.shape[1]

    features = {
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
        "observation.images.head_camera": {
            "dtype": args.mode,
            "shape": (3, h, w),
            "names": ["channels", "height", "width"],
        },
    }

    root = Path(args.root) if args.root else HF_LEROBOT_HOME
    target = root / args.repo_id
    if target.exists():
        print(f"Overwriting {target}")
        shutil.rmtree(target)

    ds = LeRobotDataset.create(
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

    start = 0
    for ep_idx, end in enumerate(episode_ends):
        for t in range(start, int(end)):
            frame = {
                "observation.state": state[t],
                "action": action[t],
                "observation.images.head_camera": head_camera_hwc[t],
            }
            ds.add_frame(frame, task=args.task)
        ds.save_episode()
        start = int(end)
        print(f"  saved episode {ep_idx} ({int(end) - (int(episode_ends[ep_idx-1]) if ep_idx > 0 else 0)} steps)")

    return str(target)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Convert DP zarr to HDF5 or LeRobot v2.1")
    p.add_argument("--zarr_path", type=str, required=True,
                   help="Path to DP's processed zarr (e.g. ./data/<task>-<cfg>-<num>.zarr)")
    p.add_argument("--format", type=str, required=True, choices=["hdf5", "lerobot"])

    # HDF5 args
    p.add_argument("--out_path", type=str, default=None,
                   help="Output path for --format hdf5 (default: <zarr_path>.hdf5)")

    # LeRobot args
    p.add_argument("--repo_id", type=str, default=None)
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--mode", type=str, default="video", choices=["video", "image"])
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--robot_type", type=str, default="aloha")
    p.add_argument("--task", type=str, default="robotwin_dp_benchmark")
    p.add_argument("--tolerance_s", type=float, default=1e-4)
    p.add_argument("--image_writer_processes", type=int, default=4)
    p.add_argument("--image_writer_threads", type=int, default=4)
    p.add_argument("--video_backend", type=str, default=None)
    args = p.parse_args()

    print(f"Loading DP zarr: {args.zarr_path}")
    t0 = time.perf_counter()
    head_camera, state, action, episode_ends = load_dp_zarr(args.zarr_path)
    print(f"  head_camera: {head_camera.shape} {head_camera.dtype}")
    print(f"  state:       {state.shape}")
    print(f"  action:      {action.shape}")
    print(f"  episodes:    {len(episode_ends)}")
    print(f"  load time:   {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    if args.format == "hdf5":
        out = args.out_path
        if out is None:
            out = args.zarr_path.rstrip("/").replace(".zarr", "") + ".hdf5"
            args.out_path = out
        out_path = to_hdf5(args, head_camera, state, action, episode_ends)
        size = os.path.getsize(out_path)
        print(f"\nWrote HDF5: {out_path} ({size / 1e6:.1f} MB)")
    else:  # lerobot
        if not args.repo_id:
            raise SystemExit("--repo_id is required for --format lerobot")
        out_dir = to_lerobot(args, head_camera, state, action, episode_ends)
        total = 0
        for dp, _, fn in os.walk(out_dir):
            for name in fn:
                total += os.path.getsize(os.path.join(dp, name))
        print(f"\nWrote LeRobot: {out_dir} ({total / 1e6:.1f} MB)")
    print(f"Conversion time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()

"""
Smoke test for the HDF5 vs Zarr benchmark pipeline.

Generates small synthetic episodes (no real ACT data required), runs the
full conversion + 2x2 benchmark, and prints a compact report. Use this to
validate the pipeline end-to-end before running on real data.

Usage:
    # One-shot: create fake data, convert to zarr, run benchmark
    python smoke_test.py

    # Customize
    python smoke_test.py --num_episodes 8 --episode_len 40 --img_h 120 --img_w 160

Requirements: h5py, zarr, numcodecs, torch, numpy
"""

import os
import sys
import argparse
import shutil
import subprocess

import numpy as np


def generate_fake_hdf5(out_dir, num_episodes, episode_len, img_h, img_w,
                       state_dim=14, enriched=False):
    """Create fake HDF5 episodes that match the ACT schema.

    When enriched=True, also generates depth (float32 HxW) and segmentation
    (uint8 HxWx3) arrays per camera, matching the demo_enriched.yml output.
    """
    import h5py
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    cams = ["cam_high", "cam_right_wrist", "cam_left_wrist"]
    for i in range(num_episodes):
        T = episode_len + rng.integers(-5, 5)
        T = max(10, T)
        path = os.path.join(out_dir, f"episode_{i}.hdf5")
        with h5py.File(path, "w") as f:
            f.create_dataset("action", data=rng.standard_normal((T, state_dim)).astype(np.float32))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=rng.standard_normal((T, state_dim)).astype(np.float32))
            images = obs.create_group("images")
            for cam in cams:
                img = rng.integers(0, 255, size=(T, img_h, img_w, 3), dtype=np.uint8)
                images.create_dataset(cam, data=img, dtype=np.uint8)
            if enriched:
                depths = obs.create_group("depths")
                seg = obs.create_group("segmentation")
                for cam in cams:
                    depths.create_dataset(cam, data=rng.random((T, img_h, img_w)).astype(np.float32) * 5000)
                    seg.create_dataset(cam, data=rng.integers(0, 255, size=(T, img_h, img_w, 3), dtype=np.uint8))
        tag = " [enriched]" if enriched else ""
        print(f"  wrote {path}  T={T}  img=({img_h}x{img_w}){tag}")


def main():
    parser = argparse.ArgumentParser(description="Smoke test the benchmark pipeline")
    parser.add_argument("--work_dir", type=str, default="/tmp/act_benchmark_smoke")
    parser.add_argument("--num_episodes", type=int, default=6)
    parser.add_argument("--episode_len", type=int, default=30)
    parser.add_argument("--img_h", type=int, default=120)
    parser.add_argument("--img_w", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_iters", type=int, default=20)
    parser.add_argument("--num_warmup", type=int, default=3)
    parser.add_argument("--compressor", type=str, default="dp",
                        choices=["dp", "dp-memory", "none", "lz4", "zstd"])
    parser.add_argument("--keep", action="store_true", help="Keep temp files after run")
    parser.add_argument("--enriched", action="store_true",
                        help="Generate enriched data (depth + segmentation) for storage comparison")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip the ACT training step (no CUDA needed)")
    args = parser.parse_args()

    # Check deps
    missing = []
    for mod in ["h5py", "zarr", "numcodecs", "torch"]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"ERROR: missing dependencies: {missing}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)

    # LeRobot is optional for the smoke test; we probe for it.
    lerobot_available = True
    try:
        import lerobot  # noqa: F401
    except Exception as exc:
        lerobot_available = False
        print(f"[info] lerobot not installed ({exc!s}); LeRobot cells will be skipped.")

    hdf5_dir = os.path.join(args.work_dir, "hdf5")
    zarr_dir = os.path.join(args.work_dir, "zarr")
    lerobot_root = os.path.join(args.work_dir, "lerobot")
    lerobot_repo_id = "robotwin/smoke_test"

    if os.path.exists(args.work_dir):
        shutil.rmtree(args.work_dir)
    os.makedirs(args.work_dir, exist_ok=True)

    print("=" * 70)
    print("  SMOKE TEST  —  HDF5 vs Zarr benchmark pipeline")
    print("=" * 70)

    # -------- Step 1: generate fake HDF5 --------
    enr_tag = " [enriched]" if args.enriched else ""
    print(f"\n[1/3] Generating {args.num_episodes} fake HDF5 episodes in {hdf5_dir}{enr_tag}")
    generate_fake_hdf5(hdf5_dir, args.num_episodes, args.episode_len, args.img_h, args.img_w,
                       enriched=args.enriched)

    # -------- Step 2: convert to Zarr --------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    convert_script = os.path.join(script_dir, "convert_hdf5_to_zarr.py")
    print(f"\n[2/3] Converting HDF5 -> Zarr (compressor={args.compressor})")
    cmd = [
        sys.executable, convert_script,
        "--dataset_dir", hdf5_dir,
        "--output_dir", zarr_dir,
        "--num_episodes", str(args.num_episodes),
        "--compressor", args.compressor,
        "--layout", "per_episode",
    ]
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        print("Conversion failed.")
        sys.exit(ret.returncode)

    # -------- Step 2b: convert to LeRobot v2.1 (optional) --------
    if lerobot_available:
        convert_lerobot = os.path.join(script_dir, "convert_hdf5_to_lerobot.py")
        print(f"\n[2b/3] Converting HDF5 -> LeRobot v2.1 (mode=image for smoke speed)")
        cmd = [
            sys.executable, convert_lerobot,
            "--dataset_dir", hdf5_dir,
            "--repo_id", lerobot_repo_id,
            "--num_episodes", str(args.num_episodes),
            "--mode", "image",         # avoid pyav dep on CI
            "--fps", "25",
            "--root", lerobot_root,
        ]
        ret = subprocess.run(cmd, check=False)
        if ret.returncode != 0:
            print("LeRobot conversion failed; continuing without LeRobot cells.")
            lerobot_available = False

    # -------- Step 3: run benchmark --------
    bench_script = os.path.join(script_dir, "benchmark_data_formats.py")
    print(f"\n[3/3] Running 3x2 benchmark")
    cmd = [
        sys.executable, bench_script,
        "--hdf5_dir", hdf5_dir,
        "--zarr_dir", zarr_dir,
        "--num_episodes", str(args.num_episodes),
        "--batch_size", str(args.batch_size),
        "--num_warmup", str(args.num_warmup),
        "--num_iters", str(args.num_iters),
        "--output", os.path.join(args.work_dir, "results.json"),
    ]
    if lerobot_available:
        cmd += [
            "--lerobot_repo_id", lerobot_repo_id,
            "--lerobot_root", lerobot_root,
        ]
    if args.skip_training:
        cmd.append("--skip_training")
    ret = subprocess.run(cmd, check=False)

    # -------- Cleanup --------
    if not args.keep:
        print(f"\nCleaning up {args.work_dir} (use --keep to retain)")
        shutil.rmtree(args.work_dir)

    if ret.returncode == 0:
        print("\n\u2713 smoke test passed")
    else:
        print("\n\u2717 smoke test failed")
        sys.exit(ret.returncode)


if __name__ == "__main__":
    main()

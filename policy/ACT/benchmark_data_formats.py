"""
Benchmark HDF5 vs Zarr data loading for the ACT policy.

Measures:
  1. Raw I/O latency  — single-item read speed (no DataLoader overhead)
  2. DataLoader throughput — batches/sec across different num_workers & prefetch configs
  3. GPU transfer time — data-to-CUDA overhead
  4. End-to-end training step — full forward+backward with real ACT model
  5. Memory footprint — peak RSS during data loading

Usage:
    python benchmark_data_formats.py \
        --hdf5_dir <path_to_hdf5_episodes> \
        --zarr_dir <path_to_zarr_episodes> \
        --num_episodes 50 \
        --batch_size 8 \
        --num_warmup 5 \
        --num_iters 50

Output: a structured report printed to stdout and saved as benchmark_results.json.
"""

import os
import sys
import time
import json
import argparse
import resource
from contextlib import contextmanager
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data
import h5py

try:
    import zarr
except ImportError:
    zarr = None
    print("Warning: zarr not installed. Zarr benchmarks will be skipped.")

# ---------------------------------------------------------------------------
# Dataset implementations
# ---------------------------------------------------------------------------

class HDF5EpisodicDataset(torch.utils.data.Dataset):
    """Episode dataset reading from HDF5 files (mirrors original utils.py)."""

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, max_action_len):
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.max_action_len = max_action_len

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            episode_len = root["/action"].shape[0]
            start_ts = np.random.randint(0, episode_len)
            qpos = root["/observations/qpos"][start_ts]
            image_dict = {}
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][start_ts]
            action = root["/action"][start_ts:]
            action_len = episode_len - start_ts

        padded_action = np.zeros((self.max_action_len, action.shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.ones(self.max_action_len, dtype=bool)
        is_pad[:action_len] = False

        all_cam_images = np.stack([image_dict[c] for c in self.camera_names], axis=0)
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        image_data = torch.einsum("k h w c -> k c h w", image_data)
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


class ZarrEpisodicDataset(torch.utils.data.Dataset):
    """Episode dataset reading from Zarr stores."""

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, max_action_len):
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.max_action_len = max_action_len

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        zarr_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.zarr")
        root = zarr.open(zarr_path, mode="r")

        episode_len = root["action"].shape[0]
        start_ts = np.random.randint(0, episode_len)
        qpos = root["observations/qpos"][start_ts]
        image_dict = {}
        for cam_name in self.camera_names:
            image_dict[cam_name] = root[f"observations/images/{cam_name}"][start_ts]
        action = root["action"][start_ts:]
        action_len = episode_len - start_ts

        padded_action = np.zeros((self.max_action_len, action.shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.ones(self.max_action_len, dtype=bool)
        is_pad[:action_len] = False

        all_cam_images = np.stack([image_dict[c] for c in self.camera_names], axis=0)
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        image_data = torch.einsum("k h w c -> k c h w", image_data)
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


# ---------------------------------------------------------------------------
# Normalization stats (shared across both formats)
# ---------------------------------------------------------------------------

def get_norm_stats_from_hdf5(dataset_dir, num_episodes):
    """Compute normalization stats from HDF5 episodes."""
    all_qpos, all_action = [], []
    for i in range(num_episodes):
        path = os.path.join(dataset_dir, f"episode_{i}.hdf5")
        with h5py.File(path, "r") as f:
            all_qpos.append(torch.from_numpy(f["/observations/qpos"][()]))
            all_action.append(torch.from_numpy(f["/action"][()]))

    max_action_len = max(a.size(0) for a in all_action)

    # Pad and stack
    padded_qpos = []
    max_qpos_len = max(q.size(0) for q in all_qpos)
    for q in all_qpos:
        if q.size(0) < max_qpos_len:
            q = torch.cat([q, q[-1:].repeat(max_qpos_len - q.size(0), 1)])
        padded_qpos.append(q)
    padded_action = []
    for a in all_action:
        if a.size(0) < max_action_len:
            a = torch.cat([a, a[-1:].repeat(max_action_len - a.size(0), 1)])
        padded_action.append(a)

    all_qpos_t = torch.stack(padded_qpos)
    all_action_t = torch.stack(padded_action)

    stats = {
        "action_mean": all_action_t.mean(dim=[0, 1]).numpy(),
        "action_std": all_action_t.std(dim=[0, 1]).clamp(min=1e-2).numpy(),
        "qpos_mean": all_qpos_t.mean(dim=[0, 1]).numpy(),
        "qpos_std": all_qpos_t.std(dim=[0, 1]).clamp(min=1e-2).numpy(),
    }
    return stats, max_action_len


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

@contextmanager
def timer(name, results_dict):
    """Context manager that records elapsed time into results_dict[name]."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    yield
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t0
    results_dict[name].append(elapsed)


def get_peak_rss_mb():
    """Peak resident set size in MB (Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ---------------------------------------------------------------------------
# Benchmark 1: Raw I/O latency
# ---------------------------------------------------------------------------

def benchmark_raw_io(dataset, num_warmup, num_iters, label):
    """Measure single-item __getitem__ latency."""
    results = defaultdict(list)
    indices = np.random.randint(0, len(dataset), size=num_warmup + num_iters)

    # Warmup
    for i in range(num_warmup):
        _ = dataset[indices[i]]

    # Timed iterations
    for i in range(num_warmup, num_warmup + num_iters):
        with timer("item_latency", results):
            _ = dataset[indices[i]]

    latencies = np.array(results["item_latency"]) * 1000  # ms
    return {
        f"{label}/raw_io/mean_ms": float(np.mean(latencies)),
        f"{label}/raw_io/median_ms": float(np.median(latencies)),
        f"{label}/raw_io/p95_ms": float(np.percentile(latencies, 95)),
        f"{label}/raw_io/p99_ms": float(np.percentile(latencies, 99)),
        f"{label}/raw_io/std_ms": float(np.std(latencies)),
        f"{label}/raw_io/min_ms": float(np.min(latencies)),
        f"{label}/raw_io/max_ms": float(np.max(latencies)),
    }


# ---------------------------------------------------------------------------
# Benchmark 2: DataLoader throughput
# ---------------------------------------------------------------------------

def benchmark_dataloader(dataset, batch_size, num_workers, prefetch_factor,
                         num_warmup, num_iters, label):
    """Measure DataLoader throughput (batches/sec, samples/sec)."""
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    batch_times = []
    it = iter(loader)
    # Warmup
    for _ in range(num_warmup):
        try:
            _ = next(it)
        except StopIteration:
            it = iter(loader)
            _ = next(it)

    # Timed
    for _ in range(num_iters):
        t0 = time.perf_counter()
        try:
            _ = next(it)
        except StopIteration:
            it = iter(loader)
            _ = next(it)
        batch_times.append(time.perf_counter() - t0)

    batch_times = np.array(batch_times)
    samples_per_sec = batch_size / batch_times

    config_tag = f"w{num_workers}_pf{prefetch_factor}"
    return {
        f"{label}/dataloader/{config_tag}/batch_time_mean_ms": float(np.mean(batch_times) * 1000),
        f"{label}/dataloader/{config_tag}/batch_time_median_ms": float(np.median(batch_times) * 1000),
        f"{label}/dataloader/{config_tag}/batch_time_p95_ms": float(np.percentile(batch_times, 95) * 1000),
        f"{label}/dataloader/{config_tag}/samples_per_sec_mean": float(np.mean(samples_per_sec)),
        f"{label}/dataloader/{config_tag}/samples_per_sec_median": float(np.median(samples_per_sec)),
    }


# ---------------------------------------------------------------------------
# Benchmark 3: GPU transfer time
# ---------------------------------------------------------------------------

def benchmark_gpu_transfer(dataset, batch_size, num_warmup, num_iters, label):
    """Measure time to move a batch from CPU to CUDA."""
    if not torch.cuda.is_available():
        return {f"{label}/gpu_transfer/skipped": "no CUDA"}

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2,
        prefetch_factor=2,
    )
    it = iter(loader)
    transfer_times = []

    for i in range(num_warmup + num_iters):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        if i < num_warmup:
            continue
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = [t.cuda(non_blocking=True) for t in batch]
        torch.cuda.synchronize()
        transfer_times.append(time.perf_counter() - t0)

    transfer_times = np.array(transfer_times) * 1000
    return {
        f"{label}/gpu_transfer/mean_ms": float(np.mean(transfer_times)),
        f"{label}/gpu_transfer/median_ms": float(np.median(transfer_times)),
        f"{label}/gpu_transfer/p95_ms": float(np.percentile(transfer_times, 95)),
    }


# ---------------------------------------------------------------------------
# Benchmark 4: End-to-end training step
# ---------------------------------------------------------------------------

def benchmark_training_step(dataset, batch_size, num_warmup, num_iters, label, camera_names):
    """Measure full forward+backward time using a real ACT model."""
    if not torch.cuda.is_available():
        return {f"{label}/train_step/skipped": "no CUDA"}

    # Build a real ACT model
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from detr.main import build_ACT_model_and_optimizer
    except ImportError as exc:
        return {f"{label}/train_step/skipped": str(exc)}

    policy_config = {
        "lr": 1e-5,
        "num_queries": 50,
        "kl_weight": 10,
        "hidden_dim": 512,
        "dim_feedforward": 3200,
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": camera_names,
    }
    try:
        from act_policy import ACTPolicy
        policy = ACTPolicy(policy_config)
    except Exception as exc:
        return {f"{label}/train_step/skipped": str(exc)}

    policy.cuda()
    policy.train()
    optimizer = policy.configure_optimizers()

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
        num_workers=2, prefetch_factor=2,
    )
    it = iter(loader)
    step_times = []
    data_times = []
    compute_times = []

    for i in range(num_warmup + num_iters):
        # --- data loading time ---
        torch.cuda.synchronize()
        t_data_start = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        image_data, qpos_data, action_data, is_pad = [t.cuda() for t in batch]
        torch.cuda.synchronize()
        t_data_end = time.perf_counter()

        # --- compute time (forward + backward) ---
        t_compute_start = time.perf_counter()
        forward_dict = policy(qpos_data, image_data, action_data, is_pad)
        loss = forward_dict["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t_compute_end = time.perf_counter()

        if i >= num_warmup:
            data_times.append(t_data_end - t_data_start)
            compute_times.append(t_compute_end - t_compute_start)
            step_times.append(t_compute_end - t_data_start)

    data_times = np.array(data_times) * 1000
    compute_times = np.array(compute_times) * 1000
    step_times = np.array(step_times) * 1000

    return {
        f"{label}/train_step/total_mean_ms": float(np.mean(step_times)),
        f"{label}/train_step/total_median_ms": float(np.median(step_times)),
        f"{label}/train_step/data_mean_ms": float(np.mean(data_times)),
        f"{label}/train_step/data_median_ms": float(np.median(data_times)),
        f"{label}/train_step/compute_mean_ms": float(np.mean(compute_times)),
        f"{label}/train_step/compute_median_ms": float(np.median(compute_times)),
        f"{label}/train_step/data_pct": float(np.mean(data_times) / np.mean(step_times) * 100),
    }


# ---------------------------------------------------------------------------
# Main benchmark orchestrator
# ---------------------------------------------------------------------------

def run_benchmarks(args):
    print("=" * 70)
    print("  HDF5 vs Zarr Data Format Benchmark for ACT Policy")
    print("=" * 70)

    camera_names = ["cam_high", "cam_right_wrist", "cam_left_wrist"]

    # 1. Compute shared normalization stats
    print("\n[1/5] Computing normalization statistics from HDF5 ...")
    norm_stats, max_action_len = get_norm_stats_from_hdf5(args.hdf5_dir, args.num_episodes)
    print(f"  max_action_len = {max_action_len}")

    # 2. Build datasets
    episode_ids = list(range(args.num_episodes))
    hdf5_ds = HDF5EpisodicDataset(episode_ids, args.hdf5_dir, camera_names, norm_stats, max_action_len)

    zarr_ds = None
    if zarr is not None and args.zarr_dir and os.path.isdir(args.zarr_dir):
        zarr_ds = ZarrEpisodicDataset(episode_ids, args.zarr_dir, camera_names, norm_stats, max_action_len)
    elif zarr is None:
        print("  Zarr not available, skipping Zarr benchmarks.")
    else:
        print(f"  Zarr directory not found: {args.zarr_dir}")

    all_results = {}

    # 3. Raw I/O latency
    print(f"\n[2/5] Raw I/O latency (warmup={args.num_warmup}, iters={args.num_iters}) ...")
    all_results.update(benchmark_raw_io(hdf5_ds, args.num_warmup, args.num_iters, "hdf5"))
    if zarr_ds:
        all_results.update(benchmark_raw_io(zarr_ds, args.num_warmup, args.num_iters, "zarr"))

    # 4. DataLoader throughput — sweep num_workers and prefetch_factor
    print(f"\n[3/5] DataLoader throughput (batch_size={args.batch_size}) ...")
    worker_configs = [
        (0, 2),    # main process, prefetch_factor ignored
        (1, 1),    # original ACT config
        (2, 2),
        (4, 2),
        (4, 4),
        (8, 2),
    ]
    for nw, pf in worker_configs:
        tag = f"  workers={nw}, prefetch={pf}"
        print(tag)
        all_results.update(benchmark_dataloader(
            hdf5_ds, args.batch_size, nw, pf, args.num_warmup, args.num_iters, "hdf5"))
        if zarr_ds:
            all_results.update(benchmark_dataloader(
                zarr_ds, args.batch_size, nw, pf, args.num_warmup, args.num_iters, "zarr"))

    # 5. GPU transfer
    print(f"\n[4/5] GPU transfer time ...")
    all_results.update(benchmark_gpu_transfer(hdf5_ds, args.batch_size, args.num_warmup, args.num_iters, "hdf5"))
    if zarr_ds:
        all_results.update(benchmark_gpu_transfer(zarr_ds, args.batch_size, args.num_warmup, args.num_iters, "zarr"))

    # 6. End-to-end training step
    print(f"\n[5/5] End-to-end training step ...")
    all_results.update(benchmark_training_step(
        hdf5_ds, args.batch_size, args.num_warmup, min(args.num_iters, 20), "hdf5", camera_names))
    if zarr_ds:
        all_results.update(benchmark_training_step(
            zarr_ds, args.batch_size, args.num_warmup, min(args.num_iters, 20), "zarr", camera_names))

    # Peak memory
    all_results["peak_rss_mb"] = get_peak_rss_mb()

    return all_results


def print_report(results):
    """Pretty-print the benchmark results as a comparison table."""
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)

    # --- Raw I/O ---
    print("\n--- Raw I/O Latency (ms) ---")
    print(f"{'Metric':<30} {'HDF5':>12} {'Zarr':>12} {'Speedup':>10}")
    print("-" * 66)
    for suffix in ["mean_ms", "median_ms", "p95_ms", "p99_ms"]:
        h = results.get(f"hdf5/raw_io/{suffix}", None)
        z = results.get(f"zarr/raw_io/{suffix}", None)
        speedup = f"{h/z:.2f}x" if (h and z) else "N/A"
        print(f"  {suffix:<28} {h or 'N/A':>12.2f}" +
              (f" {z:>12.2f} {speedup:>10}" if z else ""))

    # --- DataLoader ---
    print("\n--- DataLoader Throughput (samples/sec) ---")
    print(f"{'Config':<25} {'HDF5':>12} {'Zarr':>12} {'Speedup':>10}")
    print("-" * 61)
    configs = set()
    for k in results:
        if "dataloader" in k and "samples_per_sec_mean" in k:
            parts = k.split("/")
            configs.add(parts[2])
    for cfg in sorted(configs):
        h = results.get(f"hdf5/dataloader/{cfg}/samples_per_sec_mean")
        z = results.get(f"zarr/dataloader/{cfg}/samples_per_sec_mean")
        speedup = f"{z/h:.2f}x" if (h and z) else "N/A"
        print(f"  {cfg:<23} {h or 'N/A':>12.1f}" +
              (f" {z:>12.1f} {speedup:>10}" if z else ""))

    # --- GPU Transfer ---
    print("\n--- GPU Transfer (ms) ---")
    for suffix in ["mean_ms", "median_ms"]:
        h = results.get(f"hdf5/gpu_transfer/{suffix}")
        z = results.get(f"zarr/gpu_transfer/{suffix}")
        if h:
            print(f"  HDF5 {suffix}: {h:.2f}")
        if z:
            print(f"  Zarr {suffix}: {z:.2f}")

    # --- Training Step ---
    print("\n--- End-to-End Training Step (ms) ---")
    print(f"{'Metric':<30} {'HDF5':>12} {'Zarr':>12}")
    print("-" * 56)
    for suffix in ["total_mean_ms", "data_mean_ms", "compute_mean_ms", "data_pct"]:
        h = results.get(f"hdf5/train_step/{suffix}")
        z = results.get(f"zarr/train_step/{suffix}")
        fmt = ".1f" if "pct" not in suffix else ".1f"
        unit = "%" if "pct" in suffix else ""
        h_str = f"{h:{fmt}}{unit}" if h else "N/A"
        z_str = f"{z:{fmt}}{unit}" if z else "N/A"
        print(f"  {suffix:<28} {h_str:>12} {z_str:>12}")

    print(f"\n  Peak RSS: {results.get('peak_rss_mb', 'N/A'):.0f} MB")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Benchmark HDF5 vs Zarr for ACT policy")
    parser.add_argument("--hdf5_dir", type=str, required=True,
                        help="Directory containing episode_*.hdf5 files")
    parser.add_argument("--zarr_dir", type=str, default=None,
                        help="Directory containing episode_*.zarr stores")
    parser.add_argument("--num_episodes", type=int, required=True,
                        help="Number of episodes to use")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_warmup", type=int, default=5,
                        help="Warmup iterations (not timed)")
    parser.add_argument("--num_iters", type=int, default=50,
                        help="Timed iterations per benchmark")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Path to save JSON results")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip the end-to-end training benchmark (faster)")
    args = parser.parse_args()

    results = run_benchmarks(args)
    print_report(results)

    # Save to JSON
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

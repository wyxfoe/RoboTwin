"""
Benchmark data loading for the Diffusion Policy (DP) across a 3x2 matrix of
(file format) x (loading strategy).

                          HDF5                 Zarr                LeRobot v2.1
    Lazy (on-disk)    HDF5-lazy           Zarr-lazy           LeRobot-lazy
    Eager (preload)   HDF5-eager          Zarr-eager          LeRobot-eager

DP's native pipeline uses Zarr + eager (ReplayBuffer.copy_from_path decompresses
every array into RAM at __init__, then samples horizon-length windows from
numpy). This benchmark exposes the other five cells on exactly the same data.

Each __getitem__ returns a horizon-length window (default horizon=8) with:
    head_camera  (T, 3, H, W) uint8  — NCHW, DP's native layout
    state        (T, state_dim) float32
    action       (T, action_dim) float32

Usage:
    python benchmark_data_formats.py \\
        --zarr_path ./data/<task>.zarr \\
        --hdf5_path ./data/<task>.hdf5 \\
        --lerobot_repo_id robotwin/<task> --lerobot_root ./processed_data_lerobot \\
        --horizon 8 --batch_size 128 --num_warmup 5 --num_iters 50

Any source path can be omitted; the corresponding cells are skipped.
Source file creation:
    python convert_dp_data.py --zarr_path <zarr> --format hdf5    --out_path <hdf5>
    python convert_dp_data.py --zarr_path <zarr> --format lerobot --repo_id <id> ...
"""

import argparse
import json
import os
import resource
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch
import torch.utils.data

try:
    import h5py
except ImportError:
    h5py = None

try:
    import zarr
except ImportError:
    zarr = None

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    _LEROBOT_IMPORT_ERR = None
except Exception as _exc:
    LeRobotDataset = None
    _LEROBOT_IMPORT_ERR = _exc


# ---------------------------------------------------------------------------
# Index construction from episode_ends
# ---------------------------------------------------------------------------

def build_window_indices(episode_ends, horizon):
    """Return array of valid start indices that yield a horizon-length window
    without crossing episode boundaries."""
    starts = []
    prev = 0
    for end in episode_ends:
        end = int(end)
        # valid start s means s + horizon <= end
        last_valid = end - horizon
        for s in range(prev, last_valid + 1):
            starts.append(s)
        prev = end
    return np.array(starts, dtype=np.int64)


# ---------------------------------------------------------------------------
# Lazy datasets — index into the on-disk store each __getitem__
# ---------------------------------------------------------------------------

class _DPZarrLazyDataset(torch.utils.data.Dataset):
    def __init__(self, zarr_path, horizon, window_starts):
        self.zarr_path = zarr_path
        self.horizon = horizon
        self.starts = window_starts
        # Open once per dataset. Worker processes will re-open this after fork.
        self._root = None

    def _ensure_open(self):
        if self._root is None:
            self._root = zarr.open(self.zarr_path, mode="r")
        return self._root

    def __len__(self):
        return len(self.starts)

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_root"] = None
        return s

    def __getitem__(self, idx):
        root = self._ensure_open()
        s = int(self.starts[idx])
        e = s + self.horizon
        head = root["data/head_camera"][s:e]
        state = root["data/state"][s:e]
        action = root["data/action"][s:e]
        return (
            torch.from_numpy(np.asarray(head)),
            torch.from_numpy(np.asarray(state, dtype=np.float32)),
            torch.from_numpy(np.asarray(action, dtype=np.float32)),
        )


class _DPHDF5LazyDataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_path, horizon, window_starts):
        self.hdf5_path = hdf5_path
        self.horizon = horizon
        self.starts = window_starts
        self._f = None

    def _ensure_open(self):
        if self._f is None:
            self._f = h5py.File(self.hdf5_path, "r", swmr=True)
        return self._f

    def __len__(self):
        return len(self.starts)

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_f"] = None
        return s

    def __getitem__(self, idx):
        f = self._ensure_open()
        s = int(self.starts[idx])
        e = s + self.horizon
        head = f["data/head_camera"][s:e]
        state = f["data/state"][s:e]
        action = f["data/action"][s:e]
        return (
            torch.from_numpy(head),
            torch.from_numpy(state.astype(np.float32, copy=False)),
            torch.from_numpy(action.astype(np.float32, copy=False)),
        )


# ---------------------------------------------------------------------------
# Eager datasets — preload everything into numpy at __init__
# ---------------------------------------------------------------------------

class _DPEagerDataset(torch.utils.data.Dataset):
    def __init__(self, head_camera, state, action, horizon, window_starts):
        self.head_camera = head_camera  # (N, 3, H, W) uint8
        self.state = state              # (N, S) float32
        self.action = action            # (N, A) float32
        self.horizon = horizon
        self.starts = window_starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = int(self.starts[idx])
        e = s + self.horizon
        return (
            torch.from_numpy(self.head_camera[s:e].copy()),
            torch.from_numpy(self.state[s:e].copy()),
            torch.from_numpy(self.action[s:e].copy()),
        )


def _build_eager_from_zarr(zarr_path):
    root = zarr.open(zarr_path, mode="r")
    return (
        root["data/head_camera"][:],
        root["data/state"][:].astype(np.float32, copy=False),
        root["data/action"][:].astype(np.float32, copy=False),
        root["meta/episode_ends"][:].astype(np.int64, copy=False),
    )


def _build_eager_from_hdf5(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        head = f["data/head_camera"][()]
        state = f["data/state"][()].astype(np.float32, copy=False)
        action = f["data/action"][()].astype(np.float32, copy=False)
        ep_ends = f["meta/episode_ends"][()].astype(np.int64, copy=False)
    return head, state, action, ep_ends


# ---------------------------------------------------------------------------
# LeRobot datasets
# ---------------------------------------------------------------------------

def _make_lerobot_dataset(repo_id, root, delta_timestamps=None):
    if LeRobotDataset is None:
        raise RuntimeError(
            f"lerobot not installed: {_LEROBOT_IMPORT_ERR}"
        )
    kwargs = {"repo_id": repo_id}
    if root is not None:
        kwargs["root"] = root
    if delta_timestamps is not None:
        kwargs["delta_timestamps"] = delta_timestamps
    try:
        return LeRobotDataset(local_files_only=True, **kwargs)
    except TypeError:
        return LeRobotDataset(**kwargs)


def _lerobot_img_to_nchw_uint8(img_tensor):
    """CHW float[0,1] or HWC uint8 -> CHW uint8."""
    if img_tensor.dtype == torch.uint8:
        if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
            img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor.numpy()
    arr = img_tensor.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = np.transpose(arr, (2, 0, 1))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


class _DPLeRobotLazyDataset(torch.utils.data.Dataset):
    """LeRobot v2.1 + lazy: use delta_timestamps to fetch a horizon window."""

    def __init__(self, repo_id, root, horizon, fps):
        # Request a horizon-long window relative to the current frame.
        dt = {
            "observation.state":
                [i / fps for i in range(horizon)],
            "action":
                [i / fps for i in range(horizon)],
            "observation.images.head_camera":
                [i / fps for i in range(horizon)],
        }
        self.ds = _make_lerobot_dataset(repo_id, root, delta_timestamps=dt)
        self.horizon = horizon
        # Only indices where [i, i+horizon-1] stay inside the same episode.
        self._valid_indices = self._compute_valid_indices()

    def _compute_valid_indices(self):
        bounds = self.ds.episode_data_index
        froms = bounds["from"].tolist() if hasattr(bounds["from"], "tolist") else list(bounds["from"])
        tos = bounds["to"].tolist() if hasattr(bounds["to"], "tolist") else list(bounds["to"])
        out = []
        for a, b in zip(froms, tos):
            last = b - self.horizon
            if last >= a:
                out.extend(range(a, last + 1))
        return np.array(out, dtype=np.int64)

    def __len__(self):
        return len(self._valid_indices)

    def __getitem__(self, idx):
        sample = self.ds[int(self._valid_indices[idx])]
        # delta_timestamps stacks frames along a new leading dim: (T, ...)
        imgs = sample["observation.images.head_camera"]   # (T, C, H, W) float[0,1]
        state = sample["observation.state"]               # (T, S)
        action = sample["action"]                         # (T, A)

        if imgs.dtype != torch.uint8:
            imgs = (imgs.clamp(0, 1) * 255.0).to(torch.uint8)
        # Ensure CHW layout (lerobot returns CHW when decoding video).
        if imgs.ndim == 4 and imgs.shape[1] != 3 and imgs.shape[-1] == 3:
            imgs = imgs.permute(0, 3, 1, 2).contiguous()
        return (imgs, state.float(), action.float())


class _DPLeRobotEagerDataset(torch.utils.data.Dataset):
    """LeRobot v2.1 + eager: decode every frame once into NCHW uint8."""

    def __init__(self, repo_id, root, horizon, window_starts_or_none=None):
        ds = _make_lerobot_dataset(repo_id, root)
        N = len(ds)
        # Peek first sample to get H,W.
        first = ds[0]
        first_img = _lerobot_img_to_nchw_uint8(first["observation.images.head_camera"])
        _, h, w = first_img.shape
        state_dim = first["observation.state"].shape[-1]
        action_dim = first["action"].shape[-1]

        head = np.zeros((N, 3, h, w), dtype=np.uint8)
        state = np.zeros((N, state_dim), dtype=np.float32)
        action = np.zeros((N, action_dim), dtype=np.float32)

        head[0] = first_img
        state[0] = np.asarray(first["observation.state"], dtype=np.float32)
        action[0] = np.asarray(first["action"], dtype=np.float32)

        for i in range(1, N):
            s = ds[i]
            head[i] = _lerobot_img_to_nchw_uint8(s["observation.images.head_camera"])
            state[i] = np.asarray(s["observation.state"], dtype=np.float32)
            action[i] = np.asarray(s["action"], dtype=np.float32)

        # Build episode_ends from LeRobot's episode index.
        bounds = ds.episode_data_index
        tos = bounds["to"].tolist() if hasattr(bounds["to"], "tolist") else list(bounds["to"])
        episode_ends = np.array(tos, dtype=np.int64)

        self.window_starts = (
            build_window_indices(episode_ends, horizon)
            if window_starts_or_none is None else window_starts_or_none
        )
        self.horizon = horizon
        self.head = head
        self.state = state
        self.action = action

    def __len__(self):
        return len(self.window_starts)

    def __getitem__(self, idx):
        s = int(self.window_starts[idx])
        e = s + self.horizon
        return (
            torch.from_numpy(self.head[s:e].copy()),
            torch.from_numpy(self.state[s:e].copy()),
            torch.from_numpy(self.action[s:e].copy()),
        )


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

@contextmanager
def timer(name, results_dict):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    results_dict[name].append(time.perf_counter() - t0)


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_raw_io(ds, num_warmup, num_iters, label):
    results = defaultdict(list)
    rng = np.random.default_rng(0)
    idxs = rng.integers(0, len(ds), size=num_warmup + num_iters)
    for i in range(num_warmup):
        _ = ds[int(idxs[i])]
    for i in range(num_warmup, num_warmup + num_iters):
        with timer("t", results):
            _ = ds[int(idxs[i])]
    arr = np.array(results["t"]) * 1000
    return {
        f"{label}/raw_io/mean_ms": float(arr.mean()),
        f"{label}/raw_io/median_ms": float(np.median(arr)),
        f"{label}/raw_io/p95_ms": float(np.percentile(arr, 95)),
        f"{label}/raw_io/p99_ms": float(np.percentile(arr, 99)),
    }


def bench_dataloader(ds, batch_size, num_workers, prefetch, num_warmup, num_iters, label):
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=prefetch if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    it = iter(loader)
    for _ in range(num_warmup):
        try:
            next(it)
        except StopIteration:
            it = iter(loader); next(it)
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        try:
            next(it)
        except StopIteration:
            it = iter(loader); next(it)
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    cfg = f"w{num_workers}_pf{prefetch}"
    return {
        f"{label}/dataloader/{cfg}/batch_ms_mean": float(times.mean() * 1000),
        f"{label}/dataloader/{cfg}/batch_ms_median": float(np.median(times) * 1000),
        f"{label}/dataloader/{cfg}/samples_per_sec_mean": float(batch_size / times.mean()),
    }


def bench_gpu_transfer(ds, batch_size, num_warmup, num_iters, label):
    if not torch.cuda.is_available():
        return {f"{label}/gpu_transfer/skipped": "no CUDA"}
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, pin_memory=True,
        num_workers=2, prefetch_factor=2,
    )
    it = iter(loader)
    for _ in range(num_warmup):
        try: next(it)
        except StopIteration: it = iter(loader); next(it)
    times = []
    for _ in range(num_iters):
        try: batch = next(it)
        except StopIteration: it = iter(loader); batch = next(it)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = [t.cuda(non_blocking=True) for t in batch]
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000
    return {
        f"{label}/gpu_transfer/mean_ms": float(times.mean()),
        f"{label}/gpu_transfer/median_ms": float(np.median(times)),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

CELL_LABELS = {
    "hdf5_lazy":     "HDF5 lazy",
    "hdf5_eager":    "HDF5 eager",
    "zarr_lazy":     "Zarr lazy",
    "zarr_eager":    "Zarr eager",
    "lerobot_lazy":  "LeRobot lazy",
    "lerobot_eager": "LeRobot eager",
}
CELL_ORDER = list(CELL_LABELS.keys())


def build_cell(cell, args, window_starts, eager_arrays):
    t0 = time.perf_counter()
    if cell == "zarr_lazy":
        ds = _DPZarrLazyDataset(args.zarr_path, args.horizon, window_starts)
    elif cell == "hdf5_lazy":
        ds = _DPHDF5LazyDataset(args.hdf5_path, args.horizon, window_starts)
    elif cell == "zarr_eager":
        head, state, action, _ = eager_arrays
        ds = _DPEagerDataset(head, state, action, args.horizon, window_starts)
    elif cell == "hdf5_eager":
        head, state, action, _ = _build_eager_from_hdf5(args.hdf5_path)
        ds = _DPEagerDataset(head, state, action, args.horizon, window_starts)
    elif cell == "lerobot_lazy":
        ds = _DPLeRobotLazyDataset(
            args.lerobot_repo_id, args.lerobot_root, args.horizon, args.lerobot_fps)
    elif cell == "lerobot_eager":
        ds = _DPLeRobotEagerDataset(
            args.lerobot_repo_id, args.lerobot_root, args.horizon, window_starts)
    else:
        raise ValueError(cell)
    return ds, time.perf_counter() - t0


def run(args):
    print("=" * 72)
    print("  DP 3x2 Benchmark: (HDF5 vs Zarr vs LeRobot v2.1) x (Lazy vs Eager)")
    print("=" * 72)

    zarr_ok = zarr is not None and args.zarr_path and os.path.exists(args.zarr_path)
    hdf5_ok = h5py is not None and args.hdf5_path and os.path.exists(args.hdf5_path)
    lerobot_ok = LeRobotDataset is not None and bool(args.lerobot_repo_id)

    if not (zarr_ok or hdf5_ok):
        raise SystemExit("Need at least one of --zarr_path or --hdf5_path.")

    # Derive episode_ends + window starts from whichever source is present.
    if zarr_ok:
        print(f"\n[setup] Reading episode_ends from Zarr: {args.zarr_path}")
        _, _, _, episode_ends = _build_eager_from_zarr(args.zarr_path) \
            if args.zarr_path and os.path.exists(args.zarr_path) else (None, None, None, None)
        eager_arrays_zarr = _build_eager_from_zarr(args.zarr_path)
    else:
        print(f"\n[setup] Reading episode_ends from HDF5: {args.hdf5_path}")
        _, _, _, episode_ends = _build_eager_from_hdf5(args.hdf5_path)
        eager_arrays_zarr = None

    window_starts = build_window_indices(episode_ends, args.horizon)
    print(f"  episodes: {len(episode_ends)}   total windows: {len(window_starts)}")
    print(f"  horizon:  {args.horizon}   batch_size: {args.batch_size}")

    cells = []
    if hdf5_ok:
        cells += ["hdf5_lazy", "hdf5_eager"]
    if zarr_ok:
        cells += ["zarr_lazy", "zarr_eager"]
    if lerobot_ok:
        cells += ["lerobot_lazy", "lerobot_eager"]
    if not lerobot_ok:
        reason = _LEROBOT_IMPORT_ERR if LeRobotDataset is None else "no --lerobot_repo_id"
        print(f"  LeRobot unavailable ({reason}); skipping LeRobot cells.")

    worker_configs = [(0, 2), (2, 2), (4, 2), (8, 2)]
    all_results = {}

    for cell in cells:
        print(f"\n{'=' * 72}\n  Cell: {cell}\n{'=' * 72}")
        try:
            ds, init_sec = build_cell(cell, args, window_starts, eager_arrays_zarr)
        except Exception as exc:
            print(f"  FAILED to build cell {cell}: {exc}")
            all_results[f"{cell}/error"] = str(exc)
            continue
        all_results[f"{cell}/init_seconds"] = init_sec
        all_results[f"{cell}/rss_after_init_mb"] = peak_rss_mb()
        print(f"  init: {init_sec:.2f}s  RSS so far: {peak_rss_mb():.0f} MB  len={len(ds)}")

        print(f"  [1/3] raw __getitem__ (iters={args.num_iters})")
        all_results.update(bench_raw_io(ds, args.num_warmup, args.num_iters, cell))

        print(f"  [2/3] DataLoader sweep (batch_size={args.batch_size})")
        for nw, pf in worker_configs:
            print(f"    workers={nw}")
            all_results.update(bench_dataloader(
                ds, args.batch_size, nw, pf, args.num_warmup, args.num_iters, cell))

        print(f"  [3/3] GPU transfer")
        all_results.update(bench_gpu_transfer(
            ds, args.batch_size, args.num_warmup, args.num_iters, cell))

        del ds

    all_results["peak_rss_mb"] = peak_rss_mb()
    return all_results


def _row(name, values, fmt=".2f", width=13):
    out = []
    for v in values:
        if v is None:
            out.append(f"{'N/A':>{width}}")
        else:
            out.append(f"{v:>{width}{fmt}}")
    print(f"  {name:<28} " + "".join(out))


def report(results):
    cells = [c for c in CELL_ORDER if any(k.startswith(c + "/") for k in results)]
    labels = [CELL_LABELS[c] for c in cells]

    def header(title):
        print(f"\n--- {title} ---")
        print(f"  {'Metric':<28} " + "".join(f"{l:>13}" for l in labels))
        print("  " + "-" * (28 + 13 * len(cells)))

    def vals(suf):
        return [results.get(f"{c}/{suf}") for c in cells]

    print("\n" + "=" * 95)
    print("  DP 3x2 BENCHMARK RESULTS")
    print("=" * 95)

    header("Dataset init")
    _row("init_seconds", vals("init_seconds"), ".2f")
    _row("rss_after_init_mb", vals("rss_after_init_mb"), ".0f")

    header("Raw __getitem__ (ms)")
    for suf in ["mean_ms", "median_ms", "p95_ms", "p99_ms"]:
        _row(suf, vals(f"raw_io/{suf}"), ".2f")

    header("DataLoader samples/sec (higher = better)")
    configs = sorted({k.split("/")[2] for k in results
                      if "/dataloader/" in k and "samples_per_sec_mean" in k})
    for cfg in configs:
        _row(cfg, vals(f"dataloader/{cfg}/samples_per_sec_mean"), ".1f")

    header("GPU transfer (ms)")
    for suf in ["mean_ms", "median_ms"]:
        _row(suf, vals(f"gpu_transfer/{suf}"), ".2f")

    print(f"\n  Peak RSS overall: {results.get('peak_rss_mb', 0):.0f} MB")
    print("=" * 95)


def main():
    p = argparse.ArgumentParser(description="DP 3x2 data-format benchmark")
    p.add_argument("--zarr_path", type=str, default=None,
                   help="DP's zarr store (e.g. ./data/<task>.zarr)")
    p.add_argument("--hdf5_path", type=str, default=None,
                   help="HDF5 equivalent (produced by convert_dp_data.py)")
    p.add_argument("--lerobot_repo_id", type=str, default=None)
    p.add_argument("--lerobot_root", type=str, default=None)
    p.add_argument("--lerobot_fps", type=int, default=25,
                   help="FPS used when the LeRobot dataset was created")
    p.add_argument("--horizon", type=int, default=8, help="DP default: 8")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_warmup", type=int, default=5)
    p.add_argument("--num_iters", type=int, default=50)
    p.add_argument("--output", type=str, default="benchmark_results_dp.json")
    args = p.parse_args()

    results = run(args)
    report(results)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

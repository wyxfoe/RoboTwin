"""
Benchmark data loading for the ACT policy across a 3x2 matrix of
(file format) x (loading strategy) to decouple the two variables:

                          HDF5                 Zarr                LeRobot v2.1
    Lazy (on-disk)    HDF5-lazy           Zarr-lazy           LeRobot-lazy
    Eager (preload)   HDF5-eager          Zarr-eager          LeRobot-eager

    - Lazy   = open the file/store and read the timestep inside __getitem__
               (what ACT currently does for HDF5; LeRobot's default mode)
    - Eager  = load all episodes once in __init__ into numpy arrays,
               then slice from RAM inside __getitem__
               (what DP does via ReplayBuffer.copy_from_path)

Comparing rows isolates the storage-format effect under a fixed strategy;
comparing columns isolates the loading strategy effect under a fixed format.

Measures per cell:
  1. Raw I/O latency  — single-item __getitem__ speed
  2. DataLoader throughput — batches/sec under different worker configs
  3. GPU transfer time — data-to-CUDA overhead
  4. End-to-end training step — forward+backward with real ACT model
  5. Init time + peak RSS

Usage:
    python benchmark_data_formats.py \
        --hdf5_dir <path_to_hdf5_episodes> \
        --zarr_dir <path_to_zarr_episodes> \
        --lerobot_repo_id <repo_id> [--lerobot_root <root>] \
        --num_episodes 50 \
        --batch_size 8 \
        --num_warmup 5 \
        --num_iters 50

Any of --zarr_dir / --lerobot_repo_id may be omitted; the corresponding cells
are skipped automatically. LeRobot cells require the `lerobot` package.

Output: structured report printed to stdout and saved as benchmark_results.json.
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

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    _LEROBOT_IMPORT_ERR = None
except Exception as _exc:
    LeRobotDataset = None
    _LEROBOT_IMPORT_ERR = _exc

# ---------------------------------------------------------------------------
# Dataset implementations
# ---------------------------------------------------------------------------

class HDF5LazyDataset(torch.utils.data.Dataset):
    """HDF5 + lazy: open the file and read one timestep inside __getitem__.
    Mirrors the original ACT utils.py behavior."""

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


class ZarrLazyDataset(torch.utils.data.Dataset):
    """Zarr + lazy: open the Zarr store and read one timestep inside __getitem__."""

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
# Eager (preload-to-RAM) datasets — mirror DP's ReplayBuffer strategy
# ---------------------------------------------------------------------------

class _EagerEpisodicDataset(torch.utils.data.Dataset):
    """Shared postprocessing for eager datasets. Subclasses must populate
    self.episodes as a list of dicts: {action, qpos, images: {cam_name: np.ndarray}}.
    """

    def __init__(self, camera_names, norm_stats, max_action_len):
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.max_action_len = max_action_len
        self.episodes = []  # populated by subclass

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        ep = self.episodes[index]
        action = ep["action"]
        qpos_all = ep["qpos"]
        episode_len = action.shape[0]
        start_ts = np.random.randint(0, episode_len)

        qpos = qpos_all[start_ts]
        image_dict = {cam: ep["images"][cam][start_ts] for cam in self.camera_names}
        action_slice = action[start_ts:]
        action_len = episode_len - start_ts

        padded_action = np.zeros((self.max_action_len, action_slice.shape[1]), dtype=np.float32)
        padded_action[:action_len] = action_slice
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


class HDF5EagerDataset(_EagerEpisodicDataset):
    """HDF5 + eager: load every episode into numpy arrays at __init__.
    __getitem__ only touches RAM — no file I/O, no h5py, no decompression."""

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, max_action_len):
        super().__init__(camera_names, norm_stats, max_action_len)
        for ep_id in episode_ids:
            path = os.path.join(dataset_dir, f"episode_{ep_id}.hdf5")
            with h5py.File(path, "r") as f:
                ep = {
                    "action": f["/action"][()],
                    "qpos": f["/observations/qpos"][()],
                    "images": {
                        cam: f[f"/observations/images/{cam}"][()] for cam in camera_names
                    },
                }
            self.episodes.append(ep)


class ZarrEagerDataset(_EagerEpisodicDataset):
    """Zarr + eager: load every episode into numpy arrays at __init__.
    Equivalent to DP's ReplayBuffer.copy_from_path(store=None) path,
    which triggers arr[:] on every zarr array to decompress into numpy."""

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, max_action_len):
        super().__init__(camera_names, norm_stats, max_action_len)
        for ep_id in episode_ids:
            path = os.path.join(dataset_dir, f"episode_{ep_id}.zarr")
            root = zarr.open(path, mode="r")
            ep = {
                "action": root["action"][:],
                "qpos": root["observations/qpos"][:],
                "images": {
                    cam: root[f"observations/images/{cam}"][:] for cam in camera_names
                },
            }
            self.episodes.append(ep)


# ---------------------------------------------------------------------------
# LeRobot v2.1 datasets
# ---------------------------------------------------------------------------
#
# LeRobotDataset wraps parquet (state/action + indices) and either MP4 videos
# (use_videos=True) or PNG image folders (use_videos=False). Its __getitem__
# returns a dict keyed by feature name; images arrive as float32 CHW tensors
# already in [0, 1]. We adapt that to ACT's (image_data, qpos, action, is_pad)
# tuple so the benchmark is apples-to-apples.


def _make_lerobot_dataset(repo_id, root):
    """Instantiate a read-only LeRobotDataset (no HF Hub calls)."""
    if LeRobotDataset is None:
        raise RuntimeError(
            f"lerobot is not installed: {_LEROBOT_IMPORT_ERR}. "
            f"Install with `pip install lerobot`."
        )
    kwargs = {"repo_id": repo_id}
    if root is not None:
        kwargs["root"] = root
    # local_files_only avoids an implicit hub fetch.
    try:
        return LeRobotDataset(local_files_only=True, **kwargs)
    except TypeError:
        # older lerobot API
        return LeRobotDataset(**kwargs)


def _lerobot_episode_bounds(ds):
    """Return list[(from_idx, to_idx)] for each episode."""
    idx = ds.episode_data_index
    froms = idx["from"].tolist() if hasattr(idx["from"], "tolist") else list(idx["from"])
    tos = idx["to"].tolist() if hasattr(idx["to"], "tolist") else list(idx["to"])
    return list(zip(froms, tos))


def _lerobot_image_to_hwc_uint8(img_tensor):
    """Convert a LeRobot image tensor (CHW float[0,1] or HWC uint8) to HWC uint8."""
    if img_tensor.dtype == torch.uint8:
        if img_tensor.ndim == 3 and img_tensor.shape[0] in (1, 3):
            img_tensor = img_tensor.permute(1, 2, 0)
        return img_tensor.numpy()
    # float CHW in [0,1]
    arr = img_tensor.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


class LeRobotLazyDataset(torch.utils.data.Dataset):
    """LeRobot v2.1 + lazy: open parquet once, decode video frames on demand.

    Each __getitem__ pulls a single-frame sample for observation state/images
    (triggering a video seek+decode when use_videos=True) and a slice of the
    parquet action column for the action chunk.
    """

    def __init__(self, episode_ids, repo_id, root, camera_names, norm_stats, max_action_len):
        self.ds = _make_lerobot_dataset(repo_id, root)
        self.episode_bounds = _lerobot_episode_bounds(self.ds)
        self.episode_ids = episode_ids
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.max_action_len = max_action_len
        # hf_dataset is an Arrow-backed table; fast for raw column slicing.
        self._hf = self.ds.hf_dataset

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        ep_id = self.episode_ids[index]
        from_i, to_i = self.episode_bounds[ep_id]
        episode_len = to_i - from_i
        start_ts = np.random.randint(0, episode_len)
        global_idx = from_i + start_ts

        # Single-frame sample: triggers image decode + state read
        sample = self.ds[global_idx]
        qpos = sample["observation.state"]
        if isinstance(qpos, torch.Tensor):
            qpos_np = qpos.detach().cpu().numpy().astype(np.float32)
        else:
            qpos_np = np.asarray(qpos, dtype=np.float32)

        image_list = []
        for cam in self.camera_names:
            img = sample[f"observation.images.{cam}"]
            image_list.append(_lerobot_image_to_hwc_uint8(img))
        all_cam_images = np.stack(image_list, axis=0)

        # Action chunk from parquet (no video decode needed)
        action_rows = self._hf.select(range(global_idx, to_i))["action"]
        action = np.asarray(action_rows, dtype=np.float32)
        action_len = action.shape[0]

        padded_action = np.zeros((self.max_action_len, action.shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.ones(self.max_action_len, dtype=bool)
        is_pad[:action_len] = False

        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos_np).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad_t = torch.from_numpy(is_pad).bool()

        image_data = torch.einsum("k h w c -> k c h w", image_data)
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad_t


class LeRobotEagerDataset(_EagerEpisodicDataset):
    """LeRobot v2.1 + eager: decode every episode once at __init__.

    For the video backend this is the expensive path — every frame is decoded
    up front. After that, __getitem__ is identical to HDF5-eager / Zarr-eager.
    """

    def __init__(self, episode_ids, repo_id, root, camera_names, norm_stats, max_action_len):
        super().__init__(camera_names, norm_stats, max_action_len)
        ds = _make_lerobot_dataset(repo_id, root)
        bounds = _lerobot_episode_bounds(ds)
        hf = ds.hf_dataset

        for ep_id in episode_ids:
            from_i, to_i = bounds[ep_id]
            T = to_i - from_i
            # Fast-path: state + action are plain parquet columns.
            state_rows = hf.select(range(from_i, to_i))["observation.state"]
            action_rows = hf.select(range(from_i, to_i))["action"]
            qpos = np.asarray(state_rows, dtype=np.float32)
            action = np.asarray(action_rows, dtype=np.float32)

            # Images must go through the dataset so videos are decoded.
            # We pre-allocate per-camera arrays and fill frame by frame.
            first_sample = ds[from_i]
            imgs_hwc = {}
            for cam in camera_names:
                h, w, _ = _lerobot_image_to_hwc_uint8(
                    first_sample[f"observation.images.{cam}"]
                ).shape
                imgs_hwc[cam] = np.zeros((T, h, w, 3), dtype=np.uint8)
            for i in range(T):
                s = ds[from_i + i]
                for cam in camera_names:
                    imgs_hwc[cam][i] = _lerobot_image_to_hwc_uint8(
                        s[f"observation.images.{cam}"]
                    )

            self.episodes.append({
                "action": action,
                "qpos": qpos,
                "images": imgs_hwc,
            })


# ---------------------------------------------------------------------------
# Normalization stats (shared across both formats)
# ---------------------------------------------------------------------------

def get_norm_stats_from_hdf5(dataset_dir, num_episodes):
    """Compute normalization stats from HDF5 episodes. Returns torch tensors."""
    all_qpos, all_action = [], []
    for i in range(num_episodes):
        path = os.path.join(dataset_dir, f"episode_{i}.hdf5")
        with h5py.File(path, "r") as f:
            all_qpos.append(torch.from_numpy(f["/observations/qpos"][()]).float())
            all_action.append(torch.from_numpy(f["/action"][()]).float())

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

    # Keep as torch tensors to avoid numpy<->torch mixing in __getitem__
    stats = {
        "action_mean": all_action_t.mean(dim=[0, 1]),
        "action_std": all_action_t.std(dim=[0, 1]).clamp(min=1e-2),
        "qpos_mean": all_qpos_t.mean(dim=[0, 1]),
        "qpos_std": all_qpos_t.std(dim=[0, 1]).clamp(min=1e-2),
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

def _build_dataset(cell, episode_ids, args, camera_names, norm_stats, max_action_len):
    """Instantiate one of the six cells and return (dataset, init_seconds)."""
    t0 = time.perf_counter()
    if cell == "hdf5_lazy":
        ds = HDF5LazyDataset(episode_ids, args.hdf5_dir, camera_names, norm_stats, max_action_len)
    elif cell == "hdf5_eager":
        ds = HDF5EagerDataset(episode_ids, args.hdf5_dir, camera_names, norm_stats, max_action_len)
    elif cell == "zarr_lazy":
        ds = ZarrLazyDataset(episode_ids, args.zarr_dir, camera_names, norm_stats, max_action_len)
    elif cell == "zarr_eager":
        ds = ZarrEagerDataset(episode_ids, args.zarr_dir, camera_names, norm_stats, max_action_len)
    elif cell == "lerobot_lazy":
        ds = LeRobotLazyDataset(episode_ids, args.lerobot_repo_id, args.lerobot_root,
                                camera_names, norm_stats, max_action_len)
    elif cell == "lerobot_eager":
        ds = LeRobotEagerDataset(episode_ids, args.lerobot_repo_id, args.lerobot_root,
                                 camera_names, norm_stats, max_action_len)
    else:
        raise ValueError(cell)
    return ds, time.perf_counter() - t0


def run_benchmarks(args):
    print("=" * 70)
    print("  3x2 Benchmark: (HDF5 vs Zarr vs LeRobot v2.1) x (Lazy vs Eager)")
    print("=" * 70)

    camera_names = ["cam_high", "cam_right_wrist", "cam_left_wrist"]

    # Compute shared normalization stats
    print("\n[setup] Computing normalization statistics from HDF5 ...")
    norm_stats, max_action_len = get_norm_stats_from_hdf5(args.hdf5_dir, args.num_episodes)
    print(f"  max_action_len = {max_action_len}")

    episode_ids = list(range(args.num_episodes))
    zarr_ok = zarr is not None and args.zarr_dir and os.path.isdir(args.zarr_dir)
    if not zarr_ok:
        print(f"  Zarr unavailable (zarr_dir={args.zarr_dir}); skipping Zarr cells.")
    lerobot_ok = LeRobotDataset is not None and bool(args.lerobot_repo_id)
    if not lerobot_ok:
        reason = _LEROBOT_IMPORT_ERR if LeRobotDataset is None else "no --lerobot_repo_id"
        print(f"  LeRobot unavailable ({reason}); skipping LeRobot cells.")

    # 3x2 matrix of cells to test
    cells = ["hdf5_lazy", "hdf5_eager"]
    if zarr_ok:
        cells += ["zarr_lazy", "zarr_eager"]
    if lerobot_ok:
        cells += ["lerobot_lazy", "lerobot_eager"]

    all_results = {}

    # Allow skipping eager cells if they'd blow out RAM
    if args.skip_eager:
        cells = [c for c in cells if not c.endswith("eager")]
        print("  --skip_eager set, eager cells skipped.")

    worker_configs = [
        (0, 2),    # main process, prefetch_factor ignored
        (1, 1),    # original ACT config
        (2, 2),
        (4, 2),
        (4, 4),
        (8, 2),
    ]

    for cell in cells:
        print(f"\n{'=' * 70}\n  Cell: {cell}\n{'=' * 70}")
        dataset, init_sec = _build_dataset(cell, episode_ids, args, camera_names, norm_stats, max_action_len)
        all_results[f"{cell}/init_seconds"] = init_sec
        all_results[f"{cell}/rss_after_init_mb"] = get_peak_rss_mb()
        print(f"  init time: {init_sec:.2f}s   peak RSS so far: {get_peak_rss_mb():.0f} MB")

        # 1. Raw I/O latency
        print(f"\n  [1/4] Raw item latency (warmup={args.num_warmup}, iters={args.num_iters})")
        all_results.update(benchmark_raw_io(dataset, args.num_warmup, args.num_iters, cell))

        # 2. DataLoader throughput sweep
        print(f"\n  [2/4] DataLoader throughput (batch_size={args.batch_size})")
        for nw, pf in worker_configs:
            print(f"    workers={nw}, prefetch={pf}")
            all_results.update(benchmark_dataloader(
                dataset, args.batch_size, nw, pf, args.num_warmup, args.num_iters, cell))

        # 3. GPU transfer
        print(f"\n  [3/4] GPU transfer")
        all_results.update(benchmark_gpu_transfer(dataset, args.batch_size, args.num_warmup, args.num_iters, cell))

        # 4. End-to-end training step
        if not args.skip_training:
            print(f"\n  [4/4] End-to-end training step")
            all_results.update(benchmark_training_step(
                dataset, args.batch_size, args.num_warmup, min(args.num_iters, 20), cell, camera_names))
        else:
            print(f"\n  [4/4] Skipped (--skip_training)")

        # Release memory before the next cell
        del dataset

    all_results["peak_rss_mb"] = get_peak_rss_mb()
    return all_results


CELL_LABELS = {
    "hdf5_lazy":     "HDF5 lazy",
    "hdf5_eager":    "HDF5 eager",
    "zarr_lazy":     "Zarr lazy",
    "zarr_eager":    "Zarr eager",
    "lerobot_lazy":  "LeRobot lazy",
    "lerobot_eager": "LeRobot eager",
}

CELL_ORDER = ["hdf5_lazy", "hdf5_eager", "zarr_lazy", "zarr_eager",
              "lerobot_lazy", "lerobot_eager"]


def _row(name, values, fmt=".2f", width=12):
    """Print one row of the 2x2 comparison table."""
    cells = []
    for v in values:
        if v is None:
            cells.append(f"{'N/A':>{width}}")
        else:
            cells.append(f"{v:>{width}{fmt}}")
    print(f"  {name:<30} " + "".join(cells))


def print_report(results):
    """Pretty-print the 2x2 matrix results.

    Layout per metric:
                          HDF5 lazy   HDF5 eager  Zarr lazy   Zarr eager
        metric_name          ...         ...         ...         ...
    """
    cells = [c for c in CELL_ORDER if any(k.startswith(c + "/") for k in results)]
    labels = [CELL_LABELS[c] for c in cells]

    def header(title):
        print(f"\n--- {title} ---")
        print(f"  {'Metric':<30} " + "".join(f"{l:>12}" for l in labels))
        print("  " + "-" * (30 + 12 * len(cells)))

    def get_vals(key_suffix):
        return [results.get(f"{c}/{key_suffix}") for c in cells]

    print("\n" + "=" * 95)
    print("  3x2 BENCHMARK RESULTS  (rows = metric, columns = cell)")
    print("=" * 95)

    # --- Init time & RSS ---
    header("Dataset init")
    _row("init_seconds", get_vals("init_seconds"), fmt=".2f")
    _row("rss_after_init_mb", get_vals("rss_after_init_mb"), fmt=".0f")

    # --- Raw I/O ---
    header("Raw __getitem__ latency (ms)")
    for suffix in ["mean_ms", "median_ms", "p95_ms", "p99_ms"]:
        _row(suffix, get_vals(f"raw_io/{suffix}"), fmt=".2f")

    # --- DataLoader ---
    header("DataLoader throughput (samples/sec)")
    configs = set()
    for k in results:
        if "dataloader" in k and "samples_per_sec_mean" in k:
            configs.add(k.split("/")[2])
    for cfg in sorted(configs):
        _row(cfg, get_vals(f"dataloader/{cfg}/samples_per_sec_mean"), fmt=".1f")

    # --- GPU Transfer ---
    header("GPU transfer (ms)")
    for suffix in ["mean_ms", "median_ms", "p95_ms"]:
        _row(suffix, get_vals(f"gpu_transfer/{suffix}"), fmt=".2f")

    # --- Training Step ---
    header("End-to-end training step (ms)")
    for suffix in ["total_mean_ms", "data_mean_ms", "compute_mean_ms"]:
        _row(suffix, get_vals(f"train_step/{suffix}"), fmt=".1f")
    _row("data_pct (%)", get_vals("train_step/data_pct"), fmt=".1f")

    print(f"\n  Peak RSS overall: {results.get('peak_rss_mb', 'N/A'):.0f} MB")

    # --- Paired comparisons ---
    print("\n--- Isolated effects ---")

    def _ratio(a_key, b_key, label):
        a = results.get(a_key)
        b = results.get(b_key)
        if a and b and a > 0:
            print(f"  {label:<45} {b/a:.2f}x")

    print("  (Format effect under same loading strategy, HDF5 = baseline)")
    _ratio("hdf5_lazy/train_step/data_mean_ms",
           "zarr_lazy/train_step/data_mean_ms",
           "zarr_lazy     / hdf5_lazy  (data time ratio)")
    _ratio("hdf5_lazy/train_step/data_mean_ms",
           "lerobot_lazy/train_step/data_mean_ms",
           "lerobot_lazy  / hdf5_lazy  (data time ratio)")
    _ratio("hdf5_eager/train_step/data_mean_ms",
           "zarr_eager/train_step/data_mean_ms",
           "zarr_eager    / hdf5_eager (data time ratio)")
    _ratio("hdf5_eager/train_step/data_mean_ms",
           "lerobot_eager/train_step/data_mean_ms",
           "lerobot_eager / hdf5_eager (data time ratio)")

    print("  (Strategy effect under same format, lazy = baseline)")
    _ratio("hdf5_lazy/train_step/data_mean_ms",
           "hdf5_eager/train_step/data_mean_ms",
           "hdf5_eager    / hdf5_lazy    (data time ratio)")
    _ratio("zarr_lazy/train_step/data_mean_ms",
           "zarr_eager/train_step/data_mean_ms",
           "zarr_eager    / zarr_lazy    (data time ratio)")
    _ratio("lerobot_lazy/train_step/data_mean_ms",
           "lerobot_eager/train_step/data_mean_ms",
           "lerobot_eager / lerobot_lazy (data time ratio)")

    print("=" * 95)


def main():
    parser = argparse.ArgumentParser(description="Benchmark HDF5 vs Zarr for ACT policy")
    parser.add_argument("--hdf5_dir", type=str, required=True,
                        help="Directory containing episode_*.hdf5 files")
    parser.add_argument("--zarr_dir", type=str, default=None,
                        help="Directory containing episode_*.zarr stores")
    parser.add_argument("--lerobot_repo_id", type=str, default=None,
                        help="LeRobot v2.1 repo_id (omit to skip LeRobot cells)")
    parser.add_argument("--lerobot_root", type=str, default=None,
                        help="Root dir for LeRobot data (defaults to HF_LEROBOT_HOME)")
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
    parser.add_argument("--skip_eager", action="store_true",
                        help="Skip eager cells (saves RAM for large datasets)")
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

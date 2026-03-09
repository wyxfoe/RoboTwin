"""RoboTwin HDF5 dataset loader for AudioX training.

HDF5 structure (per episode):
    observation/<camera>/rgb   — (T,) JPEG-encoded byte strings
    joint_action/vector        — (T, action_dim) float32
"""

import glob
import math
import os

import cv2
import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler

from .action_space import normalize_actions, compute_action_stats

# CLIP-ViT-B/32 normalization constants
CLIP_IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _decode_jpeg(jpeg_bytes):
    """Decode a JPEG-encoded byte string to a uint8 RGB image."""
    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _preprocess_image(img, size):
    """Resize and CLIP-normalize a uint8 RGB image to (C, H, W) tensor."""
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - CLIP_IMAGE_MEAN) / CLIP_IMAGE_STD
    return img.transpose(2, 0, 1)  # (C, H, W)


class RoboTwinDataset(Dataset):
    """Loads RoboTwin HDF5 episodes and yields (action_chunk, metadata) pairs.

    Each sample:
        action_chunk:  (action_dim, chunk_size) float32 — audio-format layout
        metadata:      dict with keys "prompt", "video", "proprio"
    """

    def __init__(
        self,
        data_dir,
        action_chunk_size=50,
        image_size=224,
        camera_names=None,
        task_description="complete the task",
        normalize=True,
        augment=False,
        max_episodes=None,
        action_stats=None,
    ):
        super().__init__()
        self.action_chunk_size = action_chunk_size
        self.image_size = image_size
        self.camera_names = camera_names or ["head_camera", "left_camera", "right_camera"]
        self.task_description = task_description
        self.normalize = normalize
        self.augment = augment

        # Discover episode files
        patterns = [
            os.path.join(data_dir, "episode*.hdf5"),
            os.path.join(data_dir, "**", "episode*.hdf5"),
            os.path.join(data_dir, "data", "episode*.hdf5"),
        ]
        self.episode_paths = []
        for pat in patterns:
            self.episode_paths.extend(sorted(glob.glob(pat, recursive=True)))
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for p in self.episode_paths:
            rp = os.path.realpath(p)
            if rp not in seen:
                seen.add(rp)
                unique.append(p)
        self.episode_paths = unique

        if max_episodes is not None:
            self.episode_paths = self.episode_paths[:max_episodes]

        if len(self.episode_paths) == 0:
            raise FileNotFoundError(
                f"No episode*.hdf5 found in {data_dir}. "
                "Check that the data directory contains HDF5 episodes."
            )

        # Pre-load episode metadata (lengths) and build sample index
        self.episodes = []  # list of (path, episode_length)
        self.samples = []   # list of (episode_idx, start_t)
        all_actions = []

        for ep_idx, ep_path in enumerate(self.episode_paths):
            with h5py.File(ep_path, "r") as f:
                actions = f["joint_action"]["vector"][:]  # (T, action_dim)
            ep_len = actions.shape[0]
            self.episodes.append((ep_path, ep_len))
            all_actions.append(actions)

            # Create samples: every possible chunk start
            num_chunks = max(1, ep_len - action_chunk_size + 1)
            for t in range(num_chunks):
                self.samples.append((ep_idx, t))

        # Compute or use provided action statistics
        if action_stats is not None:
            self.action_stats = action_stats
        else:
            self.action_stats = compute_action_stats(all_actions)

        print(f"[RoboTwinDataset] {len(self.episode_paths)} episodes, "
              f"{len(self.samples)} samples, "
              f"action_dim={all_actions[0].shape[1]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, start_t = self.samples[idx]
        ep_path, ep_len = self.episodes[ep_idx]

        with h5py.File(ep_path, "r") as f:
            # --- Actions ---
            end_t = min(start_t + self.action_chunk_size, ep_len)
            actions = f["joint_action"]["vector"][start_t:end_t]  # (chunk, dim)
            actions = actions.astype(np.float32)

            # Pad if chunk shorter than target
            if actions.shape[0] < self.action_chunk_size:
                pad = np.tile(actions[-1:], (self.action_chunk_size - actions.shape[0], 1))
                actions = np.concatenate([actions, pad], axis=0)

            # --- Proprio state (at start timestep) ---
            proprio = f["joint_action"]["vector"][start_t].astype(np.float32)

            # --- Camera images (at start timestep) ---
            frames = []
            for cam_name in self.camera_names:
                cam_key = f"observation/{cam_name}/rgb"
                if cam_key not in f:
                    # Skip missing cameras, use zeros
                    frames.append(np.zeros((3, self.image_size, self.image_size), dtype=np.float32))
                    continue
                jpeg_bytes = f[cam_key][start_t]
                img = _decode_jpeg(jpeg_bytes)
                img = _preprocess_image(img, self.image_size)
                frames.append(img)

        # Normalize actions to [-1, 1]
        actions_t = torch.from_numpy(actions)
        if self.normalize:
            actions_t = normalize_actions(actions_t, self.action_stats)

        # Transpose to audio format: (action_dim, chunk_size)
        actions_t = actions_t.T  # (action_dim, chunk_size)

        # Build metadata dict for conditioner
        # Each camera image is a separate (C, H, W) tensor with CLIP normalization.
        # Training code will process each through CLIP independently and concatenate
        # the token sequences (RDT-style), instead of using temporal transformer.
        camera_images = [torch.from_numpy(frame) for frame in frames]  # list of (C, H, W)
        proprio_tensor = torch.from_numpy(proprio)

        metadata = {
            "prompt": self.task_description,
            "camera_images": camera_images,
            "proprio": proprio_tensor,
        }

        return actions_t, metadata


class MultiTaskRoboTwinDataset(ConcatDataset):
    """Concatenates multiple single-task RoboTwinDatasets with shared action stats.

    Each task gets its own task_description that is passed as the "prompt"
    conditioning to T5, allowing the model to distinguish tasks.

    Usage:
        dataset = MultiTaskRoboTwinDataset(
            task_dirs=[
                ("/path/to/beat_block_hammer/data", "use the hammer to beat the block"),
                ("/path/to/lift_pot/data", "lift the pot with both hands"),
            ],
            ...
        )
    """

    def __init__(
        self,
        task_dirs,
        action_chunk_size=50,
        image_size=224,
        camera_names=None,
        normalize=True,
        augment=False,
        max_episodes=None,
    ):
        # First pass: collect all actions across tasks for shared normalization
        all_actions = []
        for data_dir, _ in task_dirs:
            patterns = [
                os.path.join(data_dir, "episode*.hdf5"),
                os.path.join(data_dir, "**", "episode*.hdf5"),
                os.path.join(data_dir, "data", "episode*.hdf5"),
            ]
            ep_paths = []
            for pat in patterns:
                ep_paths.extend(sorted(glob.glob(pat, recursive=True)))
            seen = set()
            for p in ep_paths:
                rp = os.path.realpath(p)
                if rp not in seen:
                    seen.add(rp)
                    with h5py.File(p, "r") as f:
                        all_actions.append(f["joint_action"]["vector"][:])

        shared_stats = compute_action_stats(all_actions)
        self.action_stats = shared_stats

        # Second pass: create per-task datasets with shared stats
        datasets = []
        for data_dir, task_desc in task_dirs:
            ds = RoboTwinDataset(
                data_dir=data_dir,
                action_chunk_size=action_chunk_size,
                image_size=image_size,
                camera_names=camera_names,
                task_description=task_desc,
                normalize=normalize,
                augment=augment,
                max_episodes=max_episodes,
                action_stats=shared_stats,
            )
            datasets.append(ds)

        super().__init__(datasets)

        self.task_dirs = task_dirs
        total_episodes = sum(len(d.episode_paths) for d in datasets)
        print(f"[MultiTaskRoboTwinDataset] {len(task_dirs)} tasks, "
              f"{total_episodes} total episodes, {len(self)} total samples")


class WeightedTaskSampler(Sampler):
    """Weighted random sampler for multi-task cross-task transfer learning.

    Assigns per-sample weights based on which task each sample belongs to,
    so that tasks with fewer samples can be upweighted to ensure balanced
    exposure during training. This prevents dominant tasks from drowning out
    smaller tasks and improves cross-task transfer.

    Supports DDP: each rank samples a disjoint subset of indices per epoch.
    Call ``set_epoch(epoch)`` before each epoch to ensure different shuffles
    across epochs (following the DistributedSampler convention used by
    OpenVLA, DexVLA, and other mainstream VLA frameworks).

    Weighting strategies:
        "uniform":    All tasks sampled equally regardless of size.
                      Weight per sample = 1 / (num_tasks * task_size).
        "sqrt":       Tasks weighted proportional to sqrt(task_size),
                      a balanced compromise between uniform and natural.
        "natural":    No reweighting; tasks sampled proportional to their
                      natural size (equivalent to standard shuffled sampling).
        "custom":     User-specified per-task weights via task_weights arg.

    Temperature (following RDT convention):
        When temperature != 1.0, raw task-level weights are raised to
        the power of (1 / temperature) before normalization:
            w_i' = w_i^(1/T) / sum(w_j^(1/T))
        T → 0: converges to argmax (only largest-weight task sampled).
        T = 1: no change.
        T → ∞: converges to uniform sampling.

    Args:
        dataset: A MultiTaskRoboTwinDataset (ConcatDataset of per-task datasets).
        strategy: One of "uniform", "sqrt", "natural", "custom".
        task_weights: List of floats, one per task. Only used when strategy="custom".
        num_samples: Total samples per epoch. Defaults to len(dataset).
        temperature: Smoothing temperature applied to task weights (default 1.0).
        seed: Base random seed for reproducibility across DDP ranks.
    """

    def __init__(
        self,
        dataset,
        strategy="uniform",
        task_weights=None,
        num_samples=None,
        temperature=1.0,
        seed=0,
    ):
        if not isinstance(dataset, ConcatDataset):
            raise TypeError(
                "WeightedTaskSampler requires a ConcatDataset (MultiTaskRoboTwinDataset). "
                f"Got {type(dataset).__name__}."
            )

        self.dataset = dataset
        self.seed = seed
        self.epoch = 0
        self.temperature = temperature

        # DDP awareness (matches DistributedSampler convention)
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        total_dataset_size = len(dataset)
        # Per-rank sample count (round up so all ranks get the same count)
        if num_samples is not None:
            self.total_samples = num_samples
        else:
            self.total_samples = total_dataset_size
        self.num_samples = math.ceil(self.total_samples / self.world_size)

        # Build per-task sizes from the ConcatDataset's cumulative_sizes
        task_sizes = []
        prev = 0
        for cs in dataset.cumulative_sizes:
            task_sizes.append(cs - prev)
            prev = cs
        num_tasks = len(task_sizes)

        # Compute per-task target weights
        if strategy == "uniform":
            target_weights = [1.0 / num_tasks] * num_tasks
        elif strategy == "sqrt":
            sqrt_sizes = [math.sqrt(s) for s in task_sizes]
            total_sqrt = sum(sqrt_sizes)
            target_weights = [s / total_sqrt for s in sqrt_sizes]
        elif strategy == "natural":
            total = sum(task_sizes)
            target_weights = [s / total for s in task_sizes]
        elif strategy == "custom":
            if task_weights is None or len(task_weights) != num_tasks:
                raise ValueError(
                    f"strategy='custom' requires task_weights with {num_tasks} entries, "
                    f"got {task_weights}"
                )
            total_w = sum(task_weights)
            target_weights = [w / total_w for w in task_weights]
        else:
            raise ValueError(
                f"Unknown sampling strategy '{strategy}'. "
                "Choose from: uniform, sqrt, natural, custom."
            )

        # Apply temperature scaling (RDT-style)
        if temperature != 1.0 and temperature > 0:
            inv_t = 1.0 / temperature
            target_weights = [w ** inv_t for w in target_weights]
            total_w = sum(target_weights)
            target_weights = [w / total_w for w in target_weights]

        # Convert target task weights to per-sample weights
        # sample_weight[i] = target_weight[task_of_i] / task_size[task_of_i]
        sample_weights = torch.zeros(total_dataset_size, dtype=torch.float64)
        offset = 0
        for task_idx, task_size in enumerate(task_sizes):
            w = target_weights[task_idx] / task_size
            sample_weights[offset:offset + task_size] = w
            offset += task_size

        self.sample_weights = sample_weights

        # Log effective distribution (only on rank 0)
        if self.rank == 0:
            print(f"[WeightedTaskSampler] strategy={strategy}, temperature={temperature}, "
                  f"{num_tasks} tasks, {self.total_samples} total samples/epoch, "
                  f"world_size={self.world_size}")
            for i, (ts, tw) in enumerate(zip(task_sizes, target_weights)):
                print(f"  task[{i}]: {ts} samples, target weight={tw:.4f}")

    def set_epoch(self, epoch):
        """Set epoch number to change the random seed for shuffling per epoch.

        This follows the DistributedSampler convention and ensures different
        sampling orders across epochs for better training convergence.
        """
        self.epoch = epoch

    def __iter__(self):
        # Create a generator seeded by epoch + rank for reproducibility
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Sample indices for all ranks, then slice for this rank
        all_indices = torch.multinomial(
            self.sample_weights,
            num_samples=self.num_samples * self.world_size,
            replacement=True,
            generator=g,
        )

        # Each rank takes its own slice
        indices = all_indices[self.rank::self.world_size]
        yield from indices.tolist()

    def __len__(self):
        return self.num_samples


def _collate_fn(batch):
    """Custom collate: stack actions, keep metadata as list of dicts."""
    actions_list, meta_list = zip(*batch)
    actions = torch.stack(actions_list, dim=0)  # (batch, action_dim, chunk_size)
    return actions, list(meta_list)


def create_robotwin_dataloader(
    data_dir,
    batch_size=16,
    action_chunk_size=50,
    image_size=224,
    camera_names=None,
    task_description="complete the task",
    num_workers=4,
    max_episodes=None,
    normalize=True,
    augment=False,
    shuffle=True,
    action_stats=None,
    task_dirs=None,
    sampling_strategy="uniform",
    task_weights=None,
    sampling_temperature=1.0,
):
    """Create a DataLoader and Dataset for RoboTwin training.

    For multi-task training, pass task_dirs as a list of (data_dir, task_description)
    tuples. When task_dirs is provided, data_dir and task_description are ignored.

    Args:
        sampling_strategy: Weighting strategy for multi-task sampling.
            "uniform" — equal task exposure, "sqrt" — sqrt-balanced,
            "natural" — proportional to data size, "custom" — use task_weights.
        sampling_temperature: Temperature for smoothing task weights (default 1.0).
        task_weights: Per-task weight list for strategy="custom".

    Returns:
        (dataloader, dataset) tuple.
    """
    sampler = None

    if task_dirs is not None and len(task_dirs) > 0:
        dataset = MultiTaskRoboTwinDataset(
            task_dirs=task_dirs,
            action_chunk_size=action_chunk_size,
            image_size=image_size,
            camera_names=camera_names,
            normalize=normalize,
            augment=augment,
            max_episodes=max_episodes,
        )
        # Use weighted sampler for multi-task training
        sampler = WeightedTaskSampler(
            dataset=dataset,
            strategy=sampling_strategy,
            task_weights=task_weights,
            temperature=sampling_temperature,
        )
    else:
        dataset = RoboTwinDataset(
            data_dir=data_dir,
            action_chunk_size=action_chunk_size,
            image_size=image_size,
            camera_names=camera_names,
            task_description=task_description,
            normalize=normalize,
            augment=augment,
            max_episodes=max_episodes,
            action_stats=action_stats,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate_fn,
        persistent_workers=num_workers > 0,
    )

    return dataloader, dataset

"""
Zarr-based dataset and data loading utilities for ACT policy.

Reads zarr stores produced by process_data_zarr.py with the structure:
  zarr_root/
    data/
      cam_high, cam_left_wrist, cam_right_wrist: (N, C, H, W) uint8
      qpos:   (N, D) float32
      action: (N, D) float32
    meta/
      episode_ends: (num_episodes,) int64

Provides the same (image_data, qpos_data, action_data, is_pad) interface
as the original HDF5-based EpisodicDataset, so the ACT training loop
works without modification.
"""

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader


class ZarrEpisodicDataset(torch.utils.data.Dataset):
    """
    Dataset that reads a zarr store and samples single timesteps per episode,
    matching the original ACT EpisodicDataset behavior:
      - Randomly pick a start_ts within an episode
      - Return the observation (images + qpos) at start_ts
      - Return all actions from start_ts to end, padded to max_action_len
    """

    def __init__(self, episode_ids, zarr_path, camera_names, norm_stats, max_action_len):
        super().__init__()
        self.episode_ids = episode_ids
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.max_action_len = max_action_len

        # Open zarr store and load data into memory for fast access
        root = zarr.open(zarr_path, mode="r")
        self.episode_ends = root["meta"]["episode_ends"][:]

        # Pre-compute episode starts
        self.episode_starts = np.zeros_like(self.episode_ends)
        self.episode_starts[1:] = self.episode_ends[:-1]

        # Load all data arrays (kept as zarr arrays for lazy loading,
        # or loaded to memory if small enough)
        self.data = {}
        for cam_name in camera_names:
            self.data[cam_name] = root["data"][cam_name]
        self.data["qpos"] = root["data"]["qpos"]
        self.data["action"] = root["data"]["action"]

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        start = self.episode_starts[episode_id]
        end = self.episode_ends[episode_id]
        episode_len = end - start

        # Random timestep within episode
        start_ts = np.random.choice(episode_len)
        global_ts = start + start_ts

        # Get observation at start_ts
        qpos = self.data["qpos"][global_ts].astype(np.float32)

        # Get images at start_ts for each camera
        all_cam_images = []
        for cam_name in self.camera_names:
            img = self.data[cam_name][global_ts]  # (C, H, W) uint8
            all_cam_images.append(img)
        all_cam_images = np.stack(all_cam_images, axis=0)  # (K, C, H, W)

        # Get actions from start_ts onward
        action = self.data["action"][global_ts:end].astype(np.float32)
        action_len = end - global_ts

        # Pad actions to max_action_len
        padded_action = np.zeros((self.max_action_len, action.shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.ones(self.max_action_len, dtype=bool)
        is_pad[:action_len] = False

        # Convert to tensors
        image_data = torch.from_numpy(all_cam_images).float()
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # Images are already in NCHW format from zarr, normalize to [0, 1]
        image_data = image_data / 255.0

        # Normalize action and qpos
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats_zarr(zarr_path):
    """Compute normalization statistics from the zarr store."""
    root = zarr.open(zarr_path, mode="r")
    episode_ends = root["meta"]["episode_ends"][:]
    num_episodes = len(episode_ends)

    all_qpos_data = []
    all_action_data = []

    episode_starts = np.zeros_like(episode_ends)
    episode_starts[1:] = episode_ends[:-1]

    for ep_idx in range(num_episodes):
        start = episode_starts[ep_idx]
        end = episode_ends[ep_idx]
        qpos = torch.from_numpy(root["data"]["qpos"][start:end].astype(np.float32))
        action = torch.from_numpy(root["data"]["action"][start:end].astype(np.float32))
        all_qpos_data.append(qpos)
        all_action_data.append(action)

    # Pad all tensors to the maximum size
    max_qpos_len = max(q.size(0) for q in all_qpos_data)
    max_action_len = max(a.size(0) for a in all_action_data)

    padded_qpos = []
    for qpos in all_qpos_data:
        current_len = qpos.size(0)
        if current_len < max_qpos_len:
            pad = qpos[-1:].repeat(max_qpos_len - current_len, 1)
            qpos = torch.cat([qpos, pad], dim=0)
        padded_qpos.append(qpos)

    padded_action = []
    for action in all_action_data:
        current_len = action.size(0)
        if current_len < max_action_len:
            pad = action[-1:].repeat(max_action_len - current_len, 1)
            action = torch.cat([action, pad], dim=0)
        padded_action.append(action)

    all_qpos_data = torch.stack(padded_qpos)
    all_action_data = torch.stack(padded_action)

    # Compute stats
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)

    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": all_qpos_data[0][-1].numpy(),
    }

    return stats, max_action_len


def load_data_zarr(zarr_path, camera_names, batch_size_train, batch_size_val):
    """
    Load data from a zarr store and create train/val dataloaders.

    Args:
        zarr_path: Path to the zarr store
        camera_names: List of camera names to load (e.g., ["cam_high", "cam_left_wrist", "cam_right_wrist"])
        batch_size_train: Training batch size
        batch_size_val: Validation batch size

    Returns:
        train_dataloader, val_dataloader, norm_stats
    """
    print(f"\nData from zarr: {zarr_path}\n")

    # Get episode count
    root = zarr.open(zarr_path, mode="r")
    num_episodes = len(root["meta"]["episode_ends"][:])

    # Train/val split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # Compute normalization stats
    norm_stats, max_action_len = get_norm_stats_zarr(zarr_path)

    # Create datasets
    train_dataset = ZarrEpisodicDataset(train_indices, zarr_path, camera_names, norm_stats, max_action_len)
    val_dataset = ZarrEpisodicDataset(val_indices, zarr_path, camera_names, norm_stats, max_action_len)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )

    return train_dataloader, val_dataloader, norm_stats

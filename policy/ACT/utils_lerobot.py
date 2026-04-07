"""
LeRobotDataset-based data loading utilities for ACT policy.

Wraps a LeRobotDataset (produced by process_data_lerobot.py) and converts
each sample into the (image_data, qpos_data, action_data, is_pad) format
expected by the ACT training loop.

The key difference from the original HDF5-based EpisodicDataset:
  - Data is stored in the LeRobot v2/v3 format (Parquet + images/videos)
  - We use LeRobotDataset's delta_timestamps to load action chunks
  - Normalization stats are computed from the LeRobot dataset
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


class LeRobotACTDataset(torch.utils.data.Dataset):
    """
    Wraps a LeRobotDataset and converts samples to ACT's expected format:
      (image_data, qpos_data, action_data, is_pad)

    Uses delta_timestamps to load a chunk of future actions for each timestep.
    """

    def __init__(self, lerobot_dataset, camera_names, norm_stats, chunk_size):
        super().__init__()
        self.dataset = lerobot_dataset
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]

        # Get qpos (observation.state)
        qpos = sample["observation.state"].numpy().astype(np.float32)

        # Get images from all cameras and stack them
        all_cam_images = []
        for cam_name in self.camera_names:
            key = f"observation.images.{cam_name}"
            img = sample[key]  # (C, H, W) or (H, W, C)
            if img.ndim == 3 and img.shape[-1] == 3:
                # HWC -> CHW
                img = img.permute(2, 0, 1)
            all_cam_images.append(img.numpy())
        all_cam_images = np.stack(all_cam_images, axis=0)  # (K, C, H, W)

        # Get action chunk via delta_timestamps
        # The action key with delta_timestamps returns (chunk_size, D)
        action = sample["action"]
        if action.ndim == 1:
            # Single action, expand
            action = action.unsqueeze(0)
        action = action.numpy().astype(np.float32)

        # Pad or truncate to chunk_size
        action_len = min(action.shape[0], self.chunk_size)
        padded_action = np.zeros((self.chunk_size, action.shape[-1]), dtype=np.float32)
        padded_action[:action_len] = action[:action_len]
        is_pad = np.ones(self.chunk_size, dtype=bool)
        is_pad[:action_len] = False

        # Convert to tensors
        image_data = torch.from_numpy(all_cam_images).float()
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # Normalize images to [0, 1]
        if image_data.max() > 1.0:
            image_data = image_data / 255.0

        # Normalize action and qpos
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats_lerobot(repo_id, action_key="action", state_key="observation.state"):
    """
    Compute normalization statistics from a LeRobotDataset.

    Loads the full dataset and computes mean/std for actions and qpos.
    """
    dataset = LeRobotDataset(repo_id)

    all_qpos = []
    all_actions = []

    for i in range(len(dataset)):
        sample = dataset[i]
        all_qpos.append(sample[state_key].numpy())
        all_actions.append(sample[action_key].numpy())

    all_qpos = torch.from_numpy(np.array(all_qpos)).float()
    all_actions = torch.from_numpy(np.array(all_actions)).float()

    # Compute stats
    action_mean = all_actions.mean(dim=0, keepdim=True)
    action_std = all_actions.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)

    qpos_mean = all_qpos.mean(dim=0, keepdim=True)
    qpos_std = all_qpos.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": all_qpos[-1].numpy(),
    }

    return stats


def load_data_lerobot(repo_id, camera_names, chunk_size, batch_size_train, batch_size_val):
    """
    Load data from a LeRobotDataset and create train/val dataloaders.

    Args:
        repo_id: LeRobot dataset repository ID (local or HuggingFace Hub)
        camera_names: List of camera names (e.g., ["cam_high", "cam_left_wrist", "cam_right_wrist"])
        chunk_size: Number of future actions to predict (ACT chunk_size / num_queries)
        batch_size_train: Training batch size
        batch_size_val: Validation batch size

    Returns:
        train_dataloader, val_dataloader, norm_stats
    """
    print(f"\nData from LeRobot: {repo_id}\n")

    # Compute normalization stats
    norm_stats = get_norm_stats_lerobot(repo_id)

    # Get metadata for fps
    meta = LeRobotDatasetMetadata(repo_id)
    fps = meta.fps

    # Create dataset with delta_timestamps for action chunking
    delta_timestamps = {
        "action": [t / fps for t in range(chunk_size)],
    }
    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)

    # Train/val split
    total_len = len(dataset)
    train_ratio = 0.8
    indices = np.random.permutation(total_len)
    train_size = int(train_ratio * total_len)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create wrapped datasets
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    train_wrapped = _SubsetWrapper(train_subset, camera_names, norm_stats, chunk_size)
    val_wrapped = _SubsetWrapper(val_subset, camera_names, norm_stats, chunk_size)

    train_dataloader = DataLoader(
        train_wrapped,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )
    val_dataloader = DataLoader(
        val_wrapped,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )

    return train_dataloader, val_dataloader, norm_stats


class _SubsetWrapper(torch.utils.data.Dataset):
    """Wraps a Subset of LeRobotDataset, converting to ACT format."""

    def __init__(self, subset, camera_names, norm_stats, chunk_size):
        self.subset = subset
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        sample = self.subset[index]

        # Get qpos
        qpos = sample["observation.state"].numpy().astype(np.float32)

        # Get images
        all_cam_images = []
        for cam_name in self.camera_names:
            key = f"observation.images.{cam_name}"
            img = sample[key]
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)
            all_cam_images.append(img.numpy())
        all_cam_images = np.stack(all_cam_images, axis=0)

        # Get action chunk
        action = sample["action"]
        if action.ndim == 1:
            action = action.unsqueeze(0)
        action = action.numpy().astype(np.float32)

        # Pad or truncate
        action_len = min(action.shape[0], self.chunk_size)
        padded_action = np.zeros((self.chunk_size, action.shape[-1]), dtype=np.float32)
        padded_action[:action_len] = action[:action_len]
        is_pad = np.ones(self.chunk_size, dtype=bool)
        is_pad[:action_len] = False

        # Convert to tensors
        image_data = torch.from_numpy(all_cam_images).float()
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # Normalize
        if image_data.max() > 1.0:
            image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad

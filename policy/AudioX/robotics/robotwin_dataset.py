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
from torch.utils.data import Dataset, DataLoader

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
        # AudioX CLIPConditioner expects (1, num_frames, C, H, W) per sample;
        # it concatenates along dim=0 across the batch to get (B, T, C, H, W).
        video_tensor = torch.from_numpy(np.stack(frames, axis=0)).unsqueeze(0)  # (1, num_cams, C, H, W)
        proprio_tensor = torch.from_numpy(proprio)

        metadata = {
            "prompt": self.task_description,
            "video": video_tensor,
            "proprio": proprio_tensor,
        }

        return actions_t, metadata


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
):
    """Create a DataLoader and Dataset for RoboTwin training.

    Returns:
        (dataloader, dataset) tuple.
    """
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
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate_fn,
        persistent_workers=num_workers > 0,
    )

    return dataloader, dataset

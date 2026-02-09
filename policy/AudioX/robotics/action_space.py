"""Action space normalization / denormalization utilities."""

import torch


def normalize_actions(actions, action_stats):
    """Normalize actions to [-1, 1] using precomputed statistics.

    Args:
        actions: (*, action_dim) tensor of raw joint values.
        action_stats: dict with either {action_min, action_max} or
                      {action_mean, action_std}.

    Returns:
        Normalized actions in [-1, 1].
    """
    if "action_min" in action_stats and "action_max" in action_stats:
        a_min = action_stats["action_min"].to(actions.device)
        a_max = action_stats["action_max"].to(actions.device)
        actions = 2.0 * (actions - a_min) / (a_max - a_min + 1e-8) - 1.0
    elif "action_mean" in action_stats and "action_std" in action_stats:
        a_mean = action_stats["action_mean"].to(actions.device)
        a_std = action_stats["action_std"].to(actions.device)
        actions = (actions - a_mean) / (a_std + 1e-8)
    return actions


def denormalize_actions(actions, action_stats):
    """Denormalize actions from [-1, 1] back to real joint angle space.

    Args:
        actions: (*, action_dim) tensor in normalized space.
        action_stats: dict with either {action_min, action_max} or
                      {action_mean, action_std}.

    Returns:
        Denormalized actions.
    """
    if "action_min" in action_stats and "action_max" in action_stats:
        a_min = action_stats["action_min"].to(actions.device)
        a_max = action_stats["action_max"].to(actions.device)
        actions = (actions + 1.0) / 2.0 * (a_max - a_min) + a_min
    elif "action_mean" in action_stats and "action_std" in action_stats:
        a_mean = action_stats["action_mean"].to(actions.device)
        a_std = action_stats["action_std"].to(actions.device)
        actions = actions * a_std + a_mean
    return actions


def compute_action_stats(all_actions):
    """Compute min/max normalization statistics from a list of action arrays.

    Args:
        all_actions: list of numpy arrays, each (T_i, action_dim).

    Returns:
        dict with action_min, action_max, action_mean, action_std tensors.
    """
    import numpy as np
    concat = np.concatenate(all_actions, axis=0)  # (N, action_dim)
    return {
        "action_min": torch.tensor(concat.min(axis=0), dtype=torch.float32),
        "action_max": torch.tensor(concat.max(axis=0), dtype=torch.float32),
        "action_mean": torch.tensor(concat.mean(axis=0), dtype=torch.float32),
        "action_std": torch.tensor(concat.std(axis=0), dtype=torch.float32),
    }

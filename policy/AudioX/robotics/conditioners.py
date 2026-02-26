"""Robot-specific conditioners for AudioX.

Defines conditioner types not present in the official AudioX repo
(e.g. TrajectoryConditioner for robot proprioceptive state) and
patches the AudioX conditioner factory so they can be created from
JSON configs just like the built-in types.
"""

import typing as tp

import torch
from torch import nn

from audiox.models.conditioners import (
    Conditioner,
    MultiConditioner,
    create_multi_conditioner_from_conditioning_config as _original_factory,
)


class TrajectoryConditioner(Conditioner):
    """Conditioner for robot proprioceptive state (joint angles / gripper).

    Converts a fixed-size state vector into a single conditioning token
    that can be consumed by the DiT via cross-attention.

    Modes:
        "mlp":    state → Linear → SiLU → Linear → proj_out   (richer)
        "direct": state → Linear → proj_out                    (simpler)
    """

    def __init__(
        self,
        output_dim: int,
        state_dim: int = 14,
        mode: str = "mlp",
        project_out: bool = False,
    ):
        hidden_dim = output_dim
        super().__init__(dim=hidden_dim, output_dim=output_dim, project_out=project_out)

        if mode == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        elif mode == "direct":
            self.encoder = nn.Linear(state_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown TrajectoryConditioner mode: {mode}")

    def forward(
        self,
        states: tp.List[tp.Any],
        device: tp.Union[torch.device, str] = "cuda",
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            states: List of state vectors (numpy arrays or tensors), one per batch item.
            device: Target device.

        Returns:
            (features, mask) — features: (B, 1, output_dim), mask: (B, 1).
        """
        batch = []
        for s in states:
            if isinstance(s, torch.Tensor):
                batch.append(s.float())
            else:
                batch.append(torch.tensor(s, dtype=torch.float32))

        state_tensor = torch.stack(batch).to(device)  # (B, state_dim)
        features = self.encoder(state_tensor)           # (B, hidden_dim)
        features = self.proj_out(features)              # (B, output_dim)
        features = features.unsqueeze(1)                # (B, 1, output_dim)
        mask = torch.ones(features.shape[0], 1, device=device)
        return features, mask


# ---------------------------------------------------------------------------
# Patched factory that adds robot-specific conditioner types
# ---------------------------------------------------------------------------

def create_robot_conditioner_from_config(config: tp.Dict) -> MultiConditioner:
    """Extended version of AudioX's conditioner factory.

    Supports all official types (t5, clip, clap_text, mel_spec, …)
    plus robot-specific ones:
        - "trajectory": TrajectoryConditioner for robot joint state
    """
    import copy
    patched_config = copy.deepcopy(config)
    cond_dim = patched_config["cond_dim"]

    # Split configs into official and robot-specific
    official_configs = []
    robot_conditioners = {}

    for cond_info in patched_config.get("configs", []):
        cond_type = cond_info["type"]

        if cond_type == "trajectory":
            cond_config = {"output_dim": cond_dim}
            cond_config.update(cond_info["config"])
            robot_conditioners[cond_info["id"]] = TrajectoryConditioner(**cond_config)
        else:
            official_configs.append(cond_info)

    # Build official conditioners via AudioX factory
    if official_configs:
        patched_config["configs"] = official_configs
        multi_cond = _original_factory(patched_config)
    else:
        multi_cond = MultiConditioner({})

    # Inject robot-specific conditioners
    for cond_id, cond_module in robot_conditioners.items():
        multi_cond.conditioners[cond_id] = cond_module

    return multi_cond

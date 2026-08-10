"""Collection of policy rollouts for offline distillation."""

from __future__ import annotations

import torch
from loguru import logger
from tensordict import TensorDict

from cfg.CFG import DISTILLATION_DIR, get_map_name


class DistillationRecorder:
    """Buffers the policy inputs/outputs in RAM and writes them to ``distillation/<map_name>/`` as float32 tensors."""

    def __init__(self, obs_groups: list[str], max_gb: float, run_info: str):
        self.obs_groups = obs_groups
        self.max_bytes = int(max_gb * 1024**3)
        self.run_info = run_info
        self.out_dir = DISTILLATION_DIR / get_map_name()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.buffers: dict[str, list[torch.Tensor]] = {}
        self.num_bytes = 0

    @property
    def is_full(self) -> bool:
        """Whether the collected data has reached the configured size limit."""
        return self.num_bytes >= self.max_bytes

    def record(self, obs: TensorDict, action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Append one step, each tensor of shape (num_envs, dim), moved to CPU float32."""
        step = {group: obs[group] for group in self.obs_groups}
        step["action"] = action
        step["action_mean"] = mean
        step["action_std"] = std

        for key, tensor in step.items():
            tensor = tensor.detach().to(device="cpu", dtype=torch.float32)
            self.buffers.setdefault(key, []).append(tensor)
            self.num_bytes += tensor.numel() * 4

    def save(self) -> None:
        """Concatenate the buffered steps into (num_steps * num_envs, dim) tensors and write them to disk."""
        if not self.buffers:
            return

        path = self.out_dir / f"{self.run_info}.pt"
        torch.save({key: torch.cat(tensors) for key, tensors in self.buffers.items()}, path)
        logger.info(f"Distillation dataset written to {path} ({self.num_bytes / 1024**3:.2f} GB)")

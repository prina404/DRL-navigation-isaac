"""Collection of policy rollouts for offline distillation."""

from __future__ import annotations

import torch
from loguru import logger
from tensordict import TensorDict

from cfg.CFG import DISTILLATION_DIR, get_map_name

ENV_ID_BITS = 8
"""Width of the env-id field of an episode id; the run counter occupies the remaining 24 bits."""

MAX_ENVS = 1 << ENV_ID_BITS


class DistillationRecorder:
    """Buffers the policy inputs/outputs in RAM and writes them to ``distillation/<map_name>/`` as float32 tensors.

    Every recorded row also carries the id of the episode it came from, packed as ``(run_counter << 8) | env_id``,
    and the dataset ships a per-episode collision flag next to it. Together they let a training script drop the
    rollouts that bumped into something without having to re-run the evaluation. See ``README.md``.

    Whether the recorded actions were sampled from the policy distribution or are its mean is part of the file name
    and of the saved dict, so stochastic and deterministic datasets of the same map never overwrite each other.
    """

    def __init__(self, obs_groups: list[str], max_gb: float, run_info: str, num_envs: int, stochastic: bool):
        if num_envs > MAX_ENVS:
            raise ValueError(
                f"episode ids reserve {ENV_ID_BITS} bits for the env id, which caps the recorder at {MAX_ENVS} envs"
            )

        self.obs_groups = obs_groups
        self.max_bytes = int(max_gb * 1024**3)
        self.run_info = run_info
        self.num_envs = num_envs
        self.stochastic = stochastic
        self.out_dir = DISTILLATION_DIR / get_map_name()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.buffers: dict[str, list[torch.Tensor]] = {}
        self.num_bytes = 0

        self._env_ids = torch.arange(num_envs, dtype=torch.int64)
        self._run_counter = torch.zeros(num_envs, dtype=torch.int64)
        self._collided = torch.zeros(num_envs, dtype=torch.bool)
        self._episode_ids: list[torch.Tensor] = []  # one (num_envs,) entry per recorded step
        self._episode_collided: dict[int, bool] = {}  # episode id -> collided, filled in as episodes end

    @property
    def is_full(self) -> bool:
        """Whether the collected data has reached the configured size limit."""
        return self.num_bytes >= self.max_bytes

    @property
    def _current_episode_ids(self) -> torch.Tensor:
        """Id of the episode each env is currently running. Shape is (num_envs,)."""
        return (self._run_counter << ENV_ID_BITS) | self._env_ids

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

        # appended last so it stays row-aligned with the buffers above once everything is concatenated
        self._episode_ids.append(self._current_episode_ids.clone())
        self.num_bytes += self.num_envs * 8

    def mark_collisions(self, collided: torch.Tensor) -> None:
        """Flag the episodes currently being recorded as having collided. Shape is (num_envs,), sticky until reset."""
        self._collided |= collided.detach().cpu().bool()

    def end_episodes(self, env_ids: torch.Tensor) -> None:
        """Close the episodes of ``env_ids``: commit their collision flag and move those envs on to their next run."""
        env_ids = env_ids.detach().cpu().long()
        self._commit(env_ids)
        self._run_counter[env_ids] += 1
        self._collided[env_ids] = False

    def _commit(self, env_ids: torch.Tensor) -> None:
        """Write the collision verdict of the episodes ``env_ids`` are running into the per-episode table."""
        episode_ids = self._current_episode_ids[env_ids]
        for episode_id, collided in zip(episode_ids.tolist(), self._collided[env_ids].tolist()):
            self._episode_collided[episode_id] = collided

    def save(self) -> None:
        """Concatenate the buffered steps into (num_steps * num_envs, dim) tensors and write them to disk."""
        if not self.buffers:
            return

        # episodes still in flight when recording stopped have rows in the dataset too, so they need a verdict
        # as well: without one, filtering on the episode table would silently discard them.
        self._commit(self._env_ids)
        index = sorted(self._episode_collided)

        data: dict[str, torch.Tensor] = {key: torch.cat(tensors) for key, tensors in self.buffers.items()}
        data["episode_id"] = torch.cat(self._episode_ids)
        data["episode_index"] = torch.tensor(index, dtype=torch.int64)
        data["episode_collided"] = torch.tensor([self._episode_collided[key] for key in index], dtype=torch.bool)
        data["stochastic"] = torch.tensor(self.stochastic, dtype=torch.bool)

        policy_mode = "stochastic" if self.stochastic else "deterministic"
        path = self.out_dir / f"{self.run_info}_{policy_mode}.pt"
        torch.save(data, path)
        num_collided = int(data["episode_collided"].sum())
        logger.info(
            f"Distillation dataset written to {path} ({self.num_bytes / 1024**3:.2f} GB, "
            f"{len(index)} episodes, {num_collided} of them with a collision)"
        )

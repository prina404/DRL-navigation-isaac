"""Loader for the sim2real Go2 locomotion policy (``unitree_go2_locomotion`` repository).

Unlike the stock policy handled by :mod:`mdp.actions.go2_locomotion_policy`, this checkpoint is a **TorchScript export**
(``torch.jit``) that already contains its own ``EmpiricalNormalization`` layer. No ``rsl_rl`` model has to be rebuilt and
no legacy checkpoint conversion is needed: the module maps a raw ``(N, 360)`` observation straight to a deterministic
``(N, 12)`` joint action.
"""

from pathlib import Path

import torch

from cfg.CFG import CHECKPOINT_DIR, DEVICE

#: Ordered per-frame observation terms of the policy group, as configured by ``UnitreeGo2FlatEnvCfg_adapted``.
#: ``base_lin_vel`` and ``height_scan`` of the stock IsaacLab velocity task are disabled there, which is why the frame
#: is 45 wide instead of the stock 48.
OBS_TERM_DIMS: tuple[tuple[str, int], ...] = (
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("velocity_commands", 3),
    ("joint_pos", 12),
    ("joint_vel", 12),
    ("actions", 12),
)

GO2_SIM2REAL_CFG = {
    "checkpoint": CHECKPOINT_DIR / "unitree_go2_locomotion" / "policy_finetune10.5.5.49.pt",
    "device": DEVICE,
    # ObservationGroupCfg.history_length of the training env
    "history_length": 8,
    "frame_dim": sum(dim for _, dim in OBS_TERM_DIMS),  # 45
    "obs_dim": 8 * sum(dim for _, dim in OBS_TERM_DIMS),  # 360
    "n_actions": 12,
    # UnitreeGo2RoughEnvCfg: self.actions.joint_pos.scale, with use_default_offset=True
    "action_scale": 0.25,
    # sim.dt (0.005) * decimation (4) of the locomotion training env
    "control_dt": 0.02,
    # final stage of `velocity_command_curriculum` for (lin_vel_x, lin_vel_y) and the ang_vel_z command range
    "command_limits": (2.5, 2.5, 1.5),
    "hidden_dims": [512, 512, 256],
    "activation": "elu",
}


def get_locomotion_policy(checkpoint: str | Path | None = None, device: str | None = None) -> torch.jit.ScriptModule:
    """Load the TorchScript locomotion policy and put it in eval mode.

    The observation normalizer is baked into the module and its running statistics are frozen, so the returned module is
    a pure function of the observation.
    """
    ckpt_path = Path(checkpoint) if checkpoint is not None else Path(GO2_SIM2REAL_CFG["checkpoint"])
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Locomotion checkpoint not found: {ckpt_path}")

    policy = torch.jit.load(str(ckpt_path), map_location=device or GO2_SIM2REAL_CFG["device"])
    policy.eval()

    # Fail fast on a checkpoint whose observation layout does not match what the action term builds.
    obs_dim = dict(policy.named_buffers()).get("obs_normalizer._mean")
    if obs_dim is not None and obs_dim.shape[-1] != GO2_SIM2REAL_CFG["obs_dim"]:
        raise ValueError(
            f"Checkpoint {ckpt_path.name} expects an observation of {obs_dim.shape[-1]} dims, "
            f"but this action term builds {GO2_SIM2REAL_CFG['obs_dim']} "
            f"({GO2_SIM2REAL_CFG['history_length']} x {GO2_SIM2REAL_CFG['frame_dim']})."
        )

    return policy

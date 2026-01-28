import isaaclab.envs.mdp as mdp
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply


def cache_goal_on_reset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_name: str = "unitree_go2",
    goal_dist_m: float = 1.0,
) -> None:
    """Compute goal once on reset and store in env._goal_pos[env_ids]."""
    if not hasattr(env, "_goal_pos") or env._goal_pos is None:
        env._goal_pos = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)

    robot = env.scene[asset_name]
    default_root_state = robot.data.default_root_state  # (N, 13)
    pos0 = default_root_state[:, 0:3]
    quat0 = default_root_state[:, 3:7]

    # forward (+X in base frame)
    fwd_w = quat_apply(
        quat0,
        torch.tensor([1.0, 0.0, 0.0], device=env.device, dtype=torch.float32).repeat(env.num_envs, 1),
    )
    goal_w = pos0 + goal_dist_m * fwd_w

    env._goal_pos[env_ids] = goal_w[env_ids]
    # logger.info(f"Cached goal positions for env_ids={env_ids.tolist()}: {env._goal_pos[env_ids]}")


def dist_to_goal_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["unitree_go2"]
    pos_w = robot.data.root_pos_w
    goal_w = env._goal_pos
    delta_xy = goal_w[:, 0:2] - pos_w[:, 0:2]
    return torch.linalg.norm(delta_xy, dim=-1)


def is_goal_reached(env: ManagerBasedRLEnv, threshold_m: float = 0.2) -> torch.Tensor:
    dist_to_goal = dist_to_goal_xy(env)
    return dist_to_goal < threshold_m


def reward_distance_to_goal(env: ManagerBasedRLEnv) -> torch.Tensor:
    return 1 - dist_to_goal_xy(env)  # Reward is higher when closer to goal


def penalty_still(env: ManagerBasedRLEnv, speed_thresh: float = 0.05, penalty: float = -0.2) -> torch.Tensor:
    v = mdp.base_lin_vel(env, asset_cfg=SceneEntityCfg(name="unitree_go2"))
    speed_xy = torch.linalg.norm(v[:, 0:2], dim=-1)
    return torch.where(speed_xy < speed_thresh, torch.full_like(speed_xy, penalty), torch.zeros_like(speed_xy))


def penalty_collision_base(env: ManagerBasedRLEnv, force_thresh: float = 3.0, penalty: float = -5.0) -> torch.Tensor:
    sensor = env.scene["body_collision_sensor"]
    forces = sensor.data.net_forces_w  # (N, bodies, 3)
    mag = torch.linalg.norm(forces, dim=-1).max(dim=-1).values
    return torch.where(mag > force_thresh, torch.full_like(mag, penalty), torch.zeros_like(mag))

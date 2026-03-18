import isaaclab.envs.mdp as mdp
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.managers import SceneEntityCfg
from torch import Tensor

import lab.helpers.termination_helpers as termination
from lab.helpers.observation_helpers import get_lidar, get_path_obs
from lab.MyEnv import MyEnv


def dist_to_goal_xy(env: MyEnv) -> Tensor:
    robot: Articulation = env.scene["robot"]
    pos_w = robot.data.root_com_pos_w
    goal_local = env.path_manager.goal_pos_local
    goal_w = goal_local + env.scene.env_origins[:, :2]
    delta_xy = goal_w[:, 0:2] - pos_w[:, 0:2]
    return torch.linalg.norm(delta_xy, dim=-1)


def is_goal_reached(env: MyEnv, threshold_m: float = 0.2) -> Tensor:
    dist_to_goal = dist_to_goal_xy(env)
    path_len = env.path_manager.current_path_length
    result = (dist_to_goal < threshold_m) | (path_len < threshold_m)
    if env.long_horizon is False:
        result = result | termination.distance_traveled_termination(env)
    return result


def reward_distance_to_goal(env: MyEnv) -> Tensor:
    # find the closest point on the path and compute distance from that point to goal.
    env: MyEnv = env.unwrapped
    res = torch.zeros((env.num_envs), device=env.device)
    for id in range(env.num_envs):
        path_points = env.path_manager.path_tensors[id]  # (num_path_points, 2)
        remaining_points = path_points.size(dim=0)
        if remaining_points <= 1:
            res[id] = 0.0
        # note that I'm measuring the number of remaining points on the subsampled path
        # so the distance is approximately remaining_path * point_dist.
        remaining_distance = remaining_points * env.path_manager.point_dist
        normalized_distance = remaining_distance / env.path_manager.initial_path_length[id]

        res[id] = -normalized_distance
    return res


def penalty_still(env: MyEnv, speed_thresh: float = 0.05, penalty: float = -0.2) -> Tensor:
    v = mdp.base_lin_vel(env, asset_cfg=SceneEntityCfg(name="robot"))
    speed_xy = torch.linalg.norm(v[:, 0:2], dim=-1)
    return torch.where(
        speed_xy < speed_thresh,
        torch.full_like(speed_xy, penalty),
        torch.zeros_like(speed_xy),
    )


def detect_collision(env: MyEnv) -> Tensor:
    base_sensor = env.scene["body_collision_sensor"]
    base_forces = base_sensor.data.net_forces_w  # (N, bodies, 3)
    base_magnitude = torch.linalg.norm(base_forces, dim=-1).max(dim=-1).values

    hip_sensor = env.scene["hip_collision_sensor"]
    hip_forces = hip_sensor.data.net_forces_w  # (N, bodies, 3)
    hip_magnitude = torch.linalg.norm(hip_forces, dim=-1).max(dim=-1).values

    magnitude = torch.max(base_magnitude, hip_magnitude)
    return magnitude


def penalty_collision(env: MyEnv, force_thresh: float = 3.0, penalty: float = -5.0) -> Tensor:
    # boolean tensor indicating whether a collision has occurred
    collision_force = detect_collision(env)
    collisions = (collision_force > force_thresh).to(torch.float32)

    strong_collision_thresh = 50.0  # if the robot experiences very strong collisions, we want to penalize in a proportional way
    strong_collisions = collision_force > strong_collision_thresh

    additional_penalty = collision_force[strong_collisions] - strong_collision_thresh
    additional_penalty = torch.clamp(additional_penalty, min=0.0, max=50.0) * 0.5  # cap & scale the additional penalty

    penalty = collisions * penalty
    penalty[strong_collisions] -= additional_penalty
    return penalty


def penalty_obstacle_proximity(env: MyEnv, max_penalty: float = -2.0) -> Tensor:
    lidar_ranges = get_lidar(env, num_obstacles=10, normalize=False)  # (B, num_obstacles*2)
    min_dist = lidar_ranges[:, :10].min(dim=-1).values  # (B,)
    return max_penalty + torch.clamp(torch.pow(min_dist, 2), min=max_penalty, max=-max_penalty)  # ()


def robot_heading_reward(env: MyEnv) -> Tensor:
    # Returns 1 if robot is heading in the correct direction, 0 otherwise
    # We compute the alignment wrt to the direction of the last point of the path in hour horizon.
    obs_tensor = get_path_obs(env, num_points_forward=10, normalize=True)  # (B, 20)
    headings = obs_tensor[:, 10:]  # (B, 10)
    next_heading = headings[:, 3]  # I extract the heading of 3rd point forward (approx 90cm ahead)
    # if next_heading == 0.5 -> robot is facing the correct direction. Max reward.
    heading_delta = torch.abs(0.5 - next_heading)
    # from loguru import logger
    # logger.debug(f"Heading reward: {1.0 - heading_delta[0].cpu().item() * 2.:.2f}")
    return 1.0 - heading_delta * 2.0  # normalize to [0, 1], 1 = perfect alignment, 0 = opposite direction


def action_smoothness_penalty(env: MyEnv) -> Tensor:
    # I assume that jerky motions are correlated to sudden changes in velocity.
    delta = env._action_buffer[:, 0] - env._action_buffer[:, 1]  # (B, action_dim)
    return -torch.linalg.norm(delta, dim=-1)


def time_penalty(env: MyEnv) -> Tensor:
    # Small penalty at each timestep to encourage faster completion
    return -1.0


def collision_after_threshold_reward(env: MyEnv, penalty: float = -20.0) -> Tensor:
    return torch.where(
        termination.collision_after_threshold_termination(env),
        torch.full((env.num_envs,), penalty, device=env.device),
        torch.zeros((env.num_envs,), device=env.device),
    )

from loguru import logger
from isaaclab.assets.articulation import Articulation
import isaaclab.envs.mdp as mdp
from lab.MyEnv import MyEnv
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply
from lab.observation_helpers import get_path_obs, get_lidar


def dist_to_goal_xy(env: MyEnv) -> torch.Tensor:
    robot: Articulation = env.scene["unitree_go2"]
    pos_w = robot.data.root_com_pos_w
    goal_local = env.goal_pos
    goal_w = goal_local + env.scene.env_origins[:, :2]
    delta_xy = goal_w[:, 0:2] - pos_w[:, 0:2]
    return torch.linalg.norm(delta_xy, dim=-1)


def is_goal_reached(env: MyEnv, threshold_m: float = 0.2) -> torch.Tensor:
    dist_to_goal = dist_to_goal_xy(env)
    return dist_to_goal < threshold_m


def reward_distance_to_goal(env: MyEnv) -> torch.Tensor:
    return 1 - dist_to_goal_xy(env)  # Reward is higher when closer to goal


def penalty_still(env: MyEnv, speed_thresh: float = 0.05, penalty: float = -0.2) -> torch.Tensor:
    v = mdp.base_lin_vel(env, asset_cfg=SceneEntityCfg(name="unitree_go2"))
    speed_xy = torch.linalg.norm(v[:, 0:2], dim=-1)
    return torch.where(speed_xy < speed_thresh, torch.full_like(speed_xy, penalty), torch.zeros_like(speed_xy))


def penalty_collision_base(env: MyEnv, force_thresh: float = 3.0, penalty: float = -5.0) -> torch.Tensor:
    sensor = env.scene["body_collision_sensor"]
    forces = sensor.data.net_forces_w  # (N, bodies, 3)
    mag = torch.linalg.norm(forces, dim=-1).max(dim=-1).values
    return torch.where(mag > force_thresh, torch.full_like(mag, penalty), torch.zeros_like(mag))

def penalty_obstacle_proximity(env: MyEnv, max_penalty: float = -2.0) -> torch.Tensor:
    lidar_ranges = get_lidar(env, num_obstacles=10, normalize=False)  # (B, num_obstacles*2)
    min_dist = lidar_ranges[:, :10].min(dim=-1).values  # (B,)
    return max_penalty + torch.clamp(torch.pow(min_dist, 2), min = max_penalty, max=-max_penalty) # ()

def robot_heading_reward(env: MyEnv) -> torch.Tensor:
    # Returns 1 if robot is heading in the correct direction, 0 otherwise
    # We compute the alignment wrt to the direction of the last point of the path in hour horizon.
    obs_tensor = get_path_obs(env, num_points_forward=10, normalize=True)  # (B, 20)
    dist, headings = obs_tensor[:, :10], obs_tensor[:, 10:]  # (B, 10), (B, 10)
    next_heading = headings[:, 2] # I extract the heading of 3rd point forward (approx 1m ahead)
    # if next_heading == 0.5 -> robot is facing the correct direction. Max reward.
    heading_delta = torch.abs(0.5 - next_heading)
    return 1.0 - heading_delta * 2.0  # normalize to [0, 1]

def action_smoothness_penalty(env: MyEnv, alpha: float = 0.2) -> torch.Tensor:
    # I assume that jerky motions are correlated to sudden changes in velocity.
    delta = env._action_buffer[:, 0] - env._action_buffer[:, 1]  # (B, action_dim)
    return -alpha * torch.linalg.norm(delta, dim=-1)
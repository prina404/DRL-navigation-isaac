import torch

import lab.helpers.reward_helpers as rewards
from lab.MyEnv import MyEnv


def collision_after_threshold_termination(env: MyEnv) -> torch.Tensor:
    if env.collision_termination_thresh >= 1.0:
        return torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)

    traveled_dist = env.path_manager.initial_path_length - env.path_manager.current_path_length
    max_nav_distance = torch.min(torch.full((env.num_envs,), 5.0, device=env.device), env.path_manager.initial_path_length)
    nav_progress = traveled_dist / max_nav_distance

    collisions = rewards.detect_collision(env, force_thresh=3.0)
    over_threshold = nav_progress > env.collision_termination_thresh

    return collisions & over_threshold


def distance_traveled_termination(env: MyEnv, max_distance: float = 5.0) -> torch.Tensor:
    traveled_dist = env.path_manager.initial_path_length - env.path_manager.current_path_length
    return traveled_dist >= max_distance

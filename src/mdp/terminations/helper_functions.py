import torch

import mdp.rewards.helper_functions as rewards
from navigation_env.NavigationEnv import NavEnv


def collision_after_threshold_termination(env: NavEnv) -> torch.Tensor:
    if env.collision_termination_thresh >= 1.0:
        return torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)

    if env.collision_termination_thresh >= 1.0:
        return torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)

    traveled_dist = env.path_manager.initial_path_length - env.path_manager.current_path_length
    max_nav_distance = torch.min(torch.full((env.num_envs,), 5.0, device=env.device), env.path_manager.initial_path_length)
    nav_progress = traveled_dist / max_nav_distance

    collisions = rewards.detect_collision(env)
    over_threshold = nav_progress > env.collision_termination_thresh

    return collisions & over_threshold


def distance_traveled_termination(env: NavEnv, max_distance: float = 5.0) -> torch.Tensor:
    traveled_dist = env.path_manager.initial_path_length - env.path_manager.current_path_length
    return traveled_dist >= max_distance

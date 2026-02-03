import torch
from isaaclab.envs import ManagerBasedRLEnv
from lab.MyEnv import MyEnv
from loguru import logger

def teleport_on_reset(env: MyEnv, env_ids: torch.Tensor) -> None:
    # on reset, sample new start positions, 
    n_samples = env_ids.shape[0]
    sampled_node_id = env.sample_node_ids(n_samples)  # (num_envs_to_reset, 1)
    env.teleport_robots(env_ids, sampled_node_id)
    env.compute_goals_on_reset(env_ids)

    logger.debug(f"Teleported {n_samples} robots on reset."
                f"{env.start_pos_ids=}"
                f"{env.goal_ids=}"
            )
    
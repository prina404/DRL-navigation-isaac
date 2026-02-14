from isaaclab.assets.articulation.articulation import Articulation
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
    env.place_mid_obstacle_on_reset(env_ids)

    # Reset joints to default state after teleporting root poses.
    robot: Articulation = env.scene["unitree_go2"]
    default_pos = robot.data.default_joint_pos[env_ids]
    default_vel = robot.data.default_joint_vel[env_ids]
    robot.set_joint_position_target(default_pos, None, env_ids)
    robot.write_joint_state_to_sim(default_pos, default_vel, None, env_ids)
    #logger.debug(f"robot joint state: {robot.data.joint_pos, robot.data.joint_vel} -> [{robot.data.default_joint_pos},{robot.data.default_joint_vel}]")
    # logger.debug(f"Teleported {n_samples} robots on reset."
    #             f"{env.start_pos_ids=}"
    #             f"{env.goal_ids=}"
    #         )
    
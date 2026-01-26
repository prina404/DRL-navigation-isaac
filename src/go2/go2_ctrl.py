import os
from typing import Callable
import omni.usd
import torch
import gymnasium as gym
from isaaclab.envs import ManagerBasedEnv
from go2.go2_ctrl_cfg import unitree_go2_flat_cfg, unitree_go2_rough_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic
from loguru import logger
from cfg.CFG import CHECKPOINT_DIR
base_vel_cmd_input = None



def init_base_vel_cmd(num_envs, device: str | torch.device = "cpu"):
    global base_vel_cmd_input
    base_vel_cmd_input = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)


def _ensure_cmd_tensor(env: ManagerBasedEnv) -> None:
    """Make sure base_vel_cmd_input exists, has correct shape, and is on env.device."""
    global base_vel_cmd_input
    num_envs = getattr(env, "num_envs", None)
    if num_envs is None:
        num_envs = base_vel_cmd_input.shape[0] if base_vel_cmd_input is not None else 1

    if (
        base_vel_cmd_input is None
        or base_vel_cmd_input.shape != (num_envs, 3)
        or base_vel_cmd_input.device != env.device
    ):
        print("Initializing base_vel_cmd_input tensor, shape:", (num_envs), "device:", env.device,  "cmd:", base_vel_cmd_input )
        init_base_vel_cmd(num_envs, device=env.device)


def base_vel_cmd(env: ManagerBasedEnv) -> torch.Tensor:
    global base_vel_cmd_input, _rand_cmd_step_counter
    _ensure_cmd_tensor(env)
    return base_vel_cmd_input


def get_mpc_policy():
    cfg = unitree_go2_flat_cfg['policy']
    ckpt_path = get_checkpoint_path(log_path=str(CHECKPOINT_DIR), 
                                run_dir=unitree_go2_flat_cfg["load_run"], 
                                checkpoint=unitree_go2_flat_cfg["load_checkpoint"])

    # policy = ActorCritic( # TODO: add IsaacLab version check because ActorCritic API changed
    #     cfg.pop('actor_input_dims'),
    #     cfg.pop('critic_input_dims'),
    #     cfg.pop('n_actions'),
    #     **cfg
    # )
    obs_dict = {
        "default": torch.zeros((1, cfg.pop('actor_input_dims'))),
    }
    cfg.pop('critic_input_dims') # discard unusable variable 
    obs_groups = {
        "policy": ["default"],
        "critic": ["default"],
    }
    policy = ActorCritic(
        obs_dict,
        obs_groups,
        cfg.pop('n_actions'),
        **cfg
    )


    state_dict = torch.load(ckpt_path, weights_only=False)
    policy.load_state_dict(state_dict['model_state_dict'])

    policy = policy.to(unitree_go2_flat_cfg['device'])
    return policy.act_inference # return only the actor network
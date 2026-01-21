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
from loguru import logger
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

def get_rsl_flat_policy(cfg) -> Callable:
    logger.info("Loading MPC policy for Go2")
    cfg.observations.policy.height_scan = None
    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env)

    # Low level control: rsl control policy
    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_flat_cfg
    ckpt_path = get_checkpoint_path(log_path=os.path.abspath("ckpts"), 
                                    run_dir=agent_cfg["load_run"], 
                                    checkpoint=agent_cfg["load_checkpoint"])
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
    ppo_runner.load(ckpt_path)
    policy = ppo_runner.get_inference_policy(device=agent_cfg["device"])
    env.reset()
    env.close() # close dummy env needed to load rsl_policy
    del env
    omni.usd.get_context().new_stage()
    logger.info("MPC policy loaded successfully")
    return policy

def get_rsl_rough_policy(cfg):
    env = gym.make("Isaac-Velocity-Rough-Unitree-Go2-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env)

    # Low level control: rsl control policy
    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_rough_cfg
    ckpt_path = get_checkpoint_path(log_path=os.path.abspath("ckpts"), 
                                    run_dir=agent_cfg["load_run"], 
                                    checkpoint=agent_cfg["load_checkpoint"])
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
    ppo_runner.load(ckpt_path)
    policy = ppo_runner.get_inference_policy(device=agent_cfg["device"])
    return env, policy
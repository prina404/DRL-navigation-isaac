import os
import hydra
from omegaconf import DictConfig
import torch
import time
import math
from isaaclab.app import AppLauncher

# # add argparse arguments
# parser = argparse.ArgumentParser(description="Tutorial on basic RL environment.")

# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# args_cli = parser.parse_args()
app_launcher = AppLauncher(enable_cameras=True)  # args_cli)
simulation_app = app_launcher.app

import torch

from go2.go2_env import Go2MPCEnvCfg, camera_follow
from lab.managed_env import Go2EnvCfg
import go2.go2_sensors as go2_sensors

import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRLVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
import go2.go2_ctrl as go2_ctrl
from lab.scene_loaders import load_interiorAgent_env
from rsl_rl.runners import OnPolicyRunner
from lab.encoder_model import ViTEncoder

from lab.go2_nav_cfg import go2_policy_cfg


FILE_PATH = os.path.join(os.path.dirname(__file__), "cfg")


@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
def run_simulator(cfg: DictConfig):

    # Go2 MPC setup
    go2_mpc_cfg = Go2MPCEnvCfg()
    go2_mpc_cfg.scene.num_envs = cfg.num_envs
    mpc = go2_ctrl.get_rsl_flat_policy(go2_mpc_cfg)

    # Go2 Env setup
    go2_env_cfg = Go2EnvCfg()
    go2_env_cfg.actions.mpc_cmd.mpc_policy = mpc # inject mpc policy
    go2_env_cfg.scene.num_envs = cfg.num_envs
    go2_env_cfg.decimation = math.ceil(1.0 / go2_env_cfg.sim.dt / cfg.freq)
    go2_env_cfg.sim.render_interval = go2_env_cfg.decimation
    
    # Create the whole scene
    env = gym.make("Isaac-Go2-MPC-v0", cfg=go2_env_cfg)
    env = RslRLVecEnvWrapper(env)
    
    # Navigation Policy setup
    vit = ViTEncoder().to(go2_policy_cfg["device"])
    agent_cfg = go2_policy_cfg
    agent_cfg["num_envs"] = cfg.num_envs
    agent_cfg["policy"]["encoder"] = vit
    # ckpt_path = get_checkpoint_path(log_path=os.path.abspath("ckpts"), 
    #                                 run_dir=agent_cfg["load_run"], 
    #                                 checkpoint=agent_cfg["load_checkpoint"])
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
    #ppo_runner.load(ckpt_path)


    #load_interiorAgent_env(cfg, env.unwrapped.scene.env_ns)
    policy = ppo_runner.get_inference_policy(device=agent_cfg["device"])
    # Sensor setup
    sm = go2_sensors.SensorManager(cfg.num_envs)
    lidar_annotators = sm.add_rtx_lidar()

    # Run simulation
    sim_step_dt = float(go2_env_cfg.sim.dt * go2_env_cfg.decimation)
    obs, _ = env.reset()
    paused = False
    while simulation_app.is_running():
        start_time = time.time()

        if paused:
            time.sleep(0.01)
            continue
        with torch.no_grad():
            # control joints
            actions = policy(obs)
            # step the environment
            obs, _, _, _ = env.step(actions)

            if cfg.camera_follow:
                camera_follow(env)

        elapsed_time = time.time() - start_time
        if elapsed_time < sim_step_dt:
            sleep_duration = sim_step_dt - elapsed_time
            time.sleep(sleep_duration)


if __name__ == "__main__":
    run_simulator()

import argparse
import sys

import rclpy
import torch
import tqdm
from isaaclab.app import AppLauncher
import warp as wp

from cfg.CFG import SCENE_USD_PATH

# # add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on basic RL environment.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument(
    "--video_length",
    type=int,
    default=1000,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")

parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=100, help="RL Policy training iterations.")

AppLauncher.add_app_launcher_args(parser)

# # append AppLauncher cli args
args_cli, hydra_argv = parser.parse_known_args()
args_cli.enable_cameras = True  # always true for go2 cfg

args_cli.kit_args = (args_cli.kit_args or "") + " --enable isaacsim.sensors.rtx" + " --enable isaacsim.ros2.bridge"
sys.argv = [sys.argv[0]] + hydra_argv
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import os
import traceback
from datetime import datetime

import gymnasium as gym
import hydra
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from loguru import logger
from omegaconf import DictConfig
from ros2.RosDataManager import RobotDataManager

import go2.go2_mpc as go2_mpc
from go2.go2_nav_cfg import go2_policy_cfg
from lab.MyEnvCfg import Go2EnvCfg

# from isaacsim.core.utils import extensions
# extensions.enable_extension("isaacsim.ros2.bridge")
# simulation_app.update()

FILE_PATH = os.path.join(os.path.dirname(__file__), "src/cfg")


@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
def run_simulator(cfg: DictConfig):

    # Go2 MPC setup
    mpc = go2_mpc.get_mpc_policy()

    # Go2 Env setup
    go2_env_cfg = Go2EnvCfg()
    go2_env_cfg.curriculum = None
    go2_env_cfg.actions.mpc_cmd.mpc_policy = mpc  # inject mpc policy
    go2_env_cfg.scene.num_envs = cfg.num_envs if args_cli.num_envs is None else args_cli.num_envs
    go2_env_cfg.seed = args_cli.seed if args_cli.seed is not None else 42

    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", "validation"))
    logger.info(f"Logging experiment in directory: {log_root_path}")
    logger.info(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)

    if isinstance(go2_env_cfg, ManagerBasedRLEnvCfg):
        go2_env_cfg.export_io_descriptors = True

    go2_env_cfg.log_dir = log_dir

    # Create the whole scene
    logger.info("Creating gym environment...")
    gym.register(
        id="Isaac-indoor-navigation-go2-v0",
        entry_point="lab.MyEnv:MyEnv",
        disable_env_checker=True,
        kwargs={"scene_path": SCENE_USD_PATH, "use_long_horizon": True, "sample_voronoi": False},
    )
    env = gym.make(
        "Isaac-indoor-navigation-go2-v0",
        cfg=go2_env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    logger.info("Gym environment created")

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "validation"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        logger.info("Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env)
    logger.info("RslRlVecEnvWrapper applied to gym environment")

    obs, _ = env.reset()

    # Ros2 DataManager setup

    rclpy.init()
    __env = env.unwrapped
    ros2_dm = RobotDataManager(__env, __env.scene["lidar"], __env.scene["camera"], cfg)

    # --- Eval loop --- 
    num_envs = env.num_envs
    episodes_done = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    episode_collisions = torch.zeros(num_envs, dtype=torch.float32, device=env.device)
    termination_flags = []
    collisions = []  # per-completed-episode collision counts

    avg_episodes = 0.0
    with tqdm.tqdm(total=args_cli.max_iterations, desc="Evaluating policy") as pbar:
        while avg_episodes < args_cli.max_iterations:
            with torch.no_grad():
                action = ros2_dm.base_vel_cmd_input.to(env.device)
            _, _, dones, info = env.step(action)
            ros2_dm.pub_ros2_data()

            dones = dones.bool()
            time_outs = info.get("time_outs", torch.zeros_like(dones)).bool()

            # log collisions per env at each step
            sensor = env.unwrapped.scene["body_collision_sensor"]
            forces = wp.to_torch(sensor.data.net_forces_w)  # (N, bodies, 3)
            magnitude = torch.linalg.norm(forces, dim=-1).max(dim=-1).values
            collision_tensor = magnitude > 1.0
            episode_collisions += collision_tensor.float()

            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if len(done_ids) > 0:
                # finalize metrics for each completed env-episode
                collisions.extend(episode_collisions[done_ids].detach().cpu().tolist())
                termination_flags.extend((~time_outs[done_ids]).float().detach().cpu().tolist())

                episodes_done[done_ids] += 1
                episode_collisions[done_ids] = 0.0

            new_avg_episodes = episodes_done.float().mean().item()
            new_avg_episodes_clamped = min(new_avg_episodes, float(args_cli.max_iterations))
            pbar.update(new_avg_episodes_clamped - avg_episodes)
            avg_episodes = new_avg_episodes_clamped

    env.close()

    avg_termination_rate = sum(termination_flags) / len(termination_flags) if termination_flags else 0.0
    avg_collisions = sum(collisions) / len(collisions) if collisions else 0.0

    logger.info(f"Average termination rate over {args_cli.max_iterations} avg episodes/env: {avg_termination_rate:.3f}")
    logger.info(f"Average collisions per completed episode: {avg_collisions:.3f}")


if __name__ == "__main__":
    try:
        run_simulator()

    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()

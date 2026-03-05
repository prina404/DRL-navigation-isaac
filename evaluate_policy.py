import argparse
import sys

from isaaclab.app import AppLauncher

from cfg.CFG import SCENE_USD_PATH
import torch
import tqdm

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
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Name of the policy checkpoint file to evaluate.",
)
parser.add_argument("--max_iterations", type=int, default=100, help="RL Policy training iterations.")
parser.add_argument(
    "--debug-vis",
    action="store_true",
    default=False,
    help="Enable debug visualizations in the environment.",
)

AppLauncher.add_app_launcher_args(parser)

# # append AppLauncher cli args
args_cli, hydra_argv = parser.parse_known_args()
args_cli.enable_cameras = True  # always true for go2 cfg

args_cli.kit_args = (args_cli.kit_args or "") + " --enable isaacsim.sensors.rtx" 
sys.argv = [sys.argv[0]] + hydra_argv
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import os
import traceback
from datetime import datetime

import gymnasium as gym
import hydra
from dotenv import load_dotenv
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from loguru import logger
from omegaconf import DictConfig
from rsl_rl.runners import OnPolicyRunner

import go2.go2_mpc as go2_mpc
from go2.go2_nav_cfg import go2_policy_cfg
from lab.MyEnvCfg import Go2EnvCfg

FILE_PATH = os.path.join(os.path.dirname(__file__), "src/cfg")


@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
def run_simulator(cfg: DictConfig):

    # Go2 MPC setup
    mpc = go2_mpc.get_mpc_policy()

    # Go2 Env setup
    go2_env_cfg = Go2EnvCfg()
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
        entry_point="lab.MyEnv:MyEnv" if not args_cli.debug_vis else "lab.MyEnvDebuggingVis:MyEnvDebuggingVis",
        disable_env_checker=True,
        kwargs={"scene_path": SCENE_USD_PATH},
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

    # Navigation Policy setup
    agent_cfg = go2_policy_cfg

    agent_cfg["num_envs"] = cfg.num_envs
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=log_dir, device=agent_cfg["device"])

    if args_cli.checkpoint:
        ckpt_path = get_checkpoint_path(
            log_path=os.path.abspath("ckpts"),
            run_dir=agent_cfg["load_run"],
            checkpoint=args_cli.checkpoint,
        )
        ppo_runner.load(ckpt_path)

    policy = ppo_runner.get_inference_policy(env.device)


    termination_rates = []
    collisions = [] # track average num of collisions per episode
    
    for iteration in tqdm.tqdm(range(args_cli.max_iterations), desc="Evaluating policy"):
        obs, _ = env.reset()
        total_collisions = 0
        for step in range(go2_policy_cfg["num_steps_per_env"]):
            with torch.no_grad():
                action = policy(obs)
            obs, reward, terminated, info = env.step(action)
            if terminated.all():
                break

            # log collisions at each step
            sensor = env.unwrapped.scene["body_collision_sensor"]
            forces = sensor.data.net_forces_w  # (N, bodies, 3)
            magnitude = torch.linalg.norm(forces, dim=-1).max(dim=-1).values
            collision_tensor = magnitude > 1.0
            total_collisions += collision_tensor.sum().cpu()

        
        num_timeout = info["time_outs"].sum()
        num_env_terminated = terminated.sum() - num_timeout
        termination_rates.append(num_env_terminated / cfg.num_envs)

        collisions.append(total_collisions.item() / cfg.num_envs)
    env.close()

    logger.info(f"Average termination rate over {args_cli.max_iterations} episodes: {sum(termination_rates) / len(termination_rates):.3f}")
    logger.info(f"Average collisions per episode: {sum(collisions) / len(collisions):.3f}")


if __name__ == "__main__":
    try:
        run_simulator()

    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()

import argparse
import sys

from isaaclab.app import AppLauncher

# # add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on basic RL environment.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument(
    "--video_length",
    type=int,
    default=300,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=3000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")

parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--log_interval", type=int, default=100_000, help="Log data every n timesteps.")
parser.add_argument(
    "--checkpoint",
    action="store_true",
    default=False,
    help="Continue the training from checkpoint.",
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--wandb",
    action="store_true",
    default=False,
    help="Name of the WandB project for logging",
)
parser.add_argument(
    "--debug-vis",
    action="store_true",
    default=False,
    help="Enable debug visualizations in the environment.",
)

parser.add_argument("--task", type=str, default="go2_lidar_full", help="Name of the task configuration to use for training.")

AppLauncher.add_app_launcher_args(parser)

# # append AppLauncher cli args
args_cli, hydra_argv = parser.parse_known_args()
# args_cli.enable_cameras = True  # always true for go2 cfg

# args_cli.kit_args = (args_cli.kit_args or "") + " --enable isaacsim.sensors.rtx"
sys.argv = [sys.argv[0]] + hydra_argv
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import os
import traceback
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import hydra
from dotenv import load_dotenv
from hydra.utils import get_original_cwd
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from loguru import logger
from omegaconf import DictConfig
from rsl_rl.runners import OnPolicyRunner

from cfg.CFG import get_scene_usd_path, CHECKPOINT_DIR, get_map_name
from policy.NavPolicyv2 import go2_policy_cfg, load_policy_checkpoint, make_go2_policy_cfg
from tasks.task_utils import get_env_config


@hydra.main(config_path=None)
def run_simulator(cfg: DictConfig):
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join(get_original_cwd(), "logs", "rsl_rl", "training"))
    logger.info(f"Logging experiment in directory: {log_root_path}")
    logger.info(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)

    # Go2 Env setup
    environment_cfg = get_env_config(args_cli.task)
    environment_cfg.scene.num_envs = cfg.num_envs if args_cli.num_envs is None else args_cli.num_envs
    environment_cfg.seed = args_cli.seed if args_cli.seed is not None else 42
    environment_cfg.log_dir = log_dir

    # Create the whole scene
    logger.info("Creating gym environment...")
    gym.register(
        id="Isaac-indoor-navigation-go2-v0",
        entry_point="navigation_env.NavigationEnv:NavEnv"
        if not args_cli.debug_vis
        else "navigation_env.EnvDebugWrapper:NavEnvDebugView",
        disable_env_checker=True,
        kwargs={"scene_path": get_scene_usd_path(), "use_long_horizon": False, "debug_vis": args_cli.debug_vis, "sample_voronoi_probability": 0.25},
    )
    env = gym.make(
        "Isaac-indoor-navigation-go2-v0",
        cfg=environment_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    logger.info("Gym environment created")

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
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
    # Note: OnPolicyRunner consumes its configuration destructively, so a fresh copy is built here
    policy_cfg = make_go2_policy_cfg(environment_cfg.obs_groups)  # obs_groups needed for encoder initialization
    policy_cfg["num_envs"] = args_cli.num_envs if args_cli.num_envs is not None else cfg.num_envs

    if args_cli.wandb:
        load_dotenv()
        policy_cfg["logger"] = "wandb"
        policy_cfg["wandb_project"] = os.environ["WANDB_PROJECT"]

    ppo_runner = OnPolicyRunner(env, policy_cfg, log_dir=log_dir, device=policy_cfg["device"])

    if args_cli.checkpoint is True:
        ckpt_path = get_checkpoint_path(
            log_path=os.path.join(get_original_cwd(), "ckpts"),
            run_dir=policy_cfg["load_run"],
            checkpoint=policy_cfg["load_checkpoint"],
        )
        load_policy_checkpoint(ppo_runner, ckpt_path)

    ppo_runner.learn(
        num_learning_iterations=(
            go2_policy_cfg["max_iterations"] if args_cli.max_iterations is None else args_cli.max_iterations
        ),
    )
    store_final_policy(ppo_runner, CHECKPOINT_DIR)

def store_final_policy(ppo_runner: OnPolicyRunner, ckpt_dir: Path) -> None:
    # final path should be ckpts/unitree_go2_nav/single_env_policies/{map_name}_policy.pt
    policy_name = get_map_name() + "_policy.pt"
    policy_save_path = ckpt_dir / "unitree_go2_nav" / "single_env_policies" / policy_name
    policy_save_path.parent.mkdir(parents=True, exist_ok=True)
    if policy_save_path.exists():
        
        new_name = f"old_{get_map_name()}_policy_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"
        policy_save_path.rename(policy_save_path.parent / new_name)

    ppo_runner.save(policy_save_path)
    logger.debug(f"Final episode count: {ppo_runner.env.unwrapped.episode_counter.mean().item()}")
    ppo_runner.env.close()

if __name__ == "__main__":
    try:
        run_simulator()

    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()

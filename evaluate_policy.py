import argparse
import sys

from isaaclab.app import AppLauncher

from cfg.CFG import get_map_name, get_scene_usd_path, set_map_name

# # add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on basic RL environment.")
parser.add_argument(
    "--map",
    type=str,
    required=True,
    help="Map to run on, e.g. 40, 0040 or kujiale_0040. Overrides current_env in dataset_cfg.yaml.",
)
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
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Seed used for the environment. Passing it explicitly also reseeds every random stream right before "
    "the scored rollout, so two policies run at the same seed see the same episodes (with --num_envs=1 for the "
    "whole rollout, above it only for the first episode of each env).",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Name of the policy checkpoint file to evaluate.",
)
parser.add_argument(
    "--max_episodes",
    type=int,
    default=20,
    help="Number of episodes to record per environment.",
)
parser.add_argument(
    "--collision_force_thresh",
    type=float,
    default=3.0,
    help="Contact force (N) above which the robot is considered to be colliding.",
)
parser.add_argument(
    "--debug-vis",
    action="store_true",
    default=False,
    help="Enable debug visualizations in the environment.",
)
parser.add_argument("--task", type=str, default="go2_lidar_full", help="Name of the task configuration to use for training.")

parser.add_argument("--distillation", action="store_true", default=False, help="Store to disk the observations, actions, and policy mean+std for distillation.")
parser.add_argument(
    "--stochastic",
    action="store_true",
    default=False,
    help="Execute actions sampled from the policy distribution instead of its deterministic mean. Independent of "
    "--distillation: the mean and std are logged either way.",
)



AppLauncher.add_app_launcher_args(parser)

# # append AppLauncher cli args
args_cli, hydra_argv = parser.parse_known_args()
set_map_name(args_cli.map)  # before anything reads the map back out of cfg.CFG
# Only the vision/depth tasks and video recording need cameras. Enabling them otherwise boots the RTX rendering Kit
# experience, whose stage population cost scales with --num_envs.
args_cli.enable_cameras = args_cli.video or "vision" in args_cli.task or "depth" in args_cli.task

#args_cli.kit_args = (args_cli.kit_args or "") + " --enable isaacsim.sensors.rtx"
sys.argv = [sys.argv[0]] + hydra_argv
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import json
import os
import traceback
from datetime import datetime

import gymnasium as gym
import hydra
from hydra.utils import get_original_cwd
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from loguru import logger
from omegaconf import DictConfig
from rsl_rl.runners import OnPolicyRunner

from distillation.policy_eval import log_summary, rollout_policy, summarize
from distillation.recorder import DistillationRecorder
from policy.NavPolicyv2 import load_policy_checkpoint, make_go2_policy_cfg
from tasks.task_utils import get_env_config

FILE_PATH = os.path.join(os.path.dirname(__file__), "src/cfg")


@hydra.main(config_path=None)
def run_simulator(cfg: DictConfig):
    run_info = get_map_name() + datetime.now().strftime("_%m-%d_%H-%M")
    log_root_path = os.path.abspath(os.path.join(get_original_cwd(), "logs", "rsl_rl", "validation"))
    logger.info(f"Logging experiment in directory: {log_root_path}")
    logger.info(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)

    # Go2 Env setup
    environment_cfg = get_env_config(args_cli.task)
    environment_cfg.curriculum = None
    environment_cfg.scene.num_envs = args_cli.num_envs
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
            "video_folder": os.path.join(log_dir, "videos"),
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
    policy_cfg["num_envs"] = args_cli.num_envs

    ppo_runner = OnPolicyRunner(env, policy_cfg, log_dir=log_dir, device=policy_cfg["device"])

    if args_cli.checkpoint is not None:
        ckpt_path = get_checkpoint_path(
            log_path=os.path.join(get_original_cwd(), "ckpts"),
            run_dir=policy_cfg["load_run"],
            checkpoint=args_cli.checkpoint,
        )
        load_policy_checkpoint(ppo_runner, ckpt_path)

    policy = ppo_runner.get_inference_policy(env.device)

    recorder = (
        DistillationRecorder(policy.obs_groups, 10.0, run_info, env.num_envs, args_cli.stochastic)
        if args_cli.distillation
        else None
    )
    num_envs = env.num_envs

    records = rollout_policy(
        env,
        policy,
        args_cli.max_episodes,
        collision_force_thresh=args_cli.collision_force_thresh,
        stochastic=args_cli.stochastic,
        seed=args_cli.seed,
        recorder=recorder,
        desc="Evaluating policy",
    )
    env.close()

    summary = summarize(records)
    logger.info(f"{num_envs} envs x {args_cli.max_episodes} episodes")
    log_summary(summary)

    metrics_path = os.path.join(log_dir, "eval_metrics.json")
    if args_cli.distillation:
        policy_mode = "stochastic" if args_cli.stochastic else "deterministic"
        metrics_path = recorder.out_dir / f"eval_metrics_{policy_mode}.json"

    os.makedirs(log_dir, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({**records, "summary": summary}, f)
    logger.info(f"Per-episode metrics written to {metrics_path}")


if __name__ == "__main__":
    try:
        run_simulator()

    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()

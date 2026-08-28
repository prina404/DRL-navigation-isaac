import argparse
import sys

from isaaclab.app import AppLauncher

from cfg.CFG import get_map_name, get_scene_usd_path, set_map_name

# # add argparse arguments
parser = argparse.ArgumentParser(description="Evaluation of a distilled student policy.")
parser.add_argument(
    "--map",
    type=str,
    required=True,
    help="Map to run on, e.g. 40, 0040 or kujiale_0040. Overrides current_env in dataset_cfg.yaml.",
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
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

parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Run name under ckpts/distillation or a path to an exported actor. Defaults to the most recent run.",
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

parser.add_argument(
    "--stochastic",
    action="store_true",
    default=False,
    help="Execute actions sampled from the policy distribution instead of its deterministic mean.",
)



AppLauncher.add_app_launcher_args(parser)

# # append AppLauncher cli args
args_cli, hydra_argv = parser.parse_known_args()
set_map_name(args_cli.map)  # before anything reads the map back out of cfg.CFG
# Only the vision/depth tasks and video recording need cameras. Enabling them otherwise boots the RTX rendering Kit
# experience, whose stage population cost scales with --num_envs.
args_cli.enable_cameras = args_cli.video or "vision" in args_cli.task or "depth" in args_cli.task

sys.argv = [sys.argv[0]] + hydra_argv
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import glob
import json
import os
import shutil
import traceback
from datetime import datetime

import gymnasium as gym
import hydra
from hydra.utils import get_original_cwd
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from loguru import logger
from omegaconf import DictConfig

from distillation.policy_eval import (
    build_student_runner,
    load_student_weights,
    log_summary,
    read_student_infos,
    rollout_policy,
    summarize,
)
from tasks.task_utils import get_env_config

FILE_PATH = os.path.join(os.path.dirname(__file__), "src/cfg")


def _student_checkpoint(root: str, name: str | None) -> str:
    """Exported actor of a student run: a path, a run directory under ``root``, or the most recent run in it."""
    if name is None:
        runs = glob.glob(os.path.join(root, "*", "student_actor.pt"))
        if not runs:
            raise FileNotFoundError(f"No */student_actor.pt found under {root}")
        return max(runs, key=os.path.getmtime)
    return name if os.path.isfile(name) else os.path.join(root, name, "student_actor.pt")


def _archive_previous_run(log_dir: str, policy_mode: str) -> None:
    """Set aside a previous evaluation of ``policy_mode`` so this one does not overwrite it.

    The log directory is per map and stable across runs, so the deterministic and the stochastic evaluation of a
    map sit next to each other. Only the artefacts of the mode about to run are moved: the other mode's results
    stay where they are.
    """
    artefacts = [
        os.path.join(log_dir, f"eval_metrics_{policy_mode}.json"),
        os.path.join(log_dir, f"videos_{policy_mode}"),
    ]
    artefacts = [path for path in artefacts if os.path.exists(path)]
    if not artefacts:
        return

    # Stamp the archive with when the old run was produced, the way the old <map>_<timestamp> folders did,
    # rather than with when it happened to be superseded.
    stamp = datetime.fromtimestamp(min(os.path.getmtime(path) for path in artefacts)).strftime("%m-%d_%H-%M")
    archive_dir = os.path.join(log_dir, stamp)
    for counter in range(1, 100):  # two runs of the same mode within a minute would otherwise collide
        if not os.path.exists(archive_dir):
            break
        archive_dir = os.path.join(log_dir, f"{stamp}_{counter}")
    os.makedirs(archive_dir)

    for path in artefacts:
        shutil.move(path, os.path.join(archive_dir, os.path.basename(path)))
    logger.info(f"Archived the previous {policy_mode} run to {archive_dir}")


@hydra.main(config_path=None)
def run_simulator(cfg: DictConfig):
    # One directory per map holding both policy modes, mirroring how the distillation recorder groups its
    # datasets under distillation_data/raw_data/<map_name>/.
    policy_mode = "stochastic" if args_cli.stochastic else "deterministic"
    log_root_path = os.path.abspath(os.path.join(get_original_cwd(), "logs", "rsl_rl", "student_validation"))
    log_dir = os.path.join(log_root_path, get_map_name())
    os.makedirs(log_dir, exist_ok=True)
    logger.info(f"Logging the {policy_mode} evaluation in directory: {log_dir}")
    _archive_previous_run(log_dir, policy_mode)

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
            "video_folder": os.path.join(log_dir, f"videos_{policy_mode}"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        logger.info("Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env)
    logger.info("RslRlVecEnvWrapper applied to gym environment")

    # Student policy setup
    ckpt_path = _student_checkpoint(os.path.join(get_original_cwd(), "ckpts", "distillation"), args_cli.checkpoint)
    infos = read_student_infos(ckpt_path)
    logger.info(f"Loading student actor from {ckpt_path}")
    ppo_runner = build_student_runner(env, environment_cfg.obs_groups, infos, log_dir, args_cli.num_envs)
    load_student_weights(ppo_runner, ckpt_path)

    policy = ppo_runner.get_inference_policy(env.device)
    num_envs = env.num_envs

    records = rollout_policy(
        env,
        policy,
        args_cli.max_episodes,
        collision_force_thresh=args_cli.collision_force_thresh,
        stochastic=args_cli.stochastic,
        desc="Evaluating student",
    )
    env.close()

    summary = summarize(records)
    logger.info(f"{num_envs} envs x {args_cli.max_episodes} episodes")
    log_summary(summary)

    metrics_path = os.path.join(log_dir, f"eval_metrics_{policy_mode}.json")
    with open(metrics_path, "w") as f:
        json.dump({"checkpoint": ckpt_path, "policy_mode": policy_mode, **records, "summary": summary}, f)
    logger.info(f"Per-episode metrics written to {metrics_path}")


if __name__ == "__main__":
    try:
        run_simulator()

    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()

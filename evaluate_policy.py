import argparse
import sys

import torch
import tqdm
from isaaclab.app import AppLauncher

from cfg.CFG import get_scene_usd_path

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
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")

parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
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
args_cli.enable_cameras = True  # always true for go2 cfg

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

import mdp.rewards.helper_functions as rewards
from distillation.recorder import DistillationRecorder
from policy.NavPolicyv2 import load_policy_checkpoint, make_go2_policy_cfg
from tasks.task_utils import get_env_config

FILE_PATH = os.path.join(os.path.dirname(__file__), "src/cfg")


@hydra.main(config_path=None)
def run_simulator(cfg: DictConfig):

    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
        kwargs={"scene_path": get_scene_usd_path(), "use_long_horizon": True, "debug_vis": args_cli.debug_vis, "sample_voronoi_probability": 0.25},
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

    obs, _ = env.reset()
    base_env = env.unwrapped
    max_ep = args_cli.max_episodes

    episodes_done = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    episode_events = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    prev_in_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    successes: list[float] = []  # one entry per recorded episode
    collisions: list[int] = []  # collision events per recorded episode

    recorder = (
        DistillationRecorder(policy.obs_groups, 10.0, run_info) if args_cli.distillation else None
    )

    with tqdm.tqdm(total=env.num_envs * max_ep, desc="Evaluating policy") as pbar:
        while not bool((episodes_done >= max_ep).all()):
            with torch.no_grad():
                # The stochastic call is the only one that populates the distribution, hence the only one exposing
                # mean/std. It costs nothing extra: for a Gaussian the deterministic output *is* the mean, so both
                # candidate actions come out of this single forward pass and --stochastic only picks between them.
                sampled_action = policy(obs, stochastic_output=True)
                action = sampled_action if args_cli.stochastic else policy.output_mean
                if recorder is not None:
                    recorder.record(obs, action, policy.output_mean, policy.output_std)
            obs, _, dones, _ = env.step(action)
            dones = dones.bool()

            # count collisions as rising edges, so a sustained scrape counts once
            in_contact = rewards.detect_collision(base_env) > args_cli.collision_force_thresh
            episode_events += (in_contact & ~prev_in_contact).long()
            prev_in_contact = in_contact

            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if len(done_ids) > 0:
                # `goal_reached` covers both "arrived at the goal" and "no feasible path":
                # the path manager sets current_path_length to 0 when A* finds nothing.
                # `_term_dones` is written before the in-step auto-reset and survives it.
                reached = base_env.termination_manager.get_term("goal_reached")

                # cap per env so fast-terminating envs cannot dominate the sample
                keep = done_ids[episodes_done[done_ids] < max_ep]
                if len(keep) > 0:
                    successes.extend(reached[keep].float().cpu().tolist())
                    collisions.extend(episode_events[keep].cpu().tolist())
                    pbar.update(len(keep))

                episodes_done[done_ids] += 1
                episode_events[done_ids] = 0
                prev_in_contact[done_ids] = False

            if recorder is not None and recorder.is_full:
                logger.warning(
                    f"Distillation data reached the {recorder.max_bytes / (1024**3):.2f} GB limit, stopping the evaluation early."
                )
                break

    if recorder is not None:
        recorder.save()

    env.close()

    num_episodes = len(successes)
    succ = torch.tensor(successes, dtype=torch.float32)
    col = torch.tensor(collisions, dtype=torch.float32)
    collided = col[col > 0]

    success_rate = succ.mean().item() if num_episodes else float("nan")
    collision_free_rate = (col == 0).float().mean().item() if num_episodes else float("nan")
    mean_collisions = collided.mean().item() if len(collided) else 0.0
    # torch.std is nan for a single sample
    std_collisions = collided.std(unbiased=True).item() if len(collided) > 1 else 0.0

    logger.info(f"Episodes recorded: {num_episodes} ({env.num_envs} envs x {max_ep} episodes)")
    logger.info(f"Success rate (goal reached or no feasible path): {success_rate:.3f}")
    logger.info(f"Collision-free episodes: {collision_free_rate:.3%}")
    logger.info(f"Collisions per colliding episode: {mean_collisions:.2f} +/- {std_collisions:.2f} (n={len(collided)})")

    metrics_path = os.path.join(log_dir, "eval_metrics.json")
    os.makedirs(log_dir, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({"success": successes, "collisions": collisions}, f)
    logger.info(f"Per-episode metrics written to {metrics_path}")


if __name__ == "__main__":
    try:
        run_simulator()

    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()

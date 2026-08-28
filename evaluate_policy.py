import argparse
import sys

import torch
import tqdm
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

import mdp.rewards.helper_functions as rewards
from distillation.recorder import DistillationRecorder
from policy.NavPolicyv2 import load_policy_checkpoint, make_go2_policy_cfg
from tasks.task_utils import get_env_config

FILE_PATH = os.path.join(os.path.dirname(__file__), "src/cfg")


def _mean_std(values: torch.Tensor, empty: float = float("nan")) -> tuple[float, float]:
    """Mean and (unbiased) std of a 1-D tensor, with the degenerate sample sizes filled in."""
    if values.numel() == 0:
        return empty, empty
    if values.numel() == 1:  # torch.std is nan for a single sample
        return values.item(), 0.0
    return values.mean().item(), values.std(unbiased=True).item()


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

    obs, _ = env.reset()
    base_env = env.unwrapped
    max_ep = args_cli.max_episodes
    path_manager = base_env.path_manager
    # same threshold the goal_reached term uses, so the "arrived" / "no feasible path" split reproduces it exactly
    goal_thresh = base_env.termination_manager.get_term_cfg("goal_reached").params.get("threshold_m", 0.2)

    episodes_done = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    episode_events = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    episode_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    prev_in_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Plan lengths must be snapshotted: the in-step auto-reset overwrites them with the *next* episode's plan
    # before `step` returns. Taking the snapshot at the end of an iteration is exact rather than merely close:
    # the replan is an interval event, applied after the terminations of the step that follows it, so these are
    # precisely the values the next termination will see.
    plan_len = path_manager.current_path_length.clone()
    plan_len_max = path_manager.initial_path_length.clone()
    # Last length A* actually produced. On an infeasible plan `current_path_length` is 0, so this is what tells
    # us how much of the route was left when the episode ended.
    last_valid_plan_len = plan_len.clone()

    successes: list[float] = []  # one entry per recorded episode
    collisions: list[int] = []  # collision events per recorded episode
    outcomes: list[str] = []  # "goal" | "no_path" | "timeout" | "other"
    durations: list[float] = []  # wall-clock episode duration (s)
    path_lengths: list[float] = []  # metres of global plan actually covered

    recorder = (
        DistillationRecorder(policy.obs_groups, 10.0, run_info, env.num_envs, args_cli.stochastic)
        if args_cli.distillation
        else None
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
            episode_steps += 1

            # count collisions as rising edges, so a sustained scrape counts once
            in_contact = rewards.detect_collision(base_env) > args_cli.collision_force_thresh
            new_contact = in_contact & ~prev_in_contact
            episode_events += new_contact.long()
            prev_in_contact = in_contact
            if recorder is not None:
                recorder.mark_collisions(new_contact)

            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if len(done_ids) > 0:
                # `goal_reached` covers both "arrived at the goal" and "no feasible path":
                # the path manager sets current_path_length to 0 when A* finds nothing.
                # `_term_dones` is written before the in-step auto-reset and survives it.
                reached = base_env.termination_manager.get_term("goal_reached")
                timed_out = base_env.termination_manager.get_term("timeout")

                # Both still count as a success; the split feeds the outcome breakdown and the route below.
                no_path = reached & (plan_len < goal_thresh)
                arrived = reached & ~no_path
                # Length of the route the episode is measured against. An episode that made it to the goal walked
                # the whole plan. Any other one only got through the part of it that A* had already crossed off,
                # so it is scored on that shortened route instead of on the route it was originally given.
                route = torch.where(arrived, plan_len_max, (plan_len_max - last_valid_plan_len).clamp(min=0.0))
                elapsed = episode_steps.float() * base_env.step_dt

                # cap per env so fast-terminating envs cannot dominate the sample
                keep = done_ids[episodes_done[done_ids] < max_ep]
                if len(keep) > 0:
                    arrived_keep = arrived[keep].cpu().tolist()
                    no_path_keep = no_path[keep].cpu().tolist()
                    successes.extend(reached[keep].float().cpu().tolist())
                    collisions.extend(episode_events[keep].cpu().tolist())
                    outcomes.extend(
                        "goal" if a else "no_path" if n else "timeout" if t else "other"
                        for a, n, t in zip(arrived_keep, no_path_keep, timed_out[keep].cpu().tolist())
                    )
                    durations.extend(elapsed[keep].cpu().tolist())
                    path_lengths.extend(route[keep].cpu().tolist())
                    pbar.update(len(keep))

                episodes_done[done_ids] += 1
                episode_events[done_ids] = 0
                episode_steps[done_ids] = 0
                prev_in_contact[done_ids] = False
                if recorder is not None:
                    recorder.end_episodes(done_ids)

            # refresh the snapshot for the next step; envs that just reset restart from their fresh plan
            plan_len = path_manager.current_path_length.clone()
            plan_len_max = path_manager.initial_path_length.clone()
            last_valid_plan_len = torch.where(plan_len >= goal_thresh, plan_len, last_valid_plan_len)
            last_valid_plan_len[done_ids] = plan_len[done_ids]

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
    dur = torch.tensor(durations, dtype=torch.float32)
    plen = torch.tensor(path_lengths, dtype=torch.float32)

    success_rate = succ.mean().item() if num_episodes else float("nan")
    collision_free_rate = (col == 0).float().mean().item() if num_episodes else float("nan")
    mean_collisions, std_collisions = _mean_std(collided, empty=0.0)

    outcome_counts = {name: outcomes.count(name) for name in ("goal", "no_path", "timeout", "other")}

    # A timed-out episode never completed its route, so its duration says nothing about how fast the policy is:
    # it is the episode length by construction. Timing stats are therefore reported over the rest, normalized by
    # the route each of those episodes actually completed.
    finished = torch.tensor([o != "timeout" for o in outcomes], dtype=torch.bool)
    scorable = finished & (plen > 1e-3)  # a zero-length route cannot normalize a duration
    mean_time, std_time = _mean_std(dur[finished])
    mean_len, std_len = _mean_std(plen[finished])
    mean_norm_time, std_norm_time = _mean_std(dur[scorable] / plen[scorable])

    logger.info(f"Episodes recorded: {num_episodes} ({env.num_envs} envs x {max_ep} episodes)")
    logger.info(f"Success rate (goal reached or no feasible path): {success_rate:.3f}")
    logger.info(
        "Outcomes: {goal} reached the goal, {no_path} ended on an infeasible global path, "
        "{timeout} timed out, {other} other".format(**outcome_counts)
    )
    logger.info(f"Collision-free episodes: {collision_free_rate:.3%}")
    logger.info(f"Collisions per colliding episode: {mean_collisions:.2f} +/- {std_collisions:.2f} (n={len(collided)})")
    logger.info(f"Completion time, excluding timeouts: {mean_time:.2f} +/- {std_time:.2f} s (n={int(finished.sum())})")
    logger.info(f"Path length covered, excluding timeouts: {mean_len:.2f} +/- {std_len:.2f} m")
    logger.info(
        f"Completion time per path metre: {mean_norm_time:.3f} +/- {std_norm_time:.3f} s/m (n={int(scorable.sum())})"
    )

    metrics_path = os.path.join(log_dir, "eval_metrics.json")
    if args_cli.distillation:
        policy_mode = "stochastic" if args_cli.stochastic else "deterministic"
        metrics_path = recorder.out_dir / f"eval_metrics_{policy_mode}.json"

    os.makedirs(log_dir, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "success": successes,
                "collisions": collisions,
                "outcome": outcomes,
                "completion_time_s": durations,
                "path_length_m": path_lengths,
                "summary": {
                    "num_episodes": num_episodes,
                    "success_rate": success_rate,
                    "outcome_counts": outcome_counts,
                    "collision_free_rate": collision_free_rate,
                    "collisions_per_colliding_episode": [mean_collisions, std_collisions],
                    "completion_time_s": [mean_time, std_time],
                    "path_length_m": [mean_len, std_len],
                    "completion_time_per_metre": [mean_norm_time, std_norm_time],
                },
            },
            f,
        )
    logger.info(f"Per-episode metrics written to {metrics_path}")


if __name__ == "__main__":
    try:
        run_simulator()

    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()

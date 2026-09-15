"""Rolling a navigation policy out in the simulator and scoring the episodes it produces.

Shared by ``evaluate_policy.py`` (the PPO teacher, optionally recording a distillation dataset),
``evaluate_student.py``, which scores one checkpoint, and ``eval_distil_process.py``, which sweeps a whole
distillation run. Keeping the three on one loop is what makes their metrics comparable, and it is what lets
``seed`` put a teacher and a student on the same episodes. Like every module under ``src/``, it may only be
imported once ``AppLauncher`` has started ``simulation_app``.
"""

from __future__ import annotations

from typing import Any

import torch
import tqdm
from loguru import logger

import mdp.rewards.helper_functions as rewards


def mean_std(values: torch.Tensor, empty: float = float("nan")) -> tuple[float, float]:
    """Mean and (unbiased) std of a 1-D tensor, with the degenerate sample sizes filled in."""
    if values.numel() == 0:
        return empty, empty
    if values.numel() == 1:  # torch.std is nan for a single sample
        return values.item(), 0.0
    return values.mean().item(), values.std(unbiased=True).item()


def rollout_policy(
    env: Any,
    policy: Any,
    max_episodes: int,
    collision_force_thresh: float = 3.0,
    stochastic: bool = False,
    seed: int | None = None,
    recorder: Any = None,
    desc: str = "Evaluating",
    post_step: Any = None,
) -> dict[str, list]:
    """Run every env until it has finished ``max_episodes`` episodes, returning one record per episode.

    ``post_step``, if given, is called with no arguments right after every ``env.step``. ``eval_ros2_planner.py``
    uses it to push the simulator state out over ROS and to service the Nav2 stacks, so the planner baseline is
    scored by exactly this loop rather than by a copy of it.
    """
    if seed is not None:
        env.unwrapped.seed(seed)

    obs, _ = env.reset()
    base_env = env.unwrapped
    path_manager = base_env.path_manager
    # same threshold the goal_reached term uses, so the "arrived" / "no feasible path" split reproduces it exactly
    goal_thresh = base_env.termination_manager.get_term_cfg("goal_reached").params.get("threshold_m", 0.2)

    robot = base_env.scene["robot"]

    episodes_done = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    episode_events = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    episode_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    prev_in_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # running sum of the per-step speed, turned into the episode mean by dividing by `episode_steps`
    speed_sum = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    # still needed to tell "arrived at the goal" from "no feasible path", both of which terminate on `goal_reached`
    plan_len = path_manager.current_path_length.clone()

    successes: list[float] = []  # one entry per recorded episode
    collisions: list[int] = []  # collision events per recorded episode
    outcomes: list[str] = []  # "goal" | "no_path" | "timeout" | "other"
    durations: list[float] = []  # simulated episode duration (s)
    speeds: list[float] = []  # mean linear COM speed over the episode's steps (m/s)

    with tqdm.tqdm(total=env.num_envs * max_episodes, desc=desc) as pbar:
        while not bool((episodes_done >= max_episodes).all()):
            # Ground speed the action is about to be taken at. Sampled before `env.step`, because a step that
            # terminates an env resets it internally, and `teleport_robots` zeroes the root velocity -- reading
            # afterwards would charge the next episode's standstill to the episode that just ended.
            speed_sum += torch.linalg.norm(robot.data.root_com_lin_vel_w.torch[:, :2], dim=-1).to(env.device)
            with torch.no_grad():
                sampled_action = policy(obs, stochastic_output=True)
                action = sampled_action if stochastic else policy.output_mean
                if recorder is not None:
                    recorder.record(obs, action, policy.output_mean, policy.output_std)
            obs, _, dones, _ = env.step(action)
            if post_step is not None:
                post_step()
            dones = dones.bool()
            episode_steps += 1

            # count collisions as rising edges, so a sustained contact counts once
            in_contact = rewards.detect_collision(base_env) > collision_force_thresh
            new_contact = in_contact & ~prev_in_contact
            episode_events += new_contact.long()
            prev_in_contact = in_contact
            if recorder is not None:
                recorder.mark_collisions(new_contact)

            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if len(done_ids) > 0:
                # `goal_reached` covers both "arrived at the goal" and "no feasible path":
                reached = base_env.termination_manager.get_term("goal_reached")
                timed_out = base_env.termination_manager.get_term("timeout")

                # Both still count as a success; the split feeds the outcome breakdown.
                no_path = reached & (plan_len < goal_thresh)
                arrived = reached & ~no_path
                elapsed = episode_steps.float() * base_env.step_dt
                mean_speed = speed_sum / episode_steps.clamp(min=1).float()

                # cap per env so fast-terminating envs cannot dominate the sample
                keep = done_ids[episodes_done[done_ids] < max_episodes]
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
                    speeds.extend(mean_speed[keep].cpu().tolist())
                    pbar.update(len(keep))

                episodes_done[done_ids] += 1
                episode_events[done_ids] = 0
                episode_steps[done_ids] = 0
                speed_sum[done_ids] = 0.0
                prev_in_contact[done_ids] = False
                if recorder is not None:
                    recorder.end_episodes(done_ids)

            # refresh the snapshot for the next step; envs that just reset restart from their fresh plan
            plan_len = path_manager.current_path_length.clone()

            if recorder is not None and recorder.is_full:
                logger.warning(
                    f"Distillation data reached the {recorder.max_bytes / (1024**3):.2f} GB limit, "
                    "stopping the evaluation early."
                )
                break

    if recorder is not None:
        recorder.save()

    return {
        "success": successes,
        "collisions": collisions,
        "outcome": outcomes,
        "completion_time_s": durations,
        "mean_speed_mps": speeds,
    }


def summarize(records: dict[str, list]) -> dict[str, Any]:
    """Aggregate the per-episode records of ``rollout_policy`` into the summary block of the metrics file."""
    outcomes = records["outcome"]
    num_episodes = len(records["success"])
    succ = torch.tensor(records["success"], dtype=torch.float32)
    col = torch.tensor(records["collisions"], dtype=torch.float32)
    collided = col[col > 0]
    dur = torch.tensor(records["completion_time_s"], dtype=torch.float32)
    speed = torch.tensor(records["mean_speed_mps"], dtype=torch.float32)

    # A timed-out episode never completed its route, so its *duration* says nothing about how fast the policy is:
    # it is the episode length by construction. Completion time is therefore reported over the rest. Mean speed
    # carries no such bias -- it is an average over steps, equally well defined however the episode ended -- so
    # it is reported over every episode, timeouts included.
    finished = torch.tensor([outcome != "timeout" for outcome in outcomes], dtype=torch.bool)

    OUTCOMES = ("goal", "no_path", "timeout", "other")
    return {
        "num_episodes": num_episodes,
        "success_rate": succ.mean().item() if num_episodes else float("nan"),
        "outcome_counts": {name: outcomes.count(name) for name in OUTCOMES},
        "collision_free_rate": (col == 0).float().mean().item() if num_episodes else float("nan"),
        "collisions_per_colliding_episode": list(mean_std(collided, empty=0.0)),
        "num_colliding_episodes": int(collided.numel()),
        "completion_time_s": list(mean_std(dur[finished])),
        "num_finished_episodes": int(finished.sum()),
        "mean_speed_mps": list(mean_std(speed)),
    }


def log_summary(summary: dict[str, Any]) -> None:
    """Print the summary of :func:`summarize` the way ``evaluate_student.py`` always has."""
    logger.info(f"Episodes recorded: {summary['num_episodes']}")
    logger.info(f"Success rate (goal reached or no feasible path): {summary['success_rate']:.3f}")
    logger.info(
        "Outcomes: {goal} reached the goal, {no_path} ended on an infeasible global path, "
        "{timeout} timed out, {other} other".format(**summary["outcome_counts"])
    )
    logger.info(f"Collision-free episodes: {summary['collision_free_rate']:.3%}")
    logger.info(
        "Collisions per colliding episode: {:.2f} +/- {:.2f} (n={})".format(
            *summary["collisions_per_colliding_episode"], summary["num_colliding_episodes"]
        )
    )
    logger.info(
        "Completion time, excluding timeouts: {:.2f} +/- {:.2f} s (n={})".format(
            *summary["completion_time_s"], summary["num_finished_episodes"]
        )
    )
    logger.info("Mean linear COM speed, all episodes: {:.3f} +/- {:.3f} m/s".format(*summary["mean_speed_mps"]))


def read_student_infos(ckpt_path: str) -> dict[str, Any]:
    infos = torch.load(ckpt_path, weights_only=False, map_location="cpu")["infos"]
    if infos.get("last_activation") != "MeanTanhHead":
        raise ValueError(f"{ckpt_path} is not a student export, its head is {infos.get('last_activation')!r}")
    return infos


def build_student_runner(env: Any, obs_groups: dict[str, list[str]], infos: dict[str, Any], log_dir: str, num_envs: int):
    """A runner whose actor has the student's architecture, with no weights loaded into it yet.

    The architecture comes from :data:`distillation.student_cfg.student_policy_cfg`
    """
    from rsl_rl.runners import OnPolicyRunner

    from distillation.student_cfg import make_student_policy_cfg
    from distillation.train_student_net import MeanTanhHead
    from policy.NavPolicyv2 import ACTOR_OBS_SET

    # Note: OnPolicyRunner consumes its configuration destructively, so a fresh copy is built here
    policy_cfg = make_student_policy_cfg(obs_groups)  # obs_groups needed for encoder initialization
    policy_cfg["num_envs"] = num_envs
    policy_cfg["actor"]["distribution_cfg"] = infos["distribution_cfg"]  # the run's own init_std
    policy_cfg["obs_groups"][ACTOR_OBS_SET] = infos["obs_groups"][ACTOR_OBS_SET]

    runner = OnPolicyRunner(env, policy_cfg, log_dir=log_dir, device=policy_cfg["device"])
    runner.alg.actor.mlp.add_module(str(len(runner.alg.actor.mlp)), MeanTanhHead())
    return runner


def load_student_weights(runner: Any, ckpt_path: str) -> None:
    from policy.NavPolicyv2 import load_policy_checkpoint

    load_policy_checkpoint(
        runner, ckpt_path, load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": True}
    )

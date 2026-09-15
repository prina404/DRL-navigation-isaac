import argparse
import sys

from isaaclab.app import AppLauncher

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
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Seed used for the environment. Matches evaluate_policy.py, so the planner and the policies are "
    "scored on the same episodes.",
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
parser.add_argument("--task", type=str, default="go2_lidar_full", help="Name of the task configuration to use for training.")

AppLauncher.add_app_launcher_args(parser)

# # append AppLauncher cli args
args_cli, hydra_argv = parser.parse_known_args()
args_cli.enable_cameras = args_cli.video or "vision" in args_cli.task or "depth" in args_cli.task

sys.argv = [sys.argv[0]] + hydra_argv
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import json
import os
import signal
import subprocess
import time
import traceback
from datetime import datetime

import gymnasium as gym
import hydra
import rclpy
import torch
from hydra.utils import get_original_cwd
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from loguru import logger
from omegaconf import DictConfig
from rclpy.executors import SingleThreadedExecutor

from cfg.CFG import ROOT_DIR, get_map_name, get_scene_usd_path, set_map_name
from distillation.policy_eval import log_summary, rollout_policy, summarize
from tasks.task_utils import get_env_config
from ros2.Nav2Manager import kill_nav2_lifecycle, pump_ros_data, wait_for_nav2_ready, MultiEnvNavigator
from ros2.RosDataManager import RosDataManager

set_map_name(args_cli.map)  # before anything reads the map back out of cfg.CFG


FILE_PATH = os.path.join(os.path.dirname(__file__), "src/cfg")

NAV2_BRINGUP_ATTEMPTS = 5
"""Relaunch the whole stack this many times before giving up on getting every robot active."""


@hydra.main(config_path=None)
def run_simulator(cfg: DictConfig):

    run_info = get_map_name() + datetime.now().strftime("_%m-%d_%H-%M")
    log_root_path = os.path.abspath(os.path.join(get_original_cwd(), "logs", "rsl_rl", "ros2_validation"))
    logger.info(f"Logging experiment in directory: {log_root_path}")
    logger.info(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)
    os.makedirs(log_dir, exist_ok=True)

    # Go2 Env setup
    environment_cfg = get_env_config(args_cli.task)
    environment_cfg.curriculum = None
    environment_cfg.scene.num_envs = cfg.num_envs if args_cli.num_envs is None else args_cli.num_envs
    environment_cfg.seed = args_cli.seed if args_cli.seed is not None else 42
    environment_cfg.log_dir = log_dir

    # Create the whole scene
    logger.info("Creating gym environment...")
    gym.register(
        id="Isaac-indoor-navigation-go2-v0",
        entry_point="navigation_env.NavigationEnv:NavEnv",
        disable_env_checker=True,
        kwargs={"scene_path": get_scene_usd_path(), "use_long_horizon": False, "sample_voronoi_probability": 0.25},
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

    env = RslRlVecEnvWrapper(env) # keep the wrapper for logging purposes
    logger.info("RslRlVecEnvWrapper applied to gym environment")

    env.reset()

    # Ros2 DataManager setup

    rclpy.init()
    __env = env.unwrapped
    _camera = __env.scene.sensors.get("camera")
    ros2_dm = RosDataManager(__env, __env.scene["lidar"], _camera, is_depth_camera="depth" in environment_cfg.obs_groups["policy"])
    ros_executor = SingleThreadedExecutor()
    ros_executor.add_node(ros2_dm)
    ros2_dm.pub_ros2_data(ros2_dm.zero_time)

    # Init NavStack
    kill_nav2_lifecycle()  # ensure no leftover nodes from previous runs
    pump_ros_data(ros2_dm, 2.0)  # bring `/clock` and TF up before the nodes that read them, see pump_ros_data
    cmd = [
        "bash",
        "-lc",
        f"source {ROOT_DIR}/ros_ws/install/setup.bash && "
        "ros2 launch navigation_bringup navigation_bringup.launch.py "
        f"num_robots:={env.num_envs} robot_prefix:=robot use_sim_time:=true "
        f"use_rviz:={'false' if args_cli.headless else 'true'}",
    ]
    # `import cv2` (pulled in by MapManager) points QT_QPA_PLATFORM_PLUGIN_PATH at opencv's
    # own bundled Qt plugins. rviz2 is built against system Qt6, inherits that path through
    # this Popen, fails to load the plugin and aborts with SIGABRT -- which is why no rviz
    # window appeared. Hand the launch a clean Qt environment.
    launch_env = {k: v for k, v in os.environ.items() if k not in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH")}
    nav_proc = None

    def stop_nav2():
        if nav_proc is not None:
            nav_proc.send_signal(signal.SIGINT)
            time.sleep(2.0)
        kill_nav2_lifecycle()

    def ros_send_sigint(*args):
        logger.debug("SIGINT received, shutting down gracefully...")
        stop_nav2()
        rclpy.shutdown()
        exit(0)

    signal.signal(signal.SIGINT, ros_send_sigint)

    for attempt in range(1, NAV2_BRINGUP_ATTEMPTS + 1):
        nav_proc = subprocess.Popen(cmd, env=launch_env, stderr=subprocess.STDOUT)
        try:
            wait_for_nav2_ready(ros2_dm, env.num_envs, robot_prefix="robot", timeout=120.0)
            break
        except Exception as exc:  # noqa: BLE001 - a stalled bringup is expected, retry it
            stop_nav2()
            logger.warning(f"Nav2 bringup attempt {attempt}/{NAV2_BRINGUP_ATTEMPTS} failed: {exc}")
            if attempt == NAV2_BRINGUP_ATTEMPTS:
                raise
            pump_ros_data(ros2_dm, 2.0)

    multi_nav = MultiEnvNavigator(env.unwrapped, ros2_dm)

    # --- Eval loop ---
    drain_budget = 2 * env.num_envs + 4

    class Nav2Policy:
        """Adapter presenting the latest cmd_vel of every env with the interface `rollout_policy` expects."""

        def __call__(self, obs, stochastic_output: bool = False) -> torch.Tensor:
            for _ in range(drain_budget):
                ros_executor.spin_once(timeout_sec=0.0)  # process cmd_vel callbacks
            ros2_dm.expire_stale_commands()
            return self.output_mean

        @property
        def output_mean(self) -> torch.Tensor:
            return ros2_dm.base_vel_cmd_input.to(env.device)

    def post_step() -> None:
        # Publish TF/odom/scan before anything that can stall: the simulator is the only source of `/clock` and
        # of odom -> base_link, and Nav2 cannot activate, plan or control without them.
        ros2_dm.pub_ros2_data()
        multi_nav.step()

    records = rollout_policy(
        env,
        Nav2Policy(),
        args_cli.max_episodes,
        collision_force_thresh=args_cli.collision_force_thresh,
        seed=args_cli.seed,
        desc="Evaluating planner",
        post_step=post_step,
    )

    summary = summarize(records)
    logger.info(f"{env.num_envs} envs x {args_cli.max_episodes} episodes")
    log_summary(summary)

    metrics_path = os.path.join(log_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"planner": "nav2", **records, "summary": summary}, f)
    logger.info(f"Per-episode metrics written to {metrics_path}")

    # cleanup
    multi_nav.shutdown()
    env.close()
    ros_send_sigint()


if __name__ == "__main__":
    try:
        run_simulator()

    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()

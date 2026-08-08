import argparse
import sys

from isaaclab.app import AppLauncher

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
parser.add_argument("--task", type=str, default="go2_vision_full", help="Name of the task configuration to use for training.")

AppLauncher.add_app_launcher_args(parser)

# # append AppLauncher cli args
args_cli, hydra_argv = parser.parse_known_args()
args_cli.enable_cameras = True  # always true for go2 cfg

args_cli.kit_args = (args_cli.kit_args or "") + " --enable isaacsim.ros2.bridge"
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
import tqdm
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from loguru import logger
from omegaconf import DictConfig

import mdp.actions.go2_locomotion_policy as go2_control
import mdp.rewards.helper_functions as rewards
from cfg.CFG import ROOT_DIR, get_scene_usd_path
from tasks.task_utils import get_env_config
from ros2.Nav2Manager import kill_nav2_lifecycle, wait_for_nav2_ready, MultiEnvNavigator
from ros2.RosDataManager import RosDataManager


FILE_PATH = os.path.join(os.path.dirname(__file__), "src/cfg")


@hydra.main()
def run_simulator(cfg: DictConfig):

    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", "ros2_validation"))
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
        kwargs={"scene_path": get_scene_usd_path(), "use_long_horizon": True, "sample_voronoi_probability": 0.25},
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

    obs, _ = env.reset()

    # Ros2 DataManager setup

    rclpy.init()
    __env = env.unwrapped
    _camera = __env.scene["camera"] if "camera" in __env.scene.__dict__ else None
    ros2_dm = RosDataManager(__env, __env.scene["lidar"], _camera, is_depth_camera="depth" in environment_cfg.obs_groups["policy"])
    ros2_dm.pub_ros2_data(ros2_dm.zero_time) 

    # Init NavStack
    kill_nav2_lifecycle()  # ensure no leftover nodes from previous runs
    cmd = [
        "bash",
        "-lc",
        f"source {ROOT_DIR}/ros_ws/install/setup.bash && "
        "ros2 launch navigation_bringup navigation_bringup.launch.py "
        f"num_robots:={env.num_envs} robot_prefix:=robot use_sim_time:=true",
    ]
    # `import cv2` (pulled in by MapManager) points QT_QPA_PLATFORM_PLUGIN_PATH at opencv's
    # own bundled Qt plugins. rviz2 is built against system Qt6, inherits that path through
    # this Popen, fails to load the plugin and aborts with SIGABRT -- which is why no rviz
    # window appeared. Hand the launch a clean Qt environment.
    launch_env = {k: v for k, v in os.environ.items() if k not in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH")}
    nav_proc = subprocess.Popen(
        cmd,
        env=launch_env,
        #stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    def ros_send_sigint(*args):
        logger.debug("SIGINT received, shutting down gracefully...")
        nav_proc.send_signal(signal.SIGINT)
        time.sleep(2.0)
        kill_nav2_lifecycle()
        rclpy.shutdown()
        exit(0)

    signal.signal(signal.SIGINT, ros_send_sigint)
    wait_for_nav2_ready(ros2_dm, env.num_envs, robot_prefix="robot", timeout=60.0)

    multi_nav = MultiEnvNavigator(env.unwrapped)

    # --- Eval loop ---
    num_envs = env.num_envs
    base_env = env.unwrapped
    max_ep = args_cli.max_episodes

    episodes_done = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    episode_events = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    prev_in_contact = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

    successes: list[float] = []  # one entry per recorded episode
    collisions: list[int] = []  # collision events per recorded episode

    #env.unwrapped.nominal_weight = 0.5
    with tqdm.tqdm(total=num_envs * max_ep, desc="Evaluating planner") as pbar:
        while not bool((episodes_done >= max_ep).all()):
            rclpy.spin_once(ros2_dm, timeout_sec=0.0)  # process cmd_vel callbacks

            with torch.no_grad():
                action = ros2_dm.base_vel_cmd_input.to(env.device)

            _, _, dones, _ = env.step(action)
            # Publish TF/odom/scan before anything that can stall: the simulator is the only
            # source of odom -> base_link, and Nav2 cannot activate without it.
            ros2_dm.pub_ros2_data(ros2_dm.get_time())
            multi_nav.step()

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

    num_episodes = len(successes)
    succ = torch.tensor(successes, dtype=torch.float32)
    col = torch.tensor(collisions, dtype=torch.float32)
    collided = col[col > 0]

    success_rate = succ.mean().item() if num_episodes else float("nan")
    collision_free_rate = (col == 0).float().mean().item() if num_episodes else float("nan")
    mean_collisions = collided.mean().item() if len(collided) else 0.0
    # torch.std is nan for a single sample
    std_collisions = collided.std(unbiased=True).item() if len(collided) > 1 else 0.0

    logger.info(f"Episodes recorded: {num_episodes} ({num_envs} envs x {max_ep} episodes)")
    logger.info(f"Success rate (goal reached or no feasible path): {success_rate:.3f}")
    logger.info(f"Collision-free episodes: {collision_free_rate:.3%}")
    logger.info(f"Collisions per colliding episode: {mean_collisions:.2f} +/- {std_collisions:.2f} (n={len(collided)})")

    metrics_path = os.path.join(log_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"success": successes, "collisions": collisions}, f)
    logger.info(f"Per-episode metrics written to {metrics_path}")

    # cleanup
    multi_nav.shutdown()
    env.close()
    ros_send_sigint()


if __name__ == "__main__":
    try:
        run_simulator()
        print("asdfjdbgak")
    except Exception:
        traceback.print_exc()
        simulation_app.close()

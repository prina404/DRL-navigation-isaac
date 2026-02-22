import argparse
import sys

from isaaclab.app import AppLauncher

# # add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on basic RL environment.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")

AppLauncher.add_app_launcher_args(parser)

# # append AppLauncher cli args
args_cli, hydra_argv = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.kit_args = (
    (args_cli.kit_args or "")
    + " --enable isaacsim.sensors.rtx"
    + " --enable isaacsim.asset.gen.omap"
    + " --enable omni.kit.profiler.tracy"
)
args_cli.profiler_backend = ["tracy"]
sys.argv = [sys.argv[0]] + hydra_argv
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import time
import traceback

import gymnasium as gym
import hydra
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
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
    go2_env_cfg.decimation = 16
    go2_env_cfg.sim.render_interval = go2_env_cfg.decimation

    # Create the whole scene
    logger.info("Creating gym environment...")
    gym.register(
        id="Isaac-indoor-navigation-go2-v0",
        entry_point="lab.MyEnv:MyEnv",
        disable_env_checker=True,
    )

    env = gym.make("Isaac-indoor-navigation-go2-v0", cfg=go2_env_cfg)
    env = RslRlVecEnvWrapper(env)
    logger.info("RslRlVecEnvWrapper applied to gym environment")

    # Navigation Policy setup
    agent_cfg = go2_policy_cfg
    agent_cfg["num_envs"] = cfg.num_envs
    # agent_cfg["policy"]["obs"] = env.get_observations()
    # ckpt_path = get_checkpoint_path(log_path=os.path.abspath("ckpts"),
    #                                 run_dir=agent_cfg["load_run"],
    #                                 checkpoint=agent_cfg["load_checkpoint"])
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
    # ppo_runner.load(ckpt_path)

    policy = ppo_runner.get_inference_policy(device=agent_cfg["device"])

    # Run simulation
    obs, _ = env.reset()

    sim_step_dt = float(go2_env_cfg.sim.dt * go2_env_cfg.decimation)

    paused = False
    step_count = 0
    wall_time_acc = 0.0
    policy_time_acc = 0.0
    env_step_time_acc = 0.0

    logger.info("Starting simulation loop...")
    while simulation_app.is_running():
        wall_start = time.time()

        if paused:
            time.sleep(0.01)
            continue
        with torch.no_grad():
            policy_start = time.time()
            actions = policy(obs)
            policy_time_acc += time.time() - policy_start

            env_step_start = time.time()
            obs, reward, terminated, truncated = env.step(actions)
            env_step_time_acc += time.time() - env_step_start

        wall_time_acc += time.time() - wall_start

        step_count += 1
        if step_count % 50 == 0:
            avg_wall_ms = (wall_time_acc / 50) * 1000.0
            avg_policy_ms = (policy_time_acc / 50) * 1000.0
            avg_env_step_ms = (env_step_time_acc / 50) * 1000.0
            print(
                f"[timing] dt={sim_step_dt:.4f}s | wall={avg_wall_ms:.3f} ms | "
                f"policy={avg_policy_ms:.3f} ms | env.step={avg_env_step_ms:.3f} ms | "
                f"reward={reward.mean().item():.3f}"
            )
            wall_time_acc = 0.0
            policy_time_acc = 0.0
            env_step_time_acc = 0.0

        if step_count % 150 == 0:
            env.reset()
            logger.info("Environment reset")


if __name__ == "__main__":
    try:
        run_simulator()

    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()

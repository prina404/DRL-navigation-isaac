import os
from typing import Callable
import torch
import carb
import gymnasium as gym
from isaaclab.envs import ManagerBasedEnv
from go2.go2_ctrl_cfg import unitree_go2_flat_cfg, unitree_go2_rough_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import OnPolicyRunner

base_vel_cmd_input = None

RANDOM_CMD_ENABLED = True  # set False to go back to keyboard control
RANDOM_CMD_HOLD_STEPS = 1000  # resample every N calls to base_vel_cmd (roughly N sim steps)
RANDOM_LIN_X_RANGE = (-1.5, 1.5)  # m/s
RANDOM_LIN_Y_RANGE = (-1.0, 1.0)  # m/s
RANDOM_ANG_Z_RANGE = (-1.5, 1.5)  # rad/s

# NEW: ensure motion (avoid tiny/zero commands)
RANDOM_MIN_LIN_SPEED = 0.25  # m/s (min planar speed)
RANDOM_MIN_ANG_SPEED = 0.25  # rad/s (min yaw rate magnitude)

_rand_cmd_step_counter = 0


def init_base_vel_cmd(num_envs, device: str | torch.device = "cpu"):
    global base_vel_cmd_input
    base_vel_cmd_input = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)


def _ensure_cmd_tensor(env: ManagerBasedEnv) -> None:
    """Make sure base_vel_cmd_input exists, has correct shape, and is on env.device."""
    global base_vel_cmd_input
    num_envs = getattr(env, "num_envs", None)
    if num_envs is None:
        num_envs = base_vel_cmd_input.shape[0] if base_vel_cmd_input is not None else 1

    if (
        base_vel_cmd_input is None
        or base_vel_cmd_input.shape != (num_envs, 3)
        or base_vel_cmd_input.device != env.device
    ):
        init_base_vel_cmd(num_envs, device=env.device)


def _uniform(n: int, lo: float, hi: float, device: torch.device) -> torch.Tensor:
    return (hi - lo) * torch.rand((n,), device=device) + lo


def _resample_random_cmd(env: ManagerBasedEnv) -> None:
    global base_vel_cmd_input
    _ensure_cmd_tensor(env)

    n = base_vel_cmd_input.shape[0]
    dev = base_vel_cmd_input.device

    # sample
    vx = _uniform(n, RANDOM_LIN_X_RANGE[0], RANDOM_LIN_X_RANGE[1], dev)
    vy = _uniform(n, RANDOM_LIN_Y_RANGE[0], RANDOM_LIN_Y_RANGE[1], dev)
    wz = _uniform(n, RANDOM_ANG_Z_RANGE[0], RANDOM_ANG_Z_RANGE[1], dev)

    # enforce non-trivial planar speed
    v_norm = torch.sqrt(vx * vx + vy * vy)
    too_slow = v_norm < RANDOM_MIN_LIN_SPEED
    if torch.any(too_slow):
        # resample direction for the slow ones and set magnitude >= min
        theta = 2.0 * torch.pi * torch.rand((int(too_slow.sum().item()),), device=dev)
        mag = _uniform(int(too_slow.sum().item()), RANDOM_MIN_LIN_SPEED, max(RANDOM_MIN_LIN_SPEED, RANDOM_LIN_X_RANGE[1]), dev)
        vx[too_slow] = mag * torch.cos(theta)
        vy[too_slow] = mag * torch.sin(theta)

    # enforce non-trivial yaw rate
    too_slow_w = torch.abs(wz) < RANDOM_MIN_ANG_SPEED
    if torch.any(too_slow_w):
        sign = torch.where(torch.rand((int(too_slow_w.sum().item()),), device=dev) > 0.5, 1.0, -1.0)
        mag = _uniform(int(too_slow_w.sum().item()), RANDOM_MIN_ANG_SPEED, max(RANDOM_MIN_ANG_SPEED, RANDOM_ANG_Z_RANGE[1]), dev)
        wz[too_slow_w] = sign * mag

    base_vel_cmd_input[:, 0] = vx
    base_vel_cmd_input[:, 1] = vy
    base_vel_cmd_input[:, 2] = wz


def base_vel_cmd(env: ManagerBasedEnv) -> torch.Tensor:
    global base_vel_cmd_input, _rand_cmd_step_counter
    _ensure_cmd_tensor(env)

    if RANDOM_CMD_ENABLED:
        # NEW: if still zero (e.g., right after init/reset), force a sample so it moves
        if torch.all(base_vel_cmd_input == 0):
            _resample_random_cmd(env)
            _rand_cmd_step_counter = 1
        else:
            if _rand_cmd_step_counter % max(int(RANDOM_CMD_HOLD_STEPS), 1) == 0:
                _resample_random_cmd(env)
            _rand_cmd_step_counter += 1

    return base_vel_cmd_input.clone()


def sub_keyboard_event(event) -> bool:
    global base_vel_cmd_input
    lin_vel = 1.5
    ang_vel = 1.5
    if RANDOM_CMD_ENABLED:
        return True

    if base_vel_cmd_input is not None:
        dev = base_vel_cmd_input.device
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == 'W':
                base_vel_cmd_input[0] = torch.tensor([lin_vel, 0, 0], dtype=torch.float32, device=dev)
            elif event.input.name == 'S':
                base_vel_cmd_input[0] = torch.tensor([-lin_vel, 0, 0], dtype=torch.float32, device=dev)
            elif event.input.name == 'A':
                base_vel_cmd_input[0] = torch.tensor([0, lin_vel, 0], dtype=torch.float32, device=dev)
            elif event.input.name == 'D':
                base_vel_cmd_input[0] = torch.tensor([0, -lin_vel, 0], dtype=torch.float32, device=dev)
            elif event.input.name == 'Z':
                base_vel_cmd_input[0] = torch.tensor([0, 0, ang_vel], dtype=torch.float32, device=dev)
            elif event.input.name == 'C':
                base_vel_cmd_input[0] = torch.tensor([0, 0, -ang_vel], dtype=torch.float32, device=dev)

            if base_vel_cmd_input.shape[0] > 1:
                if event.input.name == 'I':
                    base_vel_cmd_input[1] = torch.tensor([lin_vel, 0, 0], dtype=torch.float32, device=dev)
                elif event.input.name == 'K':
                    base_vel_cmd_input[1] = torch.tensor([-lin_vel, 0, 0], dtype=torch.float32, device=dev)
                elif event.input.name == 'J':
                    base_vel_cmd_input[1] = torch.tensor([0, lin_vel, 0], dtype=torch.float32, device=dev)
                elif event.input.name == 'L':
                    base_vel_cmd_input[1] = torch.tensor([0, -lin_vel, 0], dtype=torch.float32, device=dev)
                elif event.input.name == 'M':
                    base_vel_cmd_input[1] = torch.tensor([0, 0, ang_vel], dtype=torch.float32, device=dev)
                elif event.input.name == '>':
                    base_vel_cmd_input[1] = torch.tensor([0, 0, -ang_vel], dtype=torch.float32, device=dev)

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            base_vel_cmd_input.zero_()
    return True

def get_rsl_flat_policy(cfg) -> tuple[RslRlVecEnvWrapper, Callable]:
    cfg.observations.policy.height_scan = None
    env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env)

    # Low level control: rsl control policy
    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_flat_cfg
    ckpt_path = get_checkpoint_path(log_path=os.path.abspath("ckpts"), 
                                    run_dir=agent_cfg["load_run"], 
                                    checkpoint=agent_cfg["load_checkpoint"])
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
    ppo_runner.load(ckpt_path)
    policy = ppo_runner.get_inference_policy(device=agent_cfg["device"])
    return env, policy

def get_rsl_rough_policy(cfg):
    env = gym.make("Isaac-Velocity-Rough-Unitree-Go2-v0", cfg=cfg)
    env = RslRlVecEnvWrapper(env)

    # Low level control: rsl control policy
    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_rough_cfg
    ckpt_path = get_checkpoint_path(log_path=os.path.abspath("ckpts"), 
                                    run_dir=agent_cfg["load_run"], 
                                    checkpoint=agent_cfg["load_checkpoint"])
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
    ppo_runner.load(ckpt_path)
    policy = ppo_runner.get_inference_policy(device=agent_cfg["device"])
    return env, policy
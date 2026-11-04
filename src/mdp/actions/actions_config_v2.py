"""Action term driving the Go2 with the sim2real locomotion policy from the ``unitree_go2_locomotion`` repository.

The navigation policy emits a 3D base velocity command ``(v_x, v_y, w_z)``, which the frozen locomotion policy turns
into joint targets. See LOCOMOTION_POLICY_V2.md for the checkpoint. What this term has to get right:

1. **45D frames, history of 8.** The 360D observation is grouped **by term, then by frame** (oldest first), not a stack
   of 8 whole frames.
2. **50 Hz control.** The policy was trained at 50 Hz and the env steps physics at 200 Hz, so the policy is ticked every
   4th physics step and the joint targets are held in between.

Observation normalization is baked into the TorchScript checkpoint.
"""

import torch
from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass
from torch import Tensor

from mdp.actions.go2_locomotion_policy_v2 import (
    GO2_SIM2REAL_CFG,
    OBS_TERM_DIMS,
    get_locomotion_policy,
)
from mdp.observations.helper_functions import get_goal_relative_position
from navigation_env.NavigationEnv import NavEnv


def _term_major_index(history_length: int) -> Tensor:
    """Row `oldest`: indices into a flattened (history, frame) ring buffer whose oldest frame sits in slot `oldest`,
    giving the observation grouped by term, then by frame from oldest to newest."""
    frame_dim = sum(dim for _, dim in OBS_TERM_DIMS)
    rows = []
    for oldest in range(history_length):
        slots = (torch.arange(history_length) + oldest) % history_length
        row, start = [], 0
        for _, dim in OBS_TERM_DIMS:
            row.append((slots[:, None] * frame_dim + torch.arange(start, start + dim)).flatten())
            start += dim
        rows.append(torch.cat(row))
    return torch.stack(rows)


def _clip_to_circle(lin_vel: Tensor, radius: Tensor) -> Tensor:
    return lin_vel / (torch.norm(lin_vel, dim=-1, keepdim=True) / radius).clamp(min=1.0)


def _clip_to_ellipse(lin_vel: Tensor, semi_axes: Tensor) -> Tensor:
    ellipse_norm = torch.norm(lin_vel / semi_axes, dim=-1, keepdim=True)  # > 1 outside the ellipse
    return lin_vel / ellipse_norm.clamp(min=1.0)


class _Go2Sim2RealPolicyAction(ActionTerm):
    cfg: "Go2Sim2RealPolicyActionCfg"

    def __init__(self, cfg: "Go2Sim2RealPolicyActionCfg", env: NavEnv):
        super().__init__(cfg, env)
        self._robot: Articulation = env.scene[cfg.asset_name]
        self._default_joint_pos = self._robot.data.default_joint_pos.torch
        self._default_joint_vel = self._robot.data.default_joint_vel.torch
        self._command_limits = torch.tensor(cfg.command_limits, device=self.device)
        # resampled on reset by the `sample_max_speed` event during training, and observed by the policy
        self.max_speed = torch.full((self.num_envs, 1), cfg.max_speed, device=self.device)
        self.controller = get_locomotion_policy(checkpoint=cfg.checkpoint, device=self.device)

        self._raw_actions = torch.zeros((self.num_envs, 3), device=self.device)
        self._velocity_command = torch.zeros_like(self._raw_actions)
        self._last_policy_action = torch.zeros((self.num_envs, int(GO2_SIM2REAL_CFG["n_actions"])), device=self.device)
        self._joint_target = self._default_joint_pos.clone()

        self._policy_decimation = max(1, round(cfg.control_dt / env.physics_dt))
        self._sim_step_counter = 0

        # Ring buffer of the last frames. Slot `_head` holds the oldest frame and is the next one overwritten.
        history_length = int(GO2_SIM2REAL_CFG["history_length"])
        frame_dim = int(GO2_SIM2REAL_CFG["frame_dim"])
        self._history = torch.zeros((self.num_envs, history_length, frame_dim), device=self.device)
        self._head = 0
        self._obs_index = _term_major_index(history_length).to(self.device)
        self._refill_history = torch.ones((self.num_envs, 1, 1), dtype=torch.bool, device=self.device)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> Tensor:
        """The base velocity command handed to the locomotion policy."""
        return self._velocity_command

    def process_actions(self, actions: Tensor):
        self._raw_actions = actions.clone()
        if self.cfg.squash_actions:
            self._velocity_command = self._command_from_policy(self._raw_actions)
        else:
            self._velocity_command = self._raw_actions
        # the fresh command is consumed on the first physics step of this env step
        self._sim_step_counter = 0

    def _command_from_policy(self, actions: Tensor) -> Tensor:
        # radial tanh: maps the plane one to one onto the unit disk, keeping the direction
        lin_vel, yaw_rate = actions[:, :2], actions[:, 2:]
        norm = torch.norm(lin_vel, dim=-1, keepdim=True)
        lin_vel = lin_vel * torch.tanh(norm) / norm.clamp(min=1e-6)
        command = torch.cat([lin_vel, torch.tanh(yaw_rate)], dim=-1) * self._command_limits

        nominal_weight = getattr(self._env, "nominal_weight", 0.0)
        if nominal_weight > 0.01:
            # both sides of the blend are within the limits, so the blend is too (the nominal w_z is within ±1)
            nominal = get_goal_relative_position(self._env, local_goal_points_forward=4)
            nominal[:, :2] = _clip_to_ellipse(nominal[:, :2], self._command_limits[:2])
            command = (1.0 - nominal_weight) * command + nominal_weight * nominal

        # shrink (v_x, v_y) onto the speed limit, then into the ellipse, keeping its direction; w_z is left as is
        lin_vel = _clip_to_circle(command[:, :2], self.max_speed)
        command[:, :2] = _clip_to_ellipse(lin_vel, self._command_limits[:2])
        return command

    def apply_actions(self):
        if self._sim_step_counter % self._policy_decimation == 0:
            self._run_locomotion_policy()
        self._sim_step_counter += 1
        self._robot.set_joint_position_target(self._joint_target)

    @torch.no_grad()
    def _run_locomotion_policy(self) -> None:
        data = self._robot.data
        frame = torch.cat(
            [
                data.root_ang_vel_b.torch,
                data.projected_gravity_b.torch,
                self._velocity_command,
                # training's `joint_pos_rel_calibrated` only adds a calibration offset, which is zero here
                data.joint_pos.torch - self._default_joint_pos,
                data.joint_vel.torch - self._default_joint_vel,
                self._last_policy_action,
            ],
            dim=-1,
        )
        # like IsaacLab's CircularBuffer, the first frame after a reset fills the whole history
        self._history = torch.where(self._refill_history, frame.unsqueeze(1), self._history)
        self._refill_history.zero_()
        self._history[:, self._head] = frame
        self._head = (self._head + 1) % self._history.shape[1]

        obs = self._history.flatten(1)[:, self._obs_index[self._head]]
        self._last_policy_action = self.controller(obs)
        self._joint_target = self._default_joint_pos + self._last_policy_action * self.cfg.action_scale

    def reset(self, env_ids: Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._velocity_command[env_ids] = 0.0
        self._last_policy_action[env_ids] = 0.0
        self._joint_target[env_ids] = self._default_joint_pos[env_ids]
        self._refill_history[env_ids] = True


@configclass
class Go2Sim2RealPolicyActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = _Go2Sim2RealPolicyAction
    asset_name: str = "robot"
    #: ``None`` uses the default checkpoint from :data:`GO2_SIM2REAL_CFG`.
    checkpoint: str | None = None
    #: Control period the locomotion policy was trained at (sim.dt 0.005 * decimation 4).
    control_dt: float = float(GO2_SIM2REAL_CFG["control_dt"])
    #: ``actions.joint_pos.scale`` of the training env, applied on top of the default joint positions.
    action_scale: float = float(GO2_SIM2REAL_CFG["action_scale"])
    #: Range of the commands, the one the locomotion policy was trained on: semi-axes of the (v_x, v_y) ellipse [m/s]
    #: and the independent w_z limit [rad/s]. The navigation policy's unbounded actions are squashed into it.
    command_limits: tuple[float, float, float] = (2.5, 1.0, 2.0)
    #: ``False`` takes the actions as the command itself, in m/s and rad/s, with no squashing, nominal action or
    #: clipping. Used by the Nav2 baseline, whose cmd_vel is already within `command_limits`.
    squash_actions: bool = True
    #: Cap on the (v_x, v_y) speed [m/s], applied after the squash; w_z is not capped. The policy observes it. Training
    #: resamples it per env on every reset (`sample_max_speed` event), so this value is only used without that event.
    max_speed: float = 2.0


@configclass
class ActionsCfg:
    # Term name kept identical to `mdp.actions.actions_config.ActionsCfg`, so anything referencing the action term
    # by name (action-rate rewards, `last_action` observations) keeps working unchanged.
    llc = Go2Sim2RealPolicyActionCfg()

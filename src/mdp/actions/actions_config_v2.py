"""Action term driving the Go2 with the sim2real locomotion policy from the ``unitree_go2_locomotion`` repository.

Drop-in alternative to :mod:`mdp.actions.actions_config`. The high-level interface is unchanged - the navigation policy
still emits a 3D base velocity command ``(v_x, v_y, w_z)`` - but the low-level policy behind it is different in three
ways that this term has to account for:

1. **No ``base_lin_vel``.** ``UnitreeGo2FlatEnvCfg_adapted`` disables that observation term, so a frame is 45 wide
   instead of the stock 48.
2. **Observation history of 8.** IsaacLab keeps one :class:`CircularBuffer` *per term* and flattens each separately
   before concatenating the group, so the 360D vector is grouped **by term, then by frame** - it is *not* a stack of 8
   whole frames. Within each term the oldest frame comes first and the newest last.
3. **50 Hz control.** The policy was trained at ``sim.dt * decimation = 0.005 * 4``. The navigation env steps physics at
   200 Hz, so the locomotion policy is ticked every 4th sim step and the joint targets are held in between. Running a
   history-conditioned policy at the raw physics rate would compress its 160 ms memory window to 40 ms.

Observation normalization is handled inside the TorchScript checkpoint itself, so nothing is applied here.
"""

import isaaclab.envs.mdp as isaac_mdp
import torch
from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from torch import Tensor

from mdp.actions.go2_locomotion_policy_v2 import GO2_SIM2REAL_CFG, OBS_TERM_DIMS, get_locomotion_policy
from mdp.observations.helper_functions import get_goal_relative_position
from navigation_env.NavigationEnv import NavEnv


class _Go2Sim2RealPolicyAction(ActionTerm):
    cfg: "Go2Sim2RealPolicyActionCfg"

    def __init__(self, cfg: "Go2Sim2RealPolicyActionCfg", env: NavEnv):
        super().__init__(cfg, env)
        self._robot_cfg = SceneEntityCfg(name=cfg.asset_name)
        robot: Articulation = self._env.scene[cfg.asset_name]

        self._history_length = int(GO2_SIM2REAL_CFG["history_length"])
        self._frame_dim = int(GO2_SIM2REAL_CFG["frame_dim"])

        # Tick the locomotion policy at its training rate rather than once per physics step.
        sim_dt = getattr(env, "physics_dt", None) or env.cfg.sim.dt
        self._policy_decimation = max(1, int(round(cfg.control_dt / sim_dt)))
        self._sim_step_counter = 0

        self._raw_actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._velocity_command = torch.zeros_like(self._raw_actions)
        self._cached_nominal: Tensor | float = 0.0

        # (num_envs, history_length, frame_dim), oldest frame at index 0 and newest at index -1
        self._history = torch.zeros((self.num_envs, self._history_length, self._frame_dim), device=self.device)
        self._needs_history_fill = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        self._last_policy_action = torch.zeros((self.num_envs, int(GO2_SIM2REAL_CFG["n_actions"])), device=self.device)
        self._joint_target = robot.data.default_joint_pos.torch.clone()

        self._command_limits = torch.tensor(cfg.command_limits, device=self.device)
        self.controller = get_locomotion_policy(checkpoint=cfg.checkpoint, device=self.device)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> Tensor:
        """The base velocity command actually handed to the locomotion policy (nominal added, then clipped)."""
        return self._velocity_command

    def process_actions(self, actions: Tensor):
        self._raw_actions = actions.clone()
        self._cached_nominal = self.nominal_action()
        self._velocity_command = torch.clamp(
            self._raw_actions + self._cached_nominal, -self._command_limits, self._command_limits
        )
        # Re-align the tick schedule so the fresh command is consumed on the first sim step of this env step.
        self._sim_step_counter = 0

    def reset(self, env_ids: Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)

        robot: Articulation = self._env.scene[self.cfg.asset_name]
        self._raw_actions[env_ids] = 0.0
        self._velocity_command[env_ids] = 0.0
        self._last_policy_action[env_ids] = 0.0
        self._history[env_ids] = 0.0
        self._needs_history_fill[env_ids] = True
        self._joint_target[env_ids] = robot.data.default_joint_pos.torch[env_ids]

    def _build_frame(self) -> Tensor:
        """One 45D observation frame, in the term order of the training config's policy group."""
        return torch.cat(
            [
                isaac_mdp.base_ang_vel(self._env, asset_cfg=self._robot_cfg),
                isaac_mdp.projected_gravity(self._env, asset_cfg=self._robot_cfg),
                self._velocity_command,
                # `joint_pos_rel_calibrated` in training only adds a randomized calibration offset, which is zero here
                isaac_mdp.joint_pos_rel(self._env, asset_cfg=self._robot_cfg),
                isaac_mdp.joint_vel_rel(self._env, asset_cfg=self._robot_cfg),
                self._last_policy_action,
            ],
            dim=-1,
        )

    def _push_history(self, frame: Tensor) -> None:
        """Append a frame, mirroring :class:`isaaclab.utils.buffers.CircularBuffer`.

        On the first push after a reset the buffer is filled with that frame instead of being padded with zeros, exactly
        as ``CircularBuffer.append`` does for its ``is_first_push`` batch entries.
        """
        if bool(self._needs_history_fill.any()):
            fill = self._needs_history_fill
            self._history[fill] = frame[fill].unsqueeze(1)
            self._needs_history_fill[:] = False

        self._history = torch.roll(self._history, shifts=-1, dims=1)
        self._history[:, -1] = frame

    def _flatten_history(self) -> Tensor:
        """Flatten to the 360D vector the policy expects: per term, each term's own history contiguous."""
        chunks = []
        offset = 0
        for _, dim in OBS_TERM_DIMS:
            chunks.append(self._history[:, :, offset : offset + dim].reshape(self.num_envs, -1))
            offset += dim
        return torch.cat(chunks, dim=-1)

    def _run_locomotion_policy(self) -> None:
        with torch.no_grad():
            self._push_history(self._build_frame())
            self._last_policy_action = self.controller(self._flatten_history())

        robot: Articulation = self._env.scene[self.cfg.asset_name]
        self._joint_target = robot.data.default_joint_pos.torch + self._last_policy_action * self.cfg.action_scale

    def apply_actions(self):
        if self._sim_step_counter % self._policy_decimation == 0:
            self._run_locomotion_policy()
        self._sim_step_counter += 1

        robot: Articulation = self._env.scene[self.cfg.asset_name]
        robot.set_joint_position_target(self._joint_target)

    def nominal_action(self) -> Tensor | float:
        if not hasattr(self._env, "nominal_weight") or self._env.nominal_weight <= 0.01:
            return 0.0
        else:
            weight = self._env.nominal_weight

        return weight * get_goal_relative_position(self._env, local_goal_points_forward=4)


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
    #: Clipping for (v_x, v_y, w_z); defaults to the command ranges the policy was trained on.
    command_limits: tuple[float, float, float] = GO2_SIM2REAL_CFG["command_limits"]


@configclass
class ActionsCfg:
    # Term name kept identical to `mdp.actions.actions_config.ActionsCfg` so that anything referencing the action term
    # by name (action-rate rewards, `last_action` observations) keeps working unchanged.
    mpc_cmd = Go2Sim2RealPolicyActionCfg()

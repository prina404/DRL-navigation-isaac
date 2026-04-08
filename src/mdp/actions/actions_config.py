from typing import Callable

import isaaclab.envs.mdp as isaac_mdp
import torch
from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from torch import Tensor

from mdp.actions.go2_locomotion_policy import get_locomotion_policy
from mdp.observations.helper_functions import get_goal_relative_position
from navigation_env.NavigationEnv import NavEnv


class _Go2MPCPolicyAction(ActionTerm):
    def __init__(self, cfg, env: NavEnv):
        super().__init__(cfg, env)
        self._robot_cfg = SceneEntityCfg(name=cfg.asset_name)
        self.null_cmd = torch.zeros((self.num_envs, 3), device=self.device)
        self._last_action_received = self.null_cmd.clone()
        self._last_joint_action = None

        self.controller = get_locomotion_policy()

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> Tensor:
        return self._last_action_received

    @property
    def processed_actions(self) -> Tensor:
        return self._last_action_received

    def process_actions(self, actions: Tensor):
        self._last_action_received = actions.clone()

    def reset(self, env_ids: Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self._last_action_received[env_ids] = 0.0
        if self._last_joint_action is not None:
            self._last_joint_action[env_ids] = 0.0

    def _build_low_level_obs(self) -> Tensor:
        base_lin_vel = isaac_mdp.base_lin_vel(self._env, asset_cfg=self._robot_cfg)
        base_ang_vel = isaac_mdp.base_ang_vel(self._env, asset_cfg=self._robot_cfg)
        projected_gravity = isaac_mdp.projected_gravity(self._env, asset_cfg=self._robot_cfg)

        base_vel_cmd = self._last_action_received + self.nominal_action()
        # logger.debug(f"Nominal action: {self.nominal_action()[0]}")
        # logger.debug(f"Policy command: {self._last_action_received[0]}")
        # logger.debug(f"Final cmd (nominal + action): {base_vel_cmd[0]}")

        joint_pos = isaac_mdp.joint_pos_rel(self._env, asset_cfg=self._robot_cfg)
        joint_vel = isaac_mdp.joint_vel_rel(self._env, asset_cfg=self._robot_cfg)

        if self._last_joint_action is None:
            self._last_joint_action = torch.zeros_like(joint_pos)

        return torch.cat(
            [
                base_lin_vel,
                base_ang_vel,
                projected_gravity,
                base_vel_cmd,
                joint_pos,
                joint_vel,
                self._last_joint_action,
            ],
            dim=-1,
        )

    def apply_actions(self):
        # Run low-level policy every sim step by default
        with torch.no_grad():
            mpc_obs = self._build_low_level_obs()
            mpc_obs = {"default": mpc_obs}  # Wrap in dict for ActorCritic
            self._last_joint_action = self.controller(mpc_obs)

        robot: Articulation = self._env.scene[self.cfg.asset_name]
        q0 = robot.data.default_joint_pos
        q_des = q0 + self._last_joint_action * 0.25
        robot.set_joint_position_target(q_des)

    def nominal_action(self) -> Tensor:
        if not hasattr(self._env, "nominal_weight"):
            weight = 0.0
        else:
            weight = self._env.nominal_weight

        return weight * get_goal_relative_position(self._env, local_goal_points_forward=4)


@configclass
class ActionsCfg:
    mpc_cmd = ActionTermCfg(
        class_type=_Go2MPCPolicyAction,
        asset_name="robot",
    )

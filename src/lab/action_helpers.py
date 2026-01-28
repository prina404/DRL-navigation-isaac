import isaaclab.envs.mdp as mdp
import torch
from isaaclab.managers import ActionTerm, SceneEntityCfg
from rsl_rl.modules import ActorCritic


class Go2MPCPolicyAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._robot_cfg = SceneEntityCfg(name=cfg.asset_name)
        self._cmd = torch.zeros((self.num_envs, 3), device=self.device)
        self._last_joint_action = None
        self._decimation = 1  # Run MPC every 1*dt seconds

        self.mpc: ActorCritic = getattr(cfg, "mpc_policy", None)  # type: ignore
        if self.mpc is None:
            raise ValueError("MPC policy must be provided in the environment config.")
        self.physics_step_counter = 0

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._cmd

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._cmd

    def process_actions(self, actions: torch.Tensor):
        self._cmd[:] = actions

    def _build_low_level_obs(self) -> torch.Tensor:
        base_lin_vel = mdp.base_lin_vel(self._env, asset_cfg=self._robot_cfg)
        base_ang_vel = mdp.base_ang_vel(self._env, asset_cfg=self._robot_cfg)
        projected_gravity = mdp.projected_gravity(self._env, asset_cfg=self._robot_cfg)

        base_vel_cmd = self._cmd  # + torch.tensor([1, 0, 0.1], device=self._cmd.device)

        joint_pos = mdp.joint_pos_rel(self._env, asset_cfg=self._robot_cfg)
        joint_vel = mdp.joint_vel_rel(self._env, asset_cfg=self._robot_cfg)

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
        if self.physics_step_counter % self._decimation == 0 or self._last_joint_action is None:
            with torch.no_grad():
                mpc_obs = self._build_low_level_obs()
                mpc_obs = {"default": mpc_obs}  # Wrap in dict for ActorCritic
                self._last_joint_action = self.mpc(mpc_obs)

        self.physics_step_counter += 1
        mpc_action = self._last_joint_action

        robot = self._env.scene[self.cfg.asset_name]
        q0 = robot.data.default_joint_pos
        q_des = q0 + mpc_action * 0.25
        robot.set_joint_position_target(q_des)

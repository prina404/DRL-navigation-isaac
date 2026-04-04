from typing import Callable

import isaaclab.envs.mdp as mdp
import torch
import warp as wp
from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.managers import ActionTerm, SceneEntityCfg
from torch import Tensor

from lab.helpers.observation_helpers import get_goal_relative_position
from lab.MyEnv import MyEnv
from go2.go2_control_policy import Go2CtrlPolicy
from loguru import logger

class Go2MPCPolicyAction(ActionTerm):
    def __init__(self, cfg, env: MyEnv):
        super().__init__(cfg, env)
        self._robot_cfg = SceneEntityCfg(name=cfg.asset_name)
        self.null_cmd = torch.zeros((self.num_envs, 3), device=self.device)
        self._last_action_received = self.null_cmd.clone()
        self._last_joint_action = None

        self.mpc: Callable[Tensor, Tensor] = getattr(cfg, "mpc_policy", None)  # type: ignore
        if self.mpc is None:
            raise ValueError("MPC policy must be provided in the environment config.")
        self.use_jax_policy = isinstance(self.mpc, Go2CtrlPolicy)

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

    def _build_obs_default(self) -> Tensor:
        base_lin_vel = mdp.base_lin_vel(self._env, asset_cfg=self._robot_cfg)
        base_ang_vel = mdp.base_ang_vel(self._env, asset_cfg=self._robot_cfg)
        projected_gravity = mdp.projected_gravity(self._env, asset_cfg=self._robot_cfg)

        base_vel_cmd = self.nominal_action() # + self._last_action_received

        joint_pos = mdp.joint_pos_rel(self._env, asset_cfg=self._robot_cfg)
        joint_vel = mdp.joint_vel_rel(self._env, asset_cfg=self._robot_cfg)

        if self._last_joint_action is None:
            self._last_joint_action = torch.zeros_like(joint_pos)

        output = torch.cat(
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

        return {"default": output}  # Wrap in dict for ActorCritic compatibility
    
    def man_tensor(self):
        qpos = torch.tensor([0.0, 0.02, -0.029, 0.004, -0.03, 0.028, 0.002, -0.019, -0.045, 0.001, 0.009, 0.034, 0.0], device=self.device).unsqueeze(0)
        qvel = torch.tensor([-0.0, -0.003, 0.008, 0.0, 0.004, -0.01, 0.0, -0.002, 0.007, -0.0, 0.002, -0.006, 0.0], device=self.device).unsqueeze(0)
        previous_action = torch.zeros((13,), device=self.device).unsqueeze(0)
        ang_vel = torch.zeros((1,   3), device=self.device)
        cmd_vel = torch.zeros((1, 3), device=self.device)
        projected_gravity = torch.tensor([-0.019, 0.005, -0.999], device=self.device).unsqueeze(0)
        
        jnts = torch.stack([qpos, qvel, previous_action], dim=1)
        jnts = torch.transpose(jnts, -2, -1)  # (B, 3, 13) -> (B, 13, 3)
        jnts = torch.flatten(jnts, start_dim=1)  # (B, 39)
        return torch.cat([
            jnts,
            ang_vel,
            cmd_vel,
            projected_gravity,
        ], dim=-1)

    def apply_actions(self):
        # Run low-level policy every sim step by default
        with torch.no_grad():
            mpc_obs = self._build_obs_jax() if self.use_jax_policy else self._build_obs_default()
            self._last_joint_action = self.mpc(mpc_obs)

        if self.use_jax_policy:
            # logger.debug(f"action: {self._last_joint_action[0].cpu().numpy()}")
            # raise Exception
            self._last_joint_action = self._last_joint_action[:, self.jax_to_isaac_mask]  # Reorder back to sim order
            
        robot: Articulation = self._env.scene[self.cfg.asset_name]
        q0 = wp.to_torch(robot.data.default_joint_pos)
        q_des = q0 + self._last_joint_action * 0.25
        robot.set_joint_position_target(q_des)

    def nominal_action(self) -> Tensor:
        if not hasattr(self._env, "nominal_weight"):
            weight = 0.0
        else:
            weight = self._env.nominal_weight

        return weight * get_goal_relative_position(self._env, local_goal_points_forward=4)


    def _build_obs_jax(self) -> Tensor:
        # terms and normalization expected by the original JAX policy
        joint_pos_rel = mdp.joint_pos_rel(self._env, asset_cfg=self._robot_cfg)

        joint_pos = joint_pos_rel[:, self.isaac_to_jax_mask] / torch.pi  # Reorder joints to match policy's expected order

        joint_vel = mdp.joint_vel(self._env, asset_cfg=self._robot_cfg) / 25.
        joint_vel = joint_vel[:, self.isaac_to_jax_mask]  

        if self._last_joint_action is None:
            self._last_joint_action = torch.zeros_like(joint_pos)

        prev_action = self._last_joint_action / torch.pi
        prev_action = prev_action[:, self.isaac_to_jax_mask]

        B = joint_pos.shape[0]
        joints = torch.stack([joint_pos, joint_vel, prev_action], dim=1)
        joints = torch.transpose(joints, -2, -1)  # (B, 3, 12) -> (B, 12, 3)
        zero_joint = torch.zeros(B, 1, 3, device=self.device, dtype=joints.dtype)
        joints = torch.cat([joints, zero_joint], dim=1)  # Add zero joint for the spine -> (B, 13, 3)
        joints_flat = torch.flatten(joints, start_dim=1)  # (B, 39)

        ang_vel = mdp.base_ang_vel(self._env, asset_cfg=self._robot_cfg) / 10.0
        cmd_vel = self._last_action_received + self.nominal_action() # + self._last_action_received

        projected_gravity = mdp.projected_gravity(self._env, asset_cfg=self._robot_cfg)

        # logger.debug(f"{self._env.scene["robot"].joint_names}")
        # logger.debug(f"qpos: {joint_pos[0].cpu().numpy()}")
        # logger.debug(f"qvel: {joint_vel[0].cpu().numpy()}")
        # logger.debug(f"prev_action: {prev_action[0].cpu().numpy()}")
        # logger.debug(f"ang_vel: {ang_vel[0].cpu().numpy()}")
        # logger.debug(f"proj_gravity: {projected_gravity[0].cpu().tolist()}")

        # logger.debug(f"joint_pos: {mdp.joint_pos(self._env, asset_cfg=self._robot_cfg)[:,self.isaac_to_jax_mask][0].cpu().numpy()}")
        # logger.debug(f"joint_vel : {mdp.joint_vel(self._env, asset_cfg=self._robot_cfg)[:,self.isaac_to_jax_mask][0].cpu().numpy()}")
        # logger.debug(f"ang_vel : {mdp.base_ang_vel(self._env, asset_cfg=self._robot_cfg)[0].cpu().numpy()}")

        return torch.cat(
            [
                joints_flat,
                ang_vel,
                cmd_vel,
                projected_gravity,
            ],
            dim=-1,
        )

    @property
    def isaac_to_jax_mask(self) -> Tensor:
        # This mask maps isaac -> policy joint order
        if hasattr(self, "_reorder_mask"):
            return self._reorder_mask

        # Policy expects joints in this order:
        jax_joint_names = self.policy_joint_order
        isaac_joint_names = self._env.scene["robot"].joint_names

        self._reorder_mask = self._get_mask(from_=isaac_joint_names, to_=jax_joint_names)
        return self._reorder_mask

    @property
    def jax_to_isaac_mask(self) -> Tensor:
        # This mask maps policy -> isaac joint order
        if hasattr(self, "_reorder_mask"):
            return self._reorder_mask

        # Policy expects joints in this order:
        jax_joint_names = self.policy_joint_order
        isaac_joint_names = self._env.scene["robot"].joint_names

        self._reorder_mask = self._get_mask(from_=jax_joint_names, to_=isaac_joint_names)
        return self._reorder_mask

    def _get_mask(self, from_: list[str], to_: list[str]) -> Tensor:
        return torch.tensor(
            [from_.index(joint) for joint in to_],
            device=self.device,
        )
        

    @property
    def policy_joint_order(self):
        joint_order = [
            "FL_hip_joint","FL_thigh_joint","FL_calf_joint",
            "FR_hip_joint","FR_thigh_joint","FR_calf_joint",
            "RL_hip_joint","RL_thigh_joint","RL_calf_joint",
            "RR_hip_joint","RR_thigh_joint","RR_calf_joint",
        ]

        return joint_order

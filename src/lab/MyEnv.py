import random
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import scipy.ndimage as sp
import torch
import yaml
from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_from_euler_xyz
from loguru import logger

import preprocessing.voronoi as voronoi
import pyastar2d
from cfg.CFG import SCENE_USD_PATH
from lab.managers.camera_manager import CameraManager
from lab.managers.path_manager import PathManager


class MyEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        camera_controller = ViewportCameraController(self, cfg=ViewerCfg(origin_type="world"))
        self.camera_manager = CameraManager(
            camera_controller, camera_relative_pos=torch.tensor([-1, 0.0, 1.0]), camera_lookat=torch.tensor([1.8, 0.0, -0.8])
        )

        self.path_manager = PathManager(scene_path=SCENE_USD_PATH, num_envs=self.num_envs, device=self.device)

        # Action buffer with the past 5 actions (Δx, Δy, Δtheta)
        self._action_buffer = torch.zeros((self.num_envs, 5, 3), dtype=torch.float32, device=self.device)

        # Store past 10 robot linear + angular velocity (x,y,theta)
        self._velocity_buffer = torch.zeros((self.num_envs, 10, 3), dtype=torch.float32, device=self.device)
        self._old_yaw = torch.zeros((self.num_envs), dtype=torch.float32, device=self.device)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        retVal = super().step(action)

        # Update action buffer
        self._action_buffer = torch.roll(self._action_buffer, shifts=1, dims=1)
        self._action_buffer[:, 0, :] = action.to(self.device)

        ## Update linear velocity
        robot: Articulation = self.scene["robot"]
        self._velocity_buffer = torch.roll(self._velocity_buffer, shifts=1, dims=1)
        # store only x, y linear velocity
        self._velocity_buffer[:, 0, :2] = robot.data.root_com_lin_vel_w[:, :2].to(self.device)

        ## Angular velocity
        _, _, new_yaw = euler_xyz_from_quat(robot.data.root_com_quat_w)
        theta = (new_yaw - self._old_yaw) / getattr(self.sim, "dt", 0.005)
        self._velocity_buffer[:, 0, 2] = theta
        self._old_yaw = new_yaw

        self.update_follow_camera()

        return retVal

    def teleport_robots(self, env_ids: torch.Tensor) -> None:
        ## set cartesian position, quaternion orientation in (w, x, y, z), and linear and angular velocity
        ## reset positions only for specified env indices
        robot: Articulation = self.scene["robot"]

        node_ids = self.path_manager.sample_node_ids(env_ids.shape[0])
        state = self.path_manager.compute_robot_teleport_state(
            env_ids, node_ids, self.scene.env_origins, robot.data.default_root_state[0, 2]
        )
        self.path_manager.set_start_nodes(env_ids, node_ids)

        robot.write_root_com_state_to_sim(state, env_ids)

    def compute_goals_on_reset(self, env_ids: torch.Tensor) -> None:
        self.path_manager.sample_goals(env_ids)
        self.manual_replan(env_ids)

    def manual_replan(self, env_ids: torch.Tensor) -> None:
        robot = self.scene["robot"]
        self.path_manager.compute_global_plan(env_ids, robot.data.root_com_pos_w[env_ids], self.scene.env_origins[env_ids])

    def update_lidar_buffer(self, lidar_obs: torch.Tensor) -> None:
        # store only the last lidar scan
        pass

    def update_follow_camera(self):
        robot_pos = self.scene["robot"].data.root_pos_w[0]
        robot_quat = self.scene["robot"].data.root_quat_w[0]

        self.camera_manager.update(robot_pos, robot_quat, smoothing_factor=0.3)

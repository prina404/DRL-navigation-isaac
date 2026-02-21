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


class MyEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.camera_controller = ViewportCameraController(self, cfg=ViewerCfg(origin_type="world"))
        self.old_eye_pos = None
        self.old_lookat_pos = None
        # Camera and lookat settings for video recording
        self.camera_offset = torch.tensor([-1, 0.0, 1.0], device=self.device)
        self.lookat_offset = torch.tensor([1.8, 0.0, -0.8], device=self.device)

        map_png = str(SCENE_USD_PATH.parent / SCENE_USD_PATH.stem) + "_map.png"
        map_yaml = str(SCENE_USD_PATH.parent / SCENE_USD_PATH.stem) + "_map.yaml"
        with open(map_yaml, "r") as f:
            self.img_meta = yaml.safe_load(f)

        self.graph, coords = voronoi.compute_voronoi_graph(
            map_path=map_png,
            blur_radius=15,
            min_node_distance=15,
            plot_graph=True,
            filepath=str(SCENE_USD_PATH),
        )

        self.map_img = cv2.imread(map_png, cv2.IMREAD_GRAYSCALE)
        self._costmap = self._create_inflated_costmap(inflation_scale=2)
        # store the pixel coordinates of each node (indexed by [row][col])
        self.nodes = torch.Tensor([coords[idx] for idx in self.graph.nodes]).to(self.device)

        # Tensors that map env_id -> start node_id and goal node_id
        self.start_pos_ids = torch.zeros((self.num_envs), dtype=torch.int64, device=self.device)
        self.goal_ids = torch.zeros((self.num_envs), dtype=torch.int64, device=self.device)
        self.goal_pos = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self._path_tensors: list[torch.Tensor] = [None] * self.num_envs  # list of (num_path_points, 2) tensors

        # Action buffer with the past 5 actions (Δx, Δy, Δtheta)
        self._action_buffer = torch.zeros((self.num_envs, 5, 3), dtype=torch.float32, device=self.device)

        # Store past 10 robot linear + angular velocity (x,y,theta)
        self._velocity_buffer = torch.zeros((self.num_envs, 10, 3), dtype=torch.float32, device=self.device)
        self._old_yaw = torch.zeros((self.num_envs), dtype=torch.float32, device=self.device)
        self._mid_obstacle_jitter_radius = 0.3
        self._mid_obstacle_height = 1.0

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        retVal = super().step(action)

        # Update action buffer
        self._action_buffer = torch.roll(self._action_buffer, shifts=1, dims=1)
        self._action_buffer[:, 0, :] = action.to(self.device)

        ## Update linear velocity
        robot: Articulation = self.scene["unitree_go2"]
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

    def _create_inflated_costmap(self, inflation_scale: int) -> np.ndarray:
        binary_img = self.map_img == 0  # 1= occupied, 0=free/unknown
        dist_transform = sp.distance_transform_edt(~binary_img)
        costmap = np.clip(np.max(dist_transform) - (dist_transform * inflation_scale), 0, 255).astype(np.float32)
        max_value = costmap.max()
        costmap[costmap >= max_value] = float("inf")
        return costmap + 1.0  # min_cost = 1.0 for A*

    def _map_to_local(self, map_coords: torch.Tensor) -> torch.Tensor:
        # node coords are tensor: [[row, col],
        #  [row, col], ...]
        # in isaac frame, y directions are ok the same as in pixel space, but x are inverted
        map_coords = map_coords.clone()
        _, W = self.map_img.shape
        map_coords[:, 1] = W - map_coords[:, 1]  # flip x axis
        map_coords = map_coords[:, [1, 0]]  # swap to (x, y)

        n = map_coords.shape[0]
        x, y = self.img_meta["origin"][:2]

        origin = torch.zeros((n, 2), device=self.device)
        origin[:] = torch.tensor([x, y], device=self.device)  # broadcast origin

        scaled = map_coords * self.img_meta["resolution"]  # (N, 2)
        local_coords = scaled + origin  # (N, 2)
        return local_coords

    def _local_to_map(self, local_coords: torch.Tensor) -> torch.Tensor:
        # local coords are tensor: [[x, y],
        #  [x, y], ...]
        if local_coords.size(1) > 2:
            local_coords = local_coords[:, :2]  # only consider x, y

        local_coords = local_coords.clone()
        n = len(local_coords)
        x, y = self.img_meta["origin"][:2]

        origin = torch.zeros((n, 2), device=self.device)
        origin[:] = torch.tensor([x, y], device=self.device)  # broadcast origin

        local_coords = local_coords - origin  # (N, 2)
        scaled = local_coords / self.img_meta["resolution"]  # (N, 2)

        _, W = self.map_img.shape
        scaled[:, 0] = W - scaled[:, 0]  # flip x axis
        map_coords = scaled[:, [1, 0]]  # swap to (row, col)
        return map_coords

    def sample_node_ids(self, num_samples: int) -> torch.Tensor:
        """Returns (num_samples,) tensor of node_ids"""
        return torch.randint(0, self.nodes.shape[0], (num_samples,), device=self.device)

    def teleport_robots(self, env_ids: torch.Tensor, node_ids: torch.Tensor) -> None:
        ## set cartesian position, quaternion orientation in (w, x, y, z), and linear and angular velocity
        ## reset positions only for specified env indices

        robot: Articulation = self.scene["unitree_go2"]
        num_pos = node_ids.shape[0]

        positions = self.nodes[node_ids]  # map coordinates of node_ids
        positions = self._map_to_local(positions)  # (num_pos, 2)
        z_col = torch.zeros((num_pos, 1), device=self.device) + robot.data.default_root_state[0, 2]  # set z to default
        pos_local = torch.cat([positions, z_col], dim=-1)  # (num_pos, 3)

        # swap x and y to match world frame

        # TODO: need to convert local coordinates to world coordinates based on env origin
        pos_w = pos_local + self.scene.env_origins[env_ids]

        yaw = torch.rand((num_pos), device=self.device) * 2 * torch.pi - torch.pi
        roll = torch.zeros_like(yaw)
        pitch = torch.zeros_like(yaw)
        quats = quat_from_euler_xyz(roll, pitch, yaw)

        zero_vel = torch.zeros((num_pos, 6), device=self.device)  # lin + angular vel

        state = torch.cat([pos_w, quats, zero_vel], dim=-1)  # (num_pos, 13)

        robot.write_root_com_state_to_sim(state, env_ids)
        self.start_pos_ids[env_ids] = node_ids

    def compute_goals_on_reset(self, env_ids: torch.Tensor) -> None:
        # TODO: add distance_based sampling. For the time being,
        # just select a random node and navigate towards it
        for id in env_ids:
            current_node = self.start_pos_ids[id]
            random_node = random.randint(0, self.nodes.shape[0] - 1)
            while random_node == current_node or random_node in self.graph.neighbors(current_node.item()):
                random_node = random.randint(0, self.nodes.shape[0] - 1)

            self.goal_ids[id] = random_node
            goal_pos_map = self.nodes[random_node].clone()
            self.goal_pos[id] = self._map_to_local(goal_pos_map.unsqueeze(0))[0]

        self.compute_global_plan(env_ids)

    def compute_global_plan(self, env_ids: torch.Tensor) -> None:
        robot: Articulation = self.scene["unitree_go2"]
        robot_pos_w = robot.data.root_com_pos_w[env_ids]
        robot_pos_local = robot_pos_w - self.scene.env_origins[env_ids]  # convert to local coordinates
        robot_pos_map = self._local_to_map(robot_pos_local[:, :2])  # (num_envs, 2)

        paths = []
        with ThreadPoolExecutor() as executor:  # compute paths in parallel, this should yield 5-10x speedup
            for idx, env_id in enumerate(env_ids):
                goal_node_id = self.goal_ids[env_id]
                goal_pos_map = self.nodes[goal_node_id]

                r1, c1 = robot_pos_map[idx].int().cpu().numpy()
                r2, c2 = goal_pos_map.int().cpu().numpy()
                paths.append(
                    executor.submit(
                        pyastar2d.astar_path,
                        self._costmap,
                        (r1, c1),
                        (r2, c2),
                        allow_diagonal=True,
                    )
                )

        for idx, env_id in enumerate(env_ids):
            path = paths[idx].result()
            goal_pos_map = self.nodes[self.goal_ids[env_id]]
            if path is None or len(path) <= 1:
                logger.warning(f"No path found for robot {env_id}! Returning empty path tensor")
                self._path_tensors[env_id] = goal_pos_map.unsqueeze(0)  # just return the goal as the path
                continue

            point_dist = 0.3  # meters
            path_subsampled = path[:: int(point_dist / self.img_meta["resolution"])]

            path_tensor = torch.zeros((len(path_subsampled) + 1, 2), device=self.device)
            path_tensor[:-1] = torch.tensor(path_subsampled, device=self.device)
            path_tensor[-1] = goal_pos_map  # ensure goal is included

            world_path = self._map_to_local(path_tensor)  # (num_path_points, 2)
            self._path_tensors[env_id] = world_path

    def update_follow_camera(self):
        robot_pos = self.scene["unitree_go2"].data.root_pos_w[0]
        robot_quat = self.scene["unitree_go2"].data.root_quat_w[0]

        camera_offset_w = quat_apply(robot_quat, self.camera_offset)
        lookat_offset_w = quat_apply(robot_quat, self.lookat_offset)

        if self.old_eye_pos is None:
            self.old_eye_pos = robot_pos + camera_offset_w
            self.old_lookat_pos = robot_pos + lookat_offset_w
        else:  # smooth camera movement by interpolating between old and new positions
            alpha = 0.3  # smoothing factor
            new_eye_pos = robot_pos + camera_offset_w
            new_lookat_pos = robot_pos + lookat_offset_w

            smoothed_eye_pos = (1 - alpha) * self.old_eye_pos + alpha * new_eye_pos
            smoothed_lookat_pos = (1 - alpha) * self.old_lookat_pos + alpha * new_lookat_pos

            self.camera_controller.update_view_location(
                eye=smoothed_eye_pos.cpu().numpy(),
                lookat=smoothed_lookat_pos.cpu().numpy(),
            )

            self.old_eye_pos = smoothed_eye_pos
            self.old_lookat_pos = smoothed_lookat_pos

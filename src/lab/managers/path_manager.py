import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage as sp
import torch
import yaml
from isaaclab.utils.math import quat_from_euler_xyz
from loguru import logger

import pyastar2d
from preprocessing import voronoi


class PathManager:
    def __init__(
        self,
        scene_path: str | Path,
        num_envs: int,
        device: torch.device,
    ):
        self.device = device
        self.scene_usd_path = Path(scene_path)

        map_png = str(self.scene_usd_path.parent / self.scene_usd_path.stem) + "_map.png"
        map_yaml = str(self.scene_usd_path.parent / self.scene_usd_path.stem) + "_map.yaml"

        with open(map_yaml, "r") as f:
            self.img_meta = yaml.safe_load(f)

        self.graph, coords = voronoi.compute_voronoi_graph(
            map_path=map_png,
            blur_radius=15,
            min_node_distance=15,
            plot_graph=True,
            filepath=str(self.scene_usd_path),
        )

        self.map_img = cv2.imread(map_png, cv2.IMREAD_GRAYSCALE)
        if self.map_img is None:
            raise FileNotFoundError(f"Failed to read map image: {map_png}")

        self.costmap = self._create_inflated_costmap(inflation_scale=2)
        self.nodes = torch.Tensor([coords[idx] for idx in self.graph.nodes]).to(device)

        self.start_node_idx = torch.zeros((num_envs,), dtype=torch.int64, device=device)
        self.goal_node_idx = torch.zeros((num_envs,), dtype=torch.int64, device=device)
        self.goal_pos_local = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
        self.path_tensors: list[torch.Tensor | None] = [None] * num_envs  # each: (K,2) in local (x,y)

        self._executor = ThreadPoolExecutor()

    def _create_inflated_costmap(self, inflation_scale: int) -> torch.Tensor:
        binary_img = self.map_img == 0  # occupied=True
        dist_transform = sp.distance_transform_edt(~binary_img)
        costmap = np.clip(np.max(dist_transform) - (dist_transform * inflation_scale), 0, 255).astype(np.float32)
        max_value = costmap.max()
        costmap[costmap >= max_value] = float("inf")
        return costmap + 1.0  # min_cost = 1.0 for A*

    def map_to_local_coords(self, map_coords: torch.Tensor) -> torch.Tensor:
        map_coords = map_coords.clone()
        _, W = self.map_img.shape
        map_coords[:, 1] = W - map_coords[:, 1]  # flip x axis
        map_xy = map_coords[:, [1, 0]]  # swap from (row,col) to (x, y)

        x, y = self.img_meta["origin"][:2]

        origin = torch.tensor([x, y], device=self.device)

        scaled = map_xy * self.img_meta["resolution"]  # (N, 2)
        return scaled + origin  # (N, 2)

    def local_to_map_coords(self, local_coords: torch.Tensor) -> torch.Tensor:
        if local_coords.size(1) > 2:
            local_coords = local_coords[:, :2]  # only consider x, y

        local_xy = local_coords.clone()

        x, y = self.img_meta["origin"][:2]
        origin = torch.tensor([x, y], device=self.device)

        scaled = (local_xy - origin) / self.img_meta["resolution"]

        _, W = self.map_img.shape
        scaled[:, 0] = W - scaled[:, 0]  # flip x axis
        map_coords = scaled[:, [1, 0]]  # swap to (row, col)
        return map_coords

    def sample_node_ids(self, num_samples: int) -> torch.Tensor:
        return torch.randint(0, self.nodes.shape[0], (num_samples,), device=self.device)

    def set_start_nodes(self, env_ids: torch.Tensor, node_ids: torch.Tensor) -> None:
        # set the node indices to which I have teleported the robots
        self.start_node_idx[env_ids] = node_ids

    def sample_goals(self, env_ids: torch.Tensor) -> None:
        # TODO: add distance_based sampling. For the time being,
        # just select a random node and navigate towards it
        for id in env_ids:
            current_node = self.start_node_idx[id]
            random_node = random.randint(0, self.nodes.shape[0] - 1)
            while random_node == current_node or random_node in self.graph.neighbors(current_node.item()):
                random_node = random.randint(0, self.nodes.shape[0] - 1)

            self.goal_node_idx[id] = random_node
            goal_pos_map = self.nodes[random_node].clone()
            self.goal_pos_local[id] = self.map_to_local_coords(goal_pos_map.unsqueeze(0))[0]

    def compute_global_plan(
        self, env_ids: torch.Tensor, robot_pos_w: torch.Tensor, env_origins: torch.Tensor, point_dist: float = 0.3
    ) -> None:
        assert (
            robot_pos_w.shape[0] == env_ids.shape[0] == env_origins.shape[0]
        ), "Batch size of robot positions, env origins, and env_ids must match"

        robot_pos_local = robot_pos_w - env_origins
        robot_map_coords = self.local_to_map_coords(robot_pos_local)

        paths = []
        for idx, env_id in enumerate(env_ids):
            goal_node_id = self.goal_node_idx[env_id]
            goal_pos_map = self.nodes[goal_node_id]

            r1, c1 = robot_map_coords[idx].int().cpu().numpy()
            r2, c2 = goal_pos_map.int().cpu().numpy()
            paths.append(
                self._executor.submit(
                    pyastar2d.astar_path,
                    self.costmap,
                    (r1, c1),
                    (r2, c2),
                    allow_diagonal=True,
                )
            )

        for idx, env_id in enumerate(env_ids):
            path = paths[idx].result()
            goal_pos_map = self.nodes[self.goal_node_idx[env_id]]
            if path is None or len(path) <= 1:
                logger.warning(f"No path found for robot {env_id}! Returning empty path tensor")
                self.path_tensors[env_id] = goal_pos_map.unsqueeze(0)  # just return the goal as the path
                continue

            point_dist = 0.3  # meters
            path_subsampled = path[:: int(point_dist / self.img_meta["resolution"])]

            path_tensor = torch.zeros((len(path_subsampled) + 1, 2), device=self.device)
            path_tensor[:-1] = torch.tensor(path_subsampled, device=self.device)
            path_tensor[-1] = goal_pos_map  # ensure goal is included

            world_path = self.map_to_local_coords(path_tensor)  # (num_path_points, 2)
            self.path_tensors[env_id] = world_path

    def compute_robot_teleport_state(
        self,
        env_ids: torch.Tensor,
        node_idx: torch.Tensor,
        env_origins: torch.Tensor,
        robot_z_pos: float = 0.0,
    ) -> torch.Tensor:
        map_coords = self.nodes[node_idx]
        local_coords = self.map_to_local_coords(map_coords)

        num_robots = env_ids.shape[0]
        z_col = torch.zeros((num_robots, 1), device=self.device) + robot_z_pos  # set z to default
        pos_local = torch.cat([local_coords, z_col], dim=-1)

        pos_w = pos_local + env_origins[env_ids]
        yaw = torch.rand((num_robots), device=self.device) * 2 * torch.pi - torch.pi
        roll = torch.zeros_like(yaw)
        pitch = torch.zeros_like(yaw)
        quats = quat_from_euler_xyz(roll, pitch, yaw)

        zero_vel = torch.zeros((num_robots, 6), device=self.device)  # lin + angular vel

        state = torch.cat([pos_w, quats, zero_vel], dim=-1)  # (num_robots, 13)
        return state

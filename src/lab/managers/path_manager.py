import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from isaaclab.utils.math import quat_from_euler_xyz
from loguru import logger

import pyastar2d
from lab.managers.map_manager import MapManager
from preprocessing import voronoi


class PathManager:
    def __init__(
        self, scene_path: str | Path, map_mgr: MapManager, num_envs: int, device: torch.device, subgoal_dist: float = 0.3
    ):
        self.device = device
        self.scene_usd_path = Path(scene_path)
        self.map_manager = map_mgr
        self.point_dist = subgoal_dist

        self.graph, coords = voronoi.compute_voronoi_graph(
            map_path=map_mgr.map_img_path,
            blur_radius=15,
            min_node_distance=15,
            plot_graph=True,
            filepath=str(self.scene_usd_path),
        )

        self.nodes = torch.Tensor([coords[idx] for idx in self.graph.nodes]).to(device)

        self.start_node_idx = torch.zeros((num_envs,), dtype=torch.int64, device=device)
        self.goal_node_idx = torch.zeros((num_envs,), dtype=torch.int64, device=device)
        self.goal_pos_local = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
        self.path_tensors: list[torch.Tensor | None] = [None] * num_envs  # each: (K,2) in local (x,y)

        # on each reset store the initial path length for reward normalization
        self.initial_path_length = torch.full((num_envs,), subgoal_dist, dtype=torch.float32, device=device)
        self.current_path_length = torch.full((num_envs,), subgoal_dist, dtype=torch.float32, device=device)

        # A* threads
        self._executor = ThreadPoolExecutor(max_workers=16)

    def map_to_local_coords(self, map_coords: torch.Tensor) -> torch.Tensor:
        return self.map_manager.map_to_local_coords(map_coords)

    def local_to_map_coords(self, local_coords: torch.Tensor) -> torch.Tensor:
        return self.map_manager.local_to_map_coords(local_coords)

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

        # set length buffers to minimum on reset, they will be updated together with the global plan
        self.initial_path_length[env_ids] = self.point_dist
        self.current_path_length[env_ids] = self.point_dist

    def compute_global_plan(self, env_ids: torch.Tensor, robot_pos_w: torch.Tensor, env_origins: torch.Tensor) -> None:
        assert (
            robot_pos_w.shape[0] == env_ids.shape[0] == env_origins.shape[0]
        ), "Batch size of robot positions, env origins, and env_ids must match"

        robot_pos_local = robot_pos_w - env_origins
        robot_map_coords = self.local_to_map_coords(robot_pos_local)

        costmap_cpu = self.map_manager.global_costmap[env_ids].squeeze(1).cpu().numpy()  # (num_envs, H, W)

        paths = []
        for idx, env_id in enumerate(env_ids):
            goal_node_id = self.goal_node_idx[env_id]
            goal_pos_map = self.nodes[goal_node_id]

            r1, c1 = robot_map_coords[idx].int().cpu().numpy()
            r2, c2 = goal_pos_map.int().cpu().numpy()
            paths.append(
                self._executor.submit(
                    pyastar2d.astar_path,
                    costmap_cpu[idx],
                    (r1, c1),
                    (r2, c2),
                    allow_diagonal=True,
                )
            )

        for idx, env_id in enumerate(env_ids):
            path = paths[idx].result()
            r, c = robot_map_coords[idx].int()
            if path is None or len(path) == 0:
                # if I am in a valid cell, but no path is found, I just stand still
                if self._valid_map_coords((r, c), env_id):
                    # logger.info(f"No path found for robot {env_id}! (likely due to dynamic obstacle)")
                    self.path_tensors[env_id] = None
                # if previous path exists keep it (this will happen when robot pos is mapped onto an occupied cell)
                if self.path_tensors[env_id] is None:  # set to current location
                    robot_pos_local = self.map_to_local_coords(robot_map_coords[idx].unsqueeze(0))
                    self.path_tensors[env_id] = robot_pos_local
                continue

            path_subsampled = path[:: int(self.point_dist / self.map_manager.resolution)]

            path_tensor = torch.zeros((len(path_subsampled) + 1, 2), device=self.device)
            path_tensor[:-1] = torch.tensor(path_subsampled, device=self.device)

            goal_pos_map = self.nodes[self.goal_node_idx[env_id]]
            path_tensor[-1] = goal_pos_map  # ensure goal is included

            world_path = self.map_to_local_coords(path_tensor)  # (num_path_points, 2)
            self.path_tensors[env_id] = world_path

            # update path length buffer for reward normalization
            path_len_m = world_path.size(0) * self.point_dist
            self.current_path_length[env_id] = path_len_m

            if path_len_m > self.initial_path_length[env_id]:
                self.initial_path_length[env_id] = path_len_m

    def _valid_map_coords(self, map_coords: tuple, env_id: int) -> bool:
        r, c = map_coords
        return self.map_manager.global_costmap[env_id, 0, r, c] < torch.inf

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

import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml
from isaaclab.utils.math import quat_from_euler_xyz

import pyastar2d
from navigation_env.managers.map_manager import MapManager
from preprocessing import voronoi
from cfg.CFG import NAVPOINTS_FILE, get_map_name


class PathManager:
    def __init__(
        self, scene_path: str | Path, map_mgr: MapManager, num_envs: int, device: torch.device, subgoal_dist: float = 0.3
    ):
        self.device = device
        self.scene_usd_path = Path(scene_path)
        self.map_manager = map_mgr
        self.point_dist = subgoal_dist

        if NAVPOINTS_FILE.exists():
            with open(NAVPOINTS_FILE, "r") as f:
                MAP_NAME = get_map_name()
                self.manual_tasks = yaml.safe_load(f)[MAP_NAME]  # type: ignore

        self.graph, coords = voronoi.compute_voronoi_graph(
            map_path=map_mgr.map_img_path,
            blur_radius=15,
            min_node_distance=15,
            plot_graph=True,
            filepath=str(self.scene_usd_path),
        )

        self.nodes = torch.Tensor([coords[idx] for idx in self.graph.nodes]).to(device)

        # cache local and map coords of start/goal
        self.start_pos_map = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
        self.goal_pos_map = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
        self.goal_pos_local = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)

        self.start_yaw = torch.zeros((num_envs,), dtype=torch.float32, device=device)
        self.goal_yaw = torch.zeros((num_envs,), dtype=torch.float32, device=device)

        self.path_tensors: list[torch.Tensor | None] = [None] * num_envs  # each: (K,2) in local (x,y)

        # Dense, goal-padded mirror of `path_tensors`, so path queries can be batched over envs
        # instead of looping in python (a loop costs one host sync per env, per query).
        # Padding with the goal reproduces the ragged semantics exactly: running off the end of a
        # path yields the goal, which is what the consumers already fall back to.
        self.max_path_points = 256
        self.path_buf = torch.zeros((num_envs, self.max_path_points, 2), dtype=torch.float32, device=device)
        self.path_len = torch.ones((num_envs,), dtype=torch.long, device=device)

        # on each reset store the initial path length for reward normalization
        self.initial_path_length = torch.full((num_envs,), subgoal_dist, dtype=torch.float32, device=device)
        self.current_path_length = torch.full((num_envs,), subgoal_dist, dtype=torch.float32, device=device)

        # store duration of each episode (in steps)
        self.duration_history = torch.zeros((num_envs, 20), dtype=torch.float32, device=device)

        # A* threads
        self._executor = ThreadPoolExecutor(max_workers=64)

    def map_to_local_coords(self, map_coords: torch.Tensor) -> torch.Tensor:
        return self.map_manager.map_to_local_coords(map_coords)

    def local_to_map_coords(self, local_coords: torch.Tensor) -> torch.Tensor:
        return self.map_manager.local_to_map_coords(local_coords)

    def sample_node_ids(self, num_samples: int) -> torch.Tensor:
        return torch.randint(0, self.nodes.shape[0], (num_samples,), device=self.device)

    def sample_nav_task(self, env_ids: torch.Tensor, sample_voronoi_prob: float = 0.25) -> None:
        # TODO: add distance_based sampling. For the time being,
        # just select a random node and navigate towards it
        for id in env_ids:
            if torch.rand(1) < sample_voronoi_prob:
                start_coord_map, goal_coord_map = self.sample_voronoi_task(id)
            else:
                start_coord_map, goal_coord_map = self.sample_manual_task(id)

            self.start_pos_map[id] = start_coord_map
            self.goal_pos_map[id] = goal_coord_map
            self.goal_pos_local[id] = self.map_to_local_coords(goal_coord_map.unsqueeze(0))[0]

        # set length buffers to minimum on reset, they will be updated together with the global plan
        self.initial_path_length[env_ids] = self.point_dist
        self.current_path_length[env_ids] = self.point_dist

    def update_episode_durations(self, env_ids: torch.Tensor, ep_length: torch.Tensor) -> None:
        self.duration_history[env_ids] = self.duration_history[env_ids].roll(shifts=1, dims=1)
        normalized_durations = ep_length.to(torch.float32) / self.initial_path_length[env_ids]  # normalize by initial path length
        self.duration_history[env_ids, 0] = normalized_durations

    def sample_voronoi_task(self, env_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        random_node = random.randint(0, self.nodes.shape[0] - 1)
        current_node = random_node

        while random_node == current_node:
            random_node = random.randint(0, self.nodes.shape[0] - 1)

        start_pos_map = self.nodes[current_node].clone()
        goal_pos_map = self.nodes[random_node].clone()

        # for voronoi tasks we use random yaw
        self.start_yaw[env_id] = torch.rand(1) * 2 * torch.pi - torch.pi
        self.goal_yaw[env_id] = torch.rand(1) * 2 * torch.pi - torch.pi

        return start_pos_map, goal_pos_map

    def sample_manual_task(self, env_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, "manual_tasks"):
            raise ValueError("No manual task file found for this scene!, create it with the task-annotator tool")
        pair_list = self.manual_tasks["pairs"]
        pair = random.choice(pair_list)

        start, goal = pair["start"], pair["goal"]

        self.start_yaw[env_id] = start["theta"]
        self.goal_yaw[env_id] = goal["theta"]

        start_pos_map = torch.tensor([start["y"], start["x"]], device=self.device)
        goal_pos_map = torch.tensor([goal["y"], goal["x"]], device=self.device)

        return start_pos_map, goal_pos_map

    def _write_path_buf(self, env_id: torch.Tensor | int, world_path: torch.Tensor) -> None:
        """Mirror ``world_path`` into the dense buffer, padding the tail with the goal position."""
        k = world_path.shape[0]
        if k > self.path_buf.shape[1]:
            self._grow_path_buf(k)
        self.path_buf[env_id, :k] = world_path
        self.path_buf[env_id, k:] = self.goal_pos_local[env_id]
        self.path_len[env_id] = k

    def _grow_path_buf(self, min_points: int) -> None:
        old_k = self.path_buf.shape[1]
        new_k = max(min_points, 2 * old_k)
        grown = torch.zeros((self.path_buf.shape[0], new_k, 2), dtype=torch.float32, device=self.device)
        grown[:, :old_k] = self.path_buf
        grown[:, old_k:] = self.goal_pos_local.unsqueeze(1)
        self.path_buf = grown
        self.max_path_points = new_k

    def compute_global_plan(self, env_ids: torch.Tensor, robot_pos_w: torch.Tensor, env_origins: torch.Tensor) -> None:
        assert (
            robot_pos_w.shape[0] == env_ids.shape[0] == env_origins.shape[0]
        ), "Batch size of robot positions, env origins, and env_ids must match"

        robot_pos_local = robot_pos_w - env_origins
        robot_map_coords = self.local_to_map_coords(robot_pos_local)


        obstacles_cpu = self.map_manager.obstacle_masks(env_ids)  # (num_envs, H, W)
        start_coords = robot_map_coords.int().cpu().numpy()  # (num_envs, 2)
        goal_coords = self.goal_pos_map[env_ids].int().cpu().numpy()  # (num_envs, 2)

        clearance_px = self.map_manager.clearance_px

        paths = []
        for idx in range(len(env_ids)):
            paths.append(
                self._executor.submit(
                    self._inflate_and_plan,
                    obstacles_cpu[idx],
                    start_coords[idx],
                    goal_coords[idx],
                    clearance_px,
                )
            )

        for idx, env_id in enumerate(env_ids):
            path, start_is_free = paths[idx].result()
            if path is None or len(path) == 0:
                # if I am in a valid cell, but no path is found, I just stand still
                if start_is_free:
                    # logger.info(f"No path found for robot {env_id}! (likely due to dynamic obstacle)")
                    self.path_tensors[env_id] = None
                # if previous path exists keep it (this will happen when robot pos is mapped onto an occupied cell)
                if self.path_tensors[env_id] is None:  # set to current location
                    robot_pos_local = self.map_to_local_coords(robot_map_coords[idx].unsqueeze(0))
                    self.path_tensors[env_id] = robot_pos_local
                    self._write_path_buf(env_id, robot_pos_local)
                    self.current_path_length[env_id] = 0.0  # reset
                continue

            path_subsampled = path[:: int(self.point_dist / self.map_manager.resolution)]

            path_tensor = torch.zeros((len(path_subsampled) + 1, 2), device=self.device)
            path_tensor[:-1] = torch.tensor(path_subsampled, device=self.device)

            path_tensor[-1] = self.goal_pos_map[env_id]  # ensure goal is included

            world_path = self.map_to_local_coords(path_tensor)  # (num_path_points, 2)
            self.path_tensors[env_id] = world_path
            self._write_path_buf(env_id, world_path)

            # update path length buffer for reward normalization
            path_len_m = world_path.size(0) * self.point_dist
            self.current_path_length[env_id] = path_len_m

            if path_len_m > self.initial_path_length[env_id]:
                self.initial_path_length[env_id] = path_len_m

    @staticmethod
    def _inflate_and_plan(obstacles: np.ndarray, start: np.ndarray, goal: np.ndarray, clearance_px: float) -> tuple:
        ''' Thread function that creates inflated global costmaps and computes A* paths. '''
        costmap = MapManager.create_inflated_costmap(obstacles, clearance_px, free_cells=(start, goal))
        path = pyastar2d.astar_path(costmap, tuple(start), tuple(goal), allow_diagonal=True)
        return path, bool(np.isfinite(costmap[start[0], start[1]]))

    def compute_robot_teleport_state(
        self,
        env_ids: torch.Tensor,
        env_origins: torch.Tensor,
        robot_z_pos: float = 0.0,
    ) -> torch.Tensor:
        map_coords = self.start_pos_map[env_ids]
        local_coords = self.map_to_local_coords(map_coords)

        num_robots = env_ids.shape[0]
        z_col = torch.zeros((num_robots, 1), device=self.device) + robot_z_pos  # set z to default
        pos_local = torch.cat([local_coords, z_col], dim=-1)

        pos_w = pos_local + env_origins[env_ids]
        yaw = self.start_yaw[env_ids]
        roll = torch.zeros_like(yaw)
        pitch = torch.zeros_like(yaw)
        quats = quat_from_euler_xyz(roll, pitch, yaw)

        zero_vel = torch.zeros((num_robots, 6), device=self.device)  # lin + angular vel

        state = torch.cat([pos_w, quats, zero_vel], dim=-1)  # (num_robots, 13)
        return state

    def sample_random_obstacle_on_path(self, env_ids: torch.Tensor) -> torch.Tensor:
        obstacle_pos = torch.zeros((len(env_ids), 2), device=self.device)  # in local coords
        for i, id in enumerate(env_ids):
            path = self.path_tensors[id].clone()
            if len(path) < 3:
                continue  # no path or path too short, skip

            obstacle_range = path[2:-2]  # avoid sampling obstacles too close to start and goal
            random_idx = random.randint(0, len(obstacle_range) - 1)
            # sample a point on the path and add gaussian noise to randomize position
            obstacle_pos[i] = obstacle_range[random_idx] + torch.randn(2, device=self.device) * 0.3

        return obstacle_pos

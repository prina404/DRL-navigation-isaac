from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.utils.math import quat_from_euler_xyz
from cfg.CFG import SCENE_USD_PATH
from isaacsim.core.utils.torch.maths import scale
import networkx as nx
import numpy as np
import torch
from go2.go2_articulation_cfg import UNITREE_GO2_CFG
import map_utils.voronoi as voronoi
import yaml
from loguru import logger
import pyastar2d
import cv2
import scipy.ndimage as sp

class MyEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        map_png = str(SCENE_USD_PATH.parent / SCENE_USD_PATH.stem) + "_map.png"
        map_yaml = str(SCENE_USD_PATH.parent / SCENE_USD_PATH.stem) + "_map.yaml"
        with open(map_yaml, 'r') as f:
            self.img_meta = yaml.safe_load(f)

        self.graph, coords = voronoi.compute_voronoi_graph(
            map_path=map_png,
            blur_radius=20,
            min_node_distance=20
        )

        self.map_img = cv2.imread(map_png, cv2.IMREAD_GRAYSCALE)
        self._costmap = self._create_inflated_costmap(inflation_scale = 3)
        # store the pixel coordinates of each node (indexed by [row][col])
        self.nodes = torch.Tensor([coords[idx] for idx in self.graph.nodes]).to(self.device)  

        # Tensors that map env_id -> start node_id and goal node_id
        self.start_pos_ids = torch.zeros((self.num_envs), dtype=torch.int64, device=self.device)
        self.goal_ids = torch.zeros((self.num_envs), dtype=torch.int64, device=self.device)  
        self.goal_pos = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)  
        self._path_tensors = [None] * self.num_envs  # list of (num_path_points, 2) tensors
        
        # Action buffer with the past 10 actions
        self._action_buffer = torch.zeros((self.num_envs, 10, 3), dtype=torch.float32, device=self.device)  
    
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        retVal = super().step(action)
        self._action_buffer = torch.roll(self._action_buffer, shifts=1, dims=1)
        self._action_buffer[:, 0, :] = action.to(self.device)
        return retVal

    

    def _create_inflated_costmap(self, inflation_scale: int) -> np.ndarray:
        binary_img = self.map_img == 0 # 1= occupied, 0=free/unknown
        dist_transform = sp.distance_transform_edt(~binary_img)
        costmap = np.clip(np.max(dist_transform) - (dist_transform * inflation_scale), 0, 255).astype(np.float32)
        costmap[costmap > 255] = float('inf')
        return costmap + 1.0 # min_cost = 1.0 for A* 

    
    def _map_to_world(self, node_coords: torch.Tensor) -> torch.Tensor:
        # node coords are tensor: [[row, col],
                                #  [row, col], ...]
        # in isaac frame, y directions are ok the same as in pixel space, but x are inverted
        H, W = self.map_img.shape
        node_coords[:, 1] = W - node_coords[:, 1]  # flip x axis
        node_coords = node_coords[:, [1,0]]  # swap to (x, y)
        
        n = node_coords.shape[0]
        x, y = self.img_meta['origin'][:2]

        origin = torch.zeros((n, 2), device=self.device)
        origin[:] = torch.tensor([x, y], device=self.device) # broadcast origin

        scaled = node_coords * self.img_meta['resolution']  # (N, 2)
        local_coords = scaled + origin  # (N, 2)
        return local_coords


    def sample_node_ids(self, num_samples: int) -> torch.Tensor:
        '''Returns (num_samples,) tensor of node_ids'''
        return torch.randint(0, self.nodes.shape[0], (num_samples,), device=self.device)



    def teleport_robots(self, env_ids: torch.Tensor, node_ids: torch.Tensor) -> None:
        ## set cartesian position, quaternion orientation in (w, x, y, z), and linear and angular velocity
        ## reset positions only for specified env indices

        robot = self.scene['unitree_go2']
        num_pos = node_ids.shape[0]

        positions = self.nodes[node_ids]    # map coordinates of node_ids
        positions = self._map_to_world(positions)  # (num_pos, 2)
        z_col = torch.zeros((num_pos, 1), device=self.device) + 0.01  # set z to 0.01
        pos_local = torch.cat([positions, z_col], dim=-1)  # (num_pos, 3)
        pos_local[:, -1] = 0.01  # set z to 0.01

        # swap x and y to match world frame

        # TODO: need to convert local coordinates to world coordinates based on env origin
        pos_w = pos_local + self.scene.env_origins[env_ids]  

        yaw = torch.rand((num_pos), device=self.device) * 2*torch.pi - torch.pi
        roll = torch.zeros_like(yaw)
        pitch = torch.zeros_like(yaw)
        quats = quat_from_euler_xyz(roll, pitch, yaw)  

        zero_vel = torch.zeros((num_pos, 6), device=self.device) # lin + angular vel

        state = torch.cat([pos_w, quats, zero_vel], dim=-1)  # (num_pos, 13)

        robot.write_root_state_to_sim(state, env_ids)
        print(pos_local, pos_w)
        self.start_pos_ids[env_ids] = node_ids


    def compute_goals_on_reset(self, env_ids: torch.Tensor) -> None:
        # TODO: add distance_based sampling. For the time being, 
        # just select a random neighboring node and navigate towards it
        for id in env_ids:
            current_node = self.start_pos_ids[id]
            nbr = next(self.graph.neighbors(current_node.item()))
            self.goal_ids[id] = nbr
            self.goal_pos[id] = self._map_to_world(self.nodes[nbr].unsqueeze(0))[0]

            r1, c1 = self.nodes[current_node].int().cpu().numpy()
            r2, c2 = self.nodes[nbr].int().cpu().numpy()
            path = pyastar2d.astar_path(self._costmap, (r1, c1), (r2, c2), allow_diagonal=True)
            if path is None or len(path) <= 1:
                logger.warning(f"No path found from {current_node} to {nbr}!")
                raise Exception # direct line

            point_dist = 0.3  # meters
            path_subsampled = path[::int(point_dist / self.img_meta['resolution'])]

            path_tensor = torch.zeros((len(path_subsampled)+1, 2), device=self.device)
            path_tensor[:-1] = torch.tensor(path_subsampled, device=self.device)
            path_tensor[-1] = torch.tensor([r2, c2], device=self.device)  # ensure goal is included

            world_path = self._map_to_world(path_tensor)  # (num_path_points, 2)
            self._path_tensors[id] = world_path  


    def _world_to_map(self, world_coords: torch.Tensor) -> torch.Tensor:
        # world coords are tensor: [[x, y],
                                #  [x, y], ...]
        logger.debug(f"world_coords: {world_coords}")
        n = len(world_coords)

        x, y = self.img_meta['origin'][:2]

        origin = torch.zeros((n, 2), device=self.device)
        origin[:] = torch.tensor([x, y], device=self.device) # broadcast origin
        print(origin)

        local_coords = world_coords - origin  # (N, 2)
        logger.debug(f"local_coords: {local_coords}")
        scaled = local_coords / self.img_meta['resolution']  # (N, 2)

        H, W = self.map_img.shape
        scaled[:, 0] = W - scaled[:, 0]  # flip x axis
        pixel_coords = scaled[:, [1,0]]  # swap to (row, col)
        logger.debug(f"pixel_coords: {pixel_coords}")
        return pixel_coords
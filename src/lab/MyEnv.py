from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils.math import quat_from_euler_xyz
from cfg.CFG import SCENE_USD_PATH
import networkx as nx
import torch
from go2.go2_articulation_cfg import UNITREE_GO2_CFG
import map_utils.voronoi as voronoi
import yaml
from loguru import logger
import pyastar2d
import cv2

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

        self._map_H, self._map_W = cv2.imread(map_png, cv2.IMREAD_GRAYSCALE).shape
        # store the pixel coordinates of each node (indexed by [row][col])
        self.nodes = torch.Tensor([coords[idx] for idx in self.graph.nodes]).to(self.device)  # (N, 2)

        # Tensors that map env_id -> start node_id and goal node_id
        self.start_pos = torch.zeros((self.num_envs), dtype=torch.int64, device=self.device)
        self.goals = torch.zeros((self.num_envs), dtype=torch.int64, device=self.device)  


    
    def _map_to_world(self, node_coords: torch.Tensor) -> torch.Tensor:
        # node coords are tensor: [[row, col],
                                #  [row, col], ...]
        # in isaac frame, y directions are ok the same as in pixel space, but x are inverted
        node_coords[:, 1] = self._map_W - node_coords[:, 1]  # flip x axis
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
        self.start_pos[env_ids] = node_ids


    def compute_goals_on_reset(self, env_ids: torch.Tensor) -> None:
        # TODO: add distance_based sampling. For the time being, 
        # just select a random neighboring node and navigate towards it
        for id in env_ids:
            current_node = self.start_pos[id]
            nbr = next(self.graph.neighbors(current_node.item()))
            self.goals[id] = nbr

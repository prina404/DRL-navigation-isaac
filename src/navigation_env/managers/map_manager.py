import time
from pathlib import Path

import cv2
import kornia.contrib as kornia
import numpy as np
import scipy.ndimage as sp
import torch
import yaml
from isaaclab.sensors import RayCasterData


class MapManager:
    def __init__(self, map_png_path: str | Path, map_yaml_path: str | Path, num_envs: int, device: torch.device):
        with open(map_yaml_path, "r") as f:
            self._img_meta = yaml.safe_load(f)

        self.device = device
        self.num_envs = num_envs

        self.map_img_path = str(map_png_path)
        self._map_img = cv2.imread(map_png_path, cv2.IMREAD_GRAYSCALE)
        if self._map_img is None:
            raise FileNotFoundError(f"Failed to read map image: {map_png_path}")

        self._map_tensor = (
            torch.from_numpy(self._map_img).unsqueeze(0).unsqueeze(0).repeat(num_envs, 1, 1, 1).to(device)
        )  # (num_envs, 1, H, W)
        self._costmap_backup = self._create_inflated_costmap_gpu(self._map_tensor, torch.arange(num_envs), inflation_scale=2)

        # Remember: in costmaps 0 = free space, inf = occupied
        self._global_costmap = self._costmap_backup.clone()  # (num_envs, 1, H, W)
        self.local_costmaps = torch.zeros_like(self._global_costmap)

        self._local_costmap_updated = False

    @property
    def resolution(self) -> float:
        return self._img_meta["resolution"]

    @property
    def origin(self) -> tuple[float, float]:
        return self._img_meta["origin"][:2]

    @property
    def shape(self) -> tuple[int, int]:
        return self._map_img.shape

    @property
    def map_img(self) -> np.ndarray:
        return self._map_img.copy()

    @property
    def global_costmap(self) -> torch.Tensor:
        """Returns a tensor of shape (num_envs, 1, H, W) with the updated global costmap"""
        if self._local_costmap_updated:
            ## Lazy update of global costmap because it is an expensive and low-frequency operation
            self._local_costmap_updated = False

            local_gridmap = (self.local_costmaps == 0.0).to(torch.int)  # 0 = occupied, 1 = free
            obstacle_map = torch.min(self._map_tensor, local_gridmap)  # keep all obstacles
            self._global_costmap = self._create_inflated_costmap_gpu(obstacle_map, torch.arange(self.num_envs), inflation_scale=2)

        return self._global_costmap

    @staticmethod
    def _create_inflated_costmap_cpu(map_img: np.ndarray, inflation_scale: int) -> torch.Tensor:
        binary_img = map_img == 0  # occupied=True
        dist_transform = sp.distance_transform_edt(~binary_img)
        costmap = np.clip(np.max(dist_transform) - (dist_transform * inflation_scale), 0, 255).astype(np.float32)
        max_value = costmap.max()
        costmap[costmap >= max_value] = float("inf")
        return costmap + 1.0  # min_cost = 1.0 for A*

    @staticmethod
    def _create_inflated_costmap_gpu(map_tensor: torch.Tensor, env_ids: torch.Tensor, inflation_scale: int) -> torch.Tensor:
        binary_img = map_tensor[env_ids] == 0
        dist_transform = kornia.distance_transform((binary_img).float(), h=0.5)
        costmap = torch.clamp(torch.max(dist_transform) - (dist_transform * inflation_scale), min=0, max=255)
        max_value = costmap.max()
        costmap[costmap >= max_value - 1] = float("inf")  # use max_value - n to introduce and solid "inflation" around obstacles
        return costmap + 1.0  # min_cost = 1.0 for A*

    def reset_costmaps(self, env_ids: torch.Tensor):
        # self.debug_vis()
        # self.debug_vis_global()
        self._global_costmap[env_ids] = self._costmap_backup[env_ids]
        self.local_costmaps[env_ids] = 0.0
        self._local_costmap_updated = True

    def update_local_costmap(self, lidar_scan: RayCasterData, env_origins: torch.Tensor):
        ## update the local costmap according to the full lidar scan
        self._local_costmap_updated = True

        # filter points further than thresh
        thresh = 3.0
        scan_dist = torch.norm(lidar_scan.ray_hits_w - lidar_scan.pos_w.unsqueeze(1), dim=-1)  # (B, N_rays)
        mask = scan_dist < thresh  # (B, N_rays)

        scan_local = lidar_scan.ray_hits_w - env_origins.unsqueeze(1)
        scan_map_coords = self.local_to_map_coords(scan_local)  # (B, N_rays, 2)

        row = scan_map_coords[..., 0]
        col = scan_map_coords[..., 1]

        B, N = mask.shape
        batches = torch.arange(self.num_envs, device=self.device)[:, None].expand(B, N)
        self.local_costmaps[batches[mask], 0, row[mask], col[mask]] = torch.inf


    def map_to_local_coords(self, map_coords: torch.Tensor) -> torch.Tensor:
        assert len(map_coords.shape) == 2 and map_coords.shape[1] == 2, "Expected map_coords to have shape (N, 2)"
        map_coords = map_coords.clone()
        _, W = self.shape
        map_coords[:, 1] = W - map_coords[:, 1]  # flip x axis
        map_xy = map_coords[:, [1, 0]]  # swap from (row,col) to (x, y)

        x, y = self.origin

        origin = torch.tensor([x, y], device=self.device)

        scaled = map_xy * self.resolution  # (N, 2)
        return scaled + origin  # (N, 2)

    def local_to_map_coords(self, local_coords: torch.Tensor) -> torch.Tensor:
        # data can be (B, N, 2|3) or (N, 2|3)
        if local_coords.size(-1) > 2:
            local_coords = local_coords[..., :2]  # only consider x, y

        local_xy = local_coords.clone()

        x, y = self.origin
        origin = torch.tensor([x, y], device=self.device)

        scaled = ((local_xy - origin) / self.resolution).to(torch.int)

        _, W = self.shape
        scaled[..., 0] = W - scaled[..., 0]  # flip x axis
        map_coords = scaled[..., [1, 0]]  # swap to (row, col)
        return map_coords
    

    # def debug_vis(self):
    #     import matplotlib.pyplot as plt

    #     # check costmaps from two separate batches
    #     local_costmap_1 = self.local_costmaps[0, 0].cpu().numpy()
    #     local_costmap_2 = self.local_costmaps[1, 0].cpu().numpy()
    #     plt.figure(figsize=(12, 6))
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(local_costmap_1, cmap="gray")
    #     plt.title("Local Costmap with Lidar Updates (Batch 0)")
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(local_costmap_2, cmap="gray")
    #     plt.title("Local Costmap with Lidar Updates (Batch 1)")
    #     plt.show()

#     def debug_vis_global(self):
#         import matplotlib.pyplot as plt

#         # check costmaps from two separate batches
#         global_costmap_1 = self._costmap_backup[0, 0].cpu().numpy()
#         global_costmap_2 = self._global_costmap[0, 0].cpu().numpy()
#         plt.figure(figsize=(12, 6))
#         plt.subplot(1, 2, 1)
#         plt.imshow(global_costmap_1, cmap="gray")
#         plt.title("Base Global Costmap (Batch 0)")
#         plt.subplot(1, 2, 2)
#         plt.imshow(global_costmap_2, cmap="gray")
#         plt.title("Updated Global Costmap (Batch 0)")
#         plt.show()

#     def print_debug(self):
#         t0 = time.time()
#         for _ in range(128):
#             cpu_impl = self._create_inflated_costmap_cpu(self.map_img, inflation_scale=2)
#         print(f"CPU costmap inflation took {time.time() - t0:.4f} seconds")
#         for _ in range(5):
#             gpu_impl = self._create_inflated_costmap_gpu(self._map_tensor, torch.arange(0, self.num_envs), inflation_scale=2)[
#                 0, 0
#             ].cpu()
#         print(f"GPU costmap inflation took {(time.time() - t0)/5:.4f} seconds")
#         import matplotlib.pyplot as plt

#         plt.figure(figsize=(12, 6))
#         plt.subplot(1, 2, 1)
#         plt.title("CPU Inflated Costmap")
#         plt.imshow(cpu_impl, cmap="gray")
#         plt.subplot(1, 2, 2)
#         plt.title("GPU Inflated Costmap")
#         plt.imshow(gpu_impl, cmap="gray")
#         plt.show()


# if __name__ == "__main__":
#     from cfg.CFG import DEVICE, SCENE_USD_PATH

#     map_png = str(SCENE_USD_PATH.parent / SCENE_USD_PATH.stem) + "_map.png"
#     map_yaml = str(SCENE_USD_PATH.parent / SCENE_USD_PATH.stem) + "_map.yaml"
#     costmap_manager = MapManager(map_png, map_yaml, num_envs=128, device=DEVICE)
#     costmap_manager.print_debug()

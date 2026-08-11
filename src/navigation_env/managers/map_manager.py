from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from isaaclab.sensors import RayCasterData

#: Narrowest gap (in meters) the planner is allowed to route through.
MIN_TRAVERSABLE_GAP = 0.10


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

        # Static occupancy of the scene, shared by every environment. The per-environment lidar
        # hits are merged onto it on demand, in `obstacle_masks`.
        self._occupied = self._map_img == 0  # (H, W), True where occupied

        # Remember: in costmaps 0 = free space, inf = occupied
        self.local_costmaps = torch.zeros((num_envs, 1, *self._map_img.shape), dtype=torch.float32, device=device)

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
    def occupancy_gridmap(self) -> np.ndarray:
        gray_pixels = (self._map_img > 1) & (self._map_img < 254)
        white_pixels = self._map_img >= 254
        black_pixels = self._map_img <= 1
        # use -1, [0...100] to represent respectively unknown, free, occupied.
        ogm = self._map_img.astype(np.int8, copy=True)
        ogm[gray_pixels] = -1
        ogm[white_pixels] = 0
        ogm[black_pixels] = 100
        return ogm

    @property
    def map_img(self) -> np.ndarray:
        return self._map_img.copy()

    def obstacle_masks(self, env_ids: torch.Tensor) -> np.ndarray:
        """Returns a (len(env_ids), H, W) bool array with the occupied cells seen by each environment.

        This is the only costmap data that has to leave the device: the inflation itself runs on the
        host, next to the A* that consumes it (see `PathManager._inflate_and_plan`).
        """
        lidar_hits = self.local_costmaps[env_ids, 0] != 0.0  # (len(env_ids), H, W)
        return self._occupied | lidar_hits.cpu().numpy()

    @property
    def clearance_px(self) -> float:
        """Distance the planner keeps from every obstacle, in cells."""
        return 0.5 * MIN_TRAVERSABLE_GAP / self.resolution

    @staticmethod
    def create_inflated_costmap(
        occupied: np.ndarray, clearance_px: float, free_cells: tuple = (), inflation_scale: int = 2
    ) -> np.ndarray:
        """Inflated A* costmap for a single environment.

        Cells closer than `clearance_px` to an obstacle are inflated

        The clearance is lifted in a `clearance_px` window around each of `free_cells` (the robot and
        its goal), so that an endpoint which ended up inside the band does not become unplannable.
        """
        dist_transform = cv2.distanceTransform((~occupied).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        costmap = np.clip(dist_transform.max() - (dist_transform * inflation_scale), 0, 255).astype(np.float32)

        blocked = dist_transform < clearance_px
        radius = int(np.ceil(clearance_px))
        for row, col in free_cells:
            if not blocked[row, col]:
                continue
            window = (slice(max(row - radius, 0), row + radius + 1), slice(max(col - radius, 0), col + radius + 1))
            blocked[window] = occupied[window]  # only the obstacles themselves are left in the way

        costmap[blocked] = np.inf
        return costmap + 1.0  # min_cost = 1.0 for A*

    def reset_costmaps(self, env_ids: torch.Tensor):
        # self.debug_vis()
        self.local_costmaps[env_ids] = 0.0

    def update_local_costmap(self, lidar_scan: RayCasterData, env_origins: torch.Tensor):
        ## update the local costmap according to the full lidar scan

        # filter points further than thresh
        thresh = 3.0
        ray_hits_w = lidar_scan.ray_hits_w.torch  # (B, N_rays, 3), warp-backed ProxyArray in IsaacLab 3.0
        scan_dist = torch.norm(ray_hits_w - lidar_scan.pos_w.torch.unsqueeze(1), dim=-1)  # (B, N_rays)
        mask = scan_dist < thresh  # (B, N_rays)

        scan_local = ray_hits_w - env_origins.unsqueeze(1)
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

        H, W = self.shape
        scaled[..., 0] = W - scaled[..., 0]  # flip x axis
        map_coords = scaled[..., [1, 0]]  # swap to (row, col)

        map_coords[..., 0] = torch.clamp(map_coords[..., 0], 0, H - 1)  # row
        map_coords[..., 1] = torch.clamp(map_coords[..., 1], 0, W - 1)  # col
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

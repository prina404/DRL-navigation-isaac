import cv2
import numpy as np
import torch
import yaml
from isaaclab.sensors import RayCasterData

from cfg.CFG import get_scene_usd_path

#: Narrowest gap (in meters) the planner is allowed to route through.
MIN_TRAVERSABLE_GAP = 0.20

#: Selects environments: a tensor of env ids, a single env id, or a slice (``slice(None)`` for all of them).
EnvIndex = torch.Tensor | int | slice


class MapManager:
    """Static occupancy grids of every loaded map, plus a lidar costmap per environment.

    The maps are stacked into a single (num_maps, H, W) grid, padded with occupied cells up to the size of the largest
    one, so that every per-env query stays batched. Each environment navigates the map `env_map_ids` assigns to it,
    so the coordinate transforms take the env ids the coordinates belong to.
    """

    def __init__(self, map_names: list[str], env_map_ids: torch.Tensor, device: torch.device):
        self.device = device
        self.map_names = map_names
        self.env_map_ids = env_map_ids.to(device)  # (num_envs,), index into map_names
        self.num_envs = len(env_map_ids)
        self._env_map_ids_cpu = env_map_ids.cpu().numpy()

        scene_usd_paths = [get_scene_usd_path(map_name) for map_name in map_names]
        self.map_img_paths = [str(path.with_name(path.stem + "_map.png")) for path in scene_usd_paths]

        map_imgs, map_metas = [], []
        for scene_usd_path, map_img_path in zip(scene_usd_paths, self.map_img_paths):
            map_img = cv2.imread(map_img_path, cv2.IMREAD_GRAYSCALE)
            if map_img is None:
                raise FileNotFoundError(f"Failed to read map image: {map_img_path}")
            map_imgs.append(map_img)

            with open(scene_usd_path.with_name(scene_usd_path.stem + "_map.yaml"), "r") as f:
                map_metas.append(yaml.safe_load(f))

        resolutions = {meta["resolution"] for meta in map_metas}
        if len(resolutions) != 1:
            raise ValueError(f"All maps must share the same resolution, got {resolutions}")
        self.resolution: float = resolutions.pop()

        # per-map frame, looked up for each coordinate in the transforms below
        self._origins = torch.tensor([meta["origin"][:2] for meta in map_metas], dtype=torch.float32, device=device)
        self._sizes = torch.tensor([img.shape for img in map_imgs], dtype=torch.int32, device=device)  # (H, W)

        # Static occupancy of each map, padded with occupied cells. The per-environment lidar hits are merged onto
        # it on demand, in `obstacle_masks`.
        height, width = self._sizes.max(dim=0).values.tolist()
        self._occupied = np.ones((len(map_names), height, width), dtype=bool)  # True where occupied
        for map_id, map_img in enumerate(map_imgs):
            self._occupied[map_id, : map_img.shape[0], : map_img.shape[1]] = map_img == 0

        # Remember: in costmaps 0 = free space, inf = occupied
        self.local_costmaps = torch.zeros((self.num_envs, 1, height, width), dtype=torch.float32, device=device)

    def obstacle_masks(self, env_ids: torch.Tensor) -> np.ndarray:
        """Returns a (len(env_ids), H, W) bool array with the occupied cells seen by each environment.

        This is the only costmap data that has to leave the device: the inflation itself runs on the
        host, next to the A* that consumes it (see `PathManager._inflate_and_plan`).
        """
        lidar_hits = self.local_costmaps[env_ids, 0] != 0.0  # (len(env_ids), H, W)
        static_occupancy = self._occupied[self._env_map_ids_cpu[env_ids.cpu().numpy()]]
        return static_occupancy | lidar_hits.cpu().numpy()

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
        self.local_costmaps[env_ids] = 0.0

    def update_local_costmap(self, lidar_scan: RayCasterData, map_origins: torch.Tensor):
        ## update the local costmap according to the full lidar scan

        # filter points further than thresh
        thresh = 3.0
        ray_hits_w = lidar_scan.ray_hits_w.torch  # (B, N_rays, 3), warp-backed ProxyArray in IsaacLab 3.0
        scan_dist = torch.norm(ray_hits_w - lidar_scan.pos_w.torch.unsqueeze(1), dim=-1)  # (B, N_rays)
        mask = scan_dist < thresh  # (B, N_rays)

        scan_local = ray_hits_w - map_origins.unsqueeze(1)
        scan_map_coords = self.local_to_map_coords(scan_local, slice(None))  # (B, N_rays, 2)

        row = scan_map_coords[..., 0]
        col = scan_map_coords[..., 1]

        B, N = mask.shape
        batches = torch.arange(self.num_envs, device=self.device)[:, None].expand(B, N)
        self.local_costmaps[batches[mask], 0, row[mask], col[mask]] = torch.inf

    def _map_values(self, values: torch.Tensor, env_ids: EnvIndex, coords: torch.Tensor) -> torch.Tensor:
        """Per-map `values` (num_maps, 2) of the envs in `env_ids`, broadcastable against `coords` (..., 2).

        `env_ids` either spans the leading dimensions of `coords`, or is a single env shared by all of them.
        """
        env_values = values[self.env_map_ids[env_ids]]  # (*env_ids.shape, 2)
        missing_dims = coords.dim() - env_values.dim()
        return env_values.view(*env_values.shape[:-1], *([1] * missing_dims), 2)

    def map_to_local_coords(self, map_coords: torch.Tensor, env_ids: EnvIndex) -> torch.Tensor:
        assert len(map_coords.shape) == 2 and map_coords.shape[1] == 2, "Expected map_coords to have shape (N, 2)"
        origin = self._map_values(self._origins, env_ids, map_coords)
        width = self._map_values(self._sizes, env_ids, map_coords)[..., 1]

        x = (width - map_coords[:, 1]) * self.resolution  # flip x axis, and swap from (row, col) to (x, y)
        y = map_coords[:, 0] * self.resolution
        return torch.stack([x, y], dim=-1) + origin  # (N, 2)

    def local_to_map_coords(self, local_coords: torch.Tensor, env_ids: EnvIndex) -> torch.Tensor:
        # data can be (B, N, 2|3) or (N, 2|3)
        local_xy = local_coords[..., :2]  # only consider x, y
        origin = self._map_values(self._origins, env_ids, local_xy)
        size = self._map_values(self._sizes, env_ids, local_xy)

        scaled = ((local_xy - origin) / self.resolution).to(torch.int)
        row = scaled[..., 1]
        col = size[..., 1] - scaled[..., 0]  # flip x axis
        map_coords = torch.stack([row, col], dim=-1)  # swap to (row, col)

        return torch.minimum(torch.clamp(map_coords, min=0), size - 1)  # stay inside the env's own map

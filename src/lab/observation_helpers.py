import math

from loguru import logger
import torch
import torch.nn as nn
from lab.MyEnv import MyEnv
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
from isaaclab.sensors import MultiMeshRayCaster
from cfg.CFG import DEVICE
from lab.vision_encoder import ViTEncoder


class VisionEncoder:
    def __init__(self, encoder: nn.Module = ViTEncoder(), device=DEVICE, normalize: bool = True):
        self.encoder = encoder.to(device)
        self.normalize = normalize

    @torch.no_grad()
    def __call__(self, env: RslRlVecEnvWrapper) -> torch.Tensor:
        # rgb = torch.zeros((env.num_envs, 3, 320, 240), device=env.device)

        cam = env.scene["camera"]
        rgb = cam.data.output["rgb"]  # (N, H, W, C)
        rgb = rgb.permute(0, 3, 1, 2)  # Convert to N, C, H, W for ViT
        rgb = rgb.to(dtype=torch.float32) / 255.0
        embedding = self.encoder(rgb)
        if self.normalize:
            embedding = torch.sigmoid(embedding)
        return embedding


def get_dummy_embedding(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros((env.num_envs, 768), device=env.device, dtype=torch.float32)


def get_dummy_lidar(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros((env.num_envs, 50), device=env.device, dtype=torch.float32)


def get_lidar(env: RslRlVecEnvWrapper, num_obstacles: int, normalize=True) -> torch.Tensor:
    """
    Returns (B, 2 * num_obstacles) where for each batch we concatenate the distance and yaw of the closest num_obstacles points:
        result[B, :num_obstacles] = distance
        result[B, num_obstacles:] = yaw (radians)
    """

    lidar: MultiMeshRayCaster = env.scene["lidar"]

    robot_rot = lidar.data.quat_w  # (B, 4)
    robot_pos = lidar.data.pos_w  # (B, 3)

    scan_w: torch.Tensor = lidar.data.ray_hits_w  # (B, N_rays, 3)
    scan = scan_w - robot_pos.unsqueeze(1)  # (B, N_rays, 3) in robot frame

    B, N_rays, _ = scan.shape

    # Distance per ray
    point_dist = torch.norm(scan, dim=-1)  # (B, N_rays)

    # Yaw per ray
    yaw_w = torch.atan2(scan[..., 1], scan[..., 0])  # (B, N_rays) yaw angle of each ray (world frame)
    _, _, robot_yaw = euler_xyz_from_quat(robot_rot)  # (B, 1) robot yaw angles
    robot_yaw += math.pi  # invert lidar direction
    yaw = -wrap_to_pi(yaw_w - robot_yaw.unsqueeze(1))  # (B, N_rays) yaw angle in robot frame

    # Voxelize pointcloud

    VOXEL_SIZE = 0.25
    VOLUME_DIMS = torch.tensor((5, 5, 1), device=env.device, dtype=torch.float32)
    MAX_POINT_DIST = torch.norm(VOLUME_DIMS)

    origin = torch.tensor(
        [
            VOLUME_DIMS[0] / 2,
            VOLUME_DIMS[1] / 2,
            0.5,
        ],
        device=env.device,
    )

    voxels = torch.floor((scan + origin) / VOXEL_SIZE).to(torch.int32)
    Nx, Ny, Nz = (VOLUME_DIMS / VOXEL_SIZE).to(torch.int32)
    V = Nx * Ny * Nz

    mask = (
          (voxels[..., 0] >= 0) & (voxels[..., 0] < Nx)
        & (voxels[..., 1] >= 0) & (voxels[..., 1] < Ny)
        & (voxels[..., 2] >= 0) & (voxels[..., 2] < Nz)
    )  # binary mask of shape (B, N_rays) which stores whether each ray is inside the volume

    voxel_hash = (
        voxels[..., 0] + voxels[..., 1] * Nx + voxels[..., 2] * Nx * Ny
    ).to(torch.int64)  # (B, N_rays, 3) -> (B, N_rays) hashed voxel coordinates

    # Invalidate out-of-bounds
    voxel_hash.masked_fill_(~mask, 0)
    point_dist.masked_fill_(~mask, float("inf"))

    # Allocate per-batch voxel distance buffer
    min_dist = torch.full(size=(B, V), fill_value=float("inf"), device=env.device)

    # scatter_reduce: if multiple rays have the same voxel_hash, set them to the min distance
    min_dist.scatter_reduce_(dim=1, index=voxel_hash, src=point_dist, reduce="amin", include_self=True)  # (B, V)

    # Expand voxel mins back to ray positions
    gathered_min = torch.gather(min_dist, 1, voxel_hash)  # (B, N_rays)

    # A ray is the voxel minimum if:
    is_min = point_dist == gathered_min
    # Remove invalid rays
    is_min = is_min & mask

    # Keep only one ray per voxel: pick the first ray (smallest ray index) among rays
    # whose distance equals the voxel minimum. This avoids later scatter overwrites.
    ray_idx = torch.arange(N_rays, device=env.device, dtype=torch.int64).unsqueeze(0).expand(B, N_rays)  # (B, N_rays)
    sentinel = N_rays  # out-of-range sentinel

    # ray index only for rays that are voxel-min; otherwise sentinel
    ray_idx_masked = torch.where(is_min, ray_idx, torch.full_like(ray_idx, sentinel))

    # per-voxel minimum ray index (amin across ray indices, so first occurrence wins)
    min_ray_idx = torch.full((B, V), sentinel, device=env.device, dtype=torch.int64)
    min_ray_idx.scatter_reduce_(1, voxel_hash, ray_idx_masked, reduce="amin", include_self=True)  # (B, V)

    # prepare gather indices (replace sentinel with 0 to avoid OOB; will mask later)
    min_ray_idx_gather = torch.where(min_ray_idx == sentinel, torch.zeros_like(min_ray_idx), min_ray_idx)

    # gather yaw for each voxel by the selected ray index
    min_yaw = torch.gather(yaw, 1, min_ray_idx_gather)  # (B, V)
    # mark voxels that had no valid ray
    no_ray_voxel = min_ray_idx == sentinel
    min_yaw.masked_fill_(no_ray_voxel, 0.0)

    # Get top-k closest voxels
    topk_dist, topk_idx = torch.topk(min_dist, k=num_obstacles, dim=1, largest=False)
    topk_yaw = torch.gather(min_yaw, 1, topk_idx)

    # Replace inf padding
    topk_dist = torch.where(torch.isinf(topk_dist), torch.full_like(topk_dist, MAX_POINT_DIST), topk_dist)

    # Normalize distances
    if normalize:
        topk_dist = torch.clamp(topk_dist / MAX_POINT_DIST, 0.0, 1.0)
        topk_yaw = (topk_yaw + torch.pi) / (2 * torch.pi)  # normalize yaw to [0, 1]

    return torch.cat([topk_dist, topk_yaw], dim=1)  # (B, 2*num_obstacles)


def _get_path_coords(env: RslRlVecEnvWrapper, num_points_forward: int) -> list[torch.Tensor]:
    # for each robot, I compute the distance to the goal, then filter all the path points that are
    # within that distance. return a tensor of fixed size with the path points.
    # If there are too few points I pad with goal pos
    env: MyEnv = env.unwrapped
    local_robot_coords = env.scene["unitree_go2"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]  # (B, 2)
    dist_to_goals = torch.norm(local_robot_coords - env.goal_pos, dim=-1)  # (B,) distance from robot to goal

    res = torch.zeros((env.num_envs, num_points_forward + 1, 2), device=env.device)  # (B, num_points_forward, 2)
    res += env.goal_pos.unsqueeze(1)  # default to goal pos

    for id in range(env.num_envs):
        path_points = env._path_tensors[id]  # variable lenth tensor of (num_path_points, 2)
        goal_pos = env.goal_pos[id]  # (2,)
        # keep only points on the path that are closer than the robot to the goal
        forward_points = path_points[torch.norm(path_points - goal_pos, dim=-1) <= dist_to_goals[id]]
        n_forward = min(forward_points.shape[0], num_points_forward + 1)
        # logger.debug(f"{forward_points.shape=}, {res.shape=}")
        res[id, :n_forward] = forward_points[:n_forward]

    return res


def get_path_obs(
    env: RslRlVecEnvWrapper, 
    num_points_forward: int, 
    normalize: bool = True, 
    debug_vis: bool = False
) -> torch.Tensor:
    env: MyEnv = env.unwrapped
    if getattr(env, "goal_pos", None) is None:  # this is needed because this is called before env is fully initialized
        return torch.zeros((env.num_envs, num_points_forward * 2), device=env.device)

    # returns a 1D tensor of size (B*num_points_forward*2) with distance and heading to each path point
    local_robot_coords = env.scene["unitree_go2"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]  # (B, 2)

    path_coords = _get_path_coords(env, num_points_forward)  # (B, num_points_forward, 2)

    robot_coords_expanded = torch.zeros_like(path_coords) + local_robot_coords.unsqueeze(
        1
    )  # (B, num_points_forward, 2)

    path_distances = torch.norm(path_coords - robot_coords_expanded, dim=-1)  # (B, num_points_forward)

    deltas = path_coords - robot_coords_expanded  # (B, num_points_forward, 2)
    path_angles = torch.atan2(deltas[..., 1], deltas[..., 0])  # angle from robot position to each path point

    robot_rot = env.scene["unitree_go2"].data.root_link_quat_w  # (B, 4)
    _, _, robot_yaw = euler_xyz_from_quat(robot_rot)  # (B, 1)

    headings = -wrap_to_pi(path_angles - robot_yaw.unsqueeze(1))  # (B, num_points_forward)

    if debug_vis:
        debug_visualize(env, num_points_forward)
        logger.info(f"{headings=}")

    if normalize:
        max_horizon_dist = 2  # horizon distance for normalization (in m)
        path_distances = torch.clamp(path_distances / max_horizon_dist, 0.0, 1.0)
        headings = (headings + math.pi) / (2 * math.pi)  # normalize yaw to [0, 1]

    return torch.cat([path_distances, headings], dim=1).flatten().unsqueeze(0)  # (B * num_points_forward * 2)




def debug_visualize(env: MyEnv, num_points_forward: int) -> None:
    from isaacsim.util.debug_draw import _debug_draw
    draw = _debug_draw.acquire_debug_draw_interface()
    z_coord = torch.zeros((env.num_envs, num_points_forward + 1, 1), device=env.device) + 0.4
    path_coords = _get_path_coords(env, num_points_forward)  # (B, num_points_forward, 2)
    path_coords_3d = torch.cat([path_coords, z_coord], dim=-1)  # (B, num_points_forward, 3)
    draw.draw_points(
        path_coords_3d.cpu().numpy().reshape(-1, 3).tolist(),
        [(0, 1, 0, 1)],
        [10.0] * (num_points_forward * env.num_envs),
    )

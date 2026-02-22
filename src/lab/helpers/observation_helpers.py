import math
from typing import Literal

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import MultiMeshRayCaster
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    subtract_frame_transforms,
    wrap_to_pi,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from cfg.CFG import DEVICE
from lab.MyEnv import MyEnv
from models.vision_encoder import ViNTVisionEncoder, ViTEncoder


class VisionEncoder:
    def __init__(self, encoder: Literal["vit", "vint"], device=DEVICE, normalize: bool = True):
        if encoder == "vit":
            self.encoder = ViTEncoder().to(device)
        elif encoder == "vint":
            self.encoder = ViNTVisionEncoder().to(device)
        else:
            raise ValueError(f"Unknown encoder type: {encoder}")
        self.normalize = normalize

    @torch.no_grad()
    def __call__(self, env: RslRlVecEnvWrapper) -> torch.Tensor:
        cam = env.scene["camera"]
        rgb = cam.data.output["rgb"]  # (N, H, W, C)
        rgb = rgb.permute(0, 3, 1, 2)  # Convert to N, C, H, W for ViT/ViNT
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

    # binary mask of shape (B, N_rays) which stores whether each ray is inside the volume
    mask = (
        (voxels[..., 0] >= 0)
        & (voxels[..., 0] < Nx)
        & (voxels[..., 1] >= 0)
        & (voxels[..., 1] < Ny)
        & (voxels[..., 2] >= 0)
        & (voxels[..., 2] < Nz)
    )

    # (B, N_rays, 3) -> (B, N_rays) hashed voxel coordinates
    voxel_hash = (voxels[..., 0] + voxels[..., 1] * Nx + voxels[..., 2] * Nx * Ny).to(torch.int64)

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


def _get_path_coords(env: RslRlVecEnvWrapper, num_points_forward: int) -> torch.Tensor:
    # I find which point is closest to the robot,
    # then use the fact that the path points are ordered and take the next num_points_forward points.
    # If the closest point is the last point, I pad with the goal pos.
    env: MyEnv = env.unwrapped
    local_robot_coords = env.scene["robot"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]  # (B, 2)
    res = torch.zeros((env.num_envs, num_points_forward + 1, 2), device=env.device)  # (B, num_points_forward, 2)
    res += env.goal_pos.unsqueeze(1)  # default to goal pos
    for id in range(env.num_envs):
        path_points = env._path_tensors[id]
        closest_idx = torch.argmin(torch.norm(path_points - local_robot_coords[id], dim=-1))
        n_forward = min(path_points.shape[0] - closest_idx, num_points_forward + 1)
        res[id, :n_forward] = path_points[closest_idx : closest_idx + n_forward]

    return res


def get_path_obs(
    env: RslRlVecEnvWrapper,
    num_points_forward: int,
    normalize: bool = True,
    debug_vis: bool = False,
) -> torch.Tensor:
    env: MyEnv = env.unwrapped
    if getattr(env, "goal_pos", None) is None:  # this is needed because this is called before env is fully initialized
        return torch.zeros((env.num_envs, num_points_forward * 2), device=env.device)

    # returns a 1D tensor of size (B*num_points_forward*2) with distance and heading to each path point
    local_robot_coords = env.scene["robot"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]  # (B, 2)

    path_coords = _get_path_coords(env, num_points_forward)  # (B, num_points_forward, 2)

    robot_coords_expanded = torch.zeros_like(path_coords) + local_robot_coords.unsqueeze(1)  # (B, num_points_forward, 2)

    path_distances = torch.norm(path_coords - robot_coords_expanded, dim=-1)  # (B, num_points_forward)

    deltas = path_coords - robot_coords_expanded  # (B, num_points_forward, 2)
    path_angles = torch.atan2(deltas[..., 1], deltas[..., 0])  # angle from robot position to each path point

    robot_rot = env.scene["robot"].data.root_link_quat_w  # (B, 4)
    _, _, robot_yaw = euler_xyz_from_quat(robot_rot)  # (B, 1)

    headings = -wrap_to_pi(path_angles - robot_yaw.unsqueeze(1))  # (B, num_points_forward)

    if normalize:
        max_horizon_dist = 2  # horizon distance for normalization (in m)
        path_distances = torch.clamp(path_distances / max_horizon_dist, 0.0, 1.0)
        headings = (headings + math.pi) / (2 * math.pi)  # normalize yaw to [0, 1]

    return torch.cat([path_distances, headings], dim=1)  # (B, num_points_forward * 2)


def get_previous_actions(env: RslRlVecEnvWrapper) -> torch.Tensor:
    env: MyEnv = env.unwrapped
    if not hasattr(env, "_action_buffer"):
        return torch.zeros((env.num_envs, 5 * 3), device=env.device)
    return env._action_buffer.reshape(env.num_envs, -1)


def get_previous_velocities(env: RslRlVecEnvWrapper) -> torch.Tensor:
    env: MyEnv = env.unwrapped
    if not hasattr(env, "_velocity_buffer"):
        return torch.zeros((env.num_envs, 10 * 3), device=env.device)
    return env._velocity_buffer.reshape(env.num_envs, -1)


def get_goal_relative_position(env: RslRlVecEnvWrapper | MyEnv) -> torch.Tensor:
    # same logic as in the nominal action computation
    if isinstance(env, RslRlVecEnvWrapper):
        env: MyEnv = env.unwrapped

    if not hasattr(env, "_path_tensors"):
        return torch.zeros((env.num_envs, 3), device=env.device)

    robot = env.scene["robot"]
    robot_pos_local = robot.data.root_com_pos_w[:, :2] - env.scene.env_origins[:, :2]  # (B, 2)
    goal_pos_local = torch.zeros((env.num_envs, 2), device=env.device)

    for i, path in enumerate(env._path_tensors):
        if path is None:
            goal_pos_local[i] = torch.zeros(2, device=env.device)
            continue
        closest = torch.argmin(torch.norm(path - robot_pos_local[i], dim=-1))
        forward_point = path[closest : closest + 4][-1]  # last point in my horizon
        goal_pos_local[i] = forward_point

    goal_pos_w = goal_pos_local + env.scene.env_origins[:, :2]  # (B, 2)
    padded_goal = torch.cat([goal_pos_w, torch.zeros((env.num_envs, 1), device=env.device)], dim=-1)  # (B, 3)

    # Now goal positions are in robot frame
    goal_pos_r, _ = subtract_frame_transforms(
        robot.data.root_com_pos_w,
        robot.data.root_link_quat_w,
        padded_goal,
    )

    goal_rot = torch.atan2(goal_pos_r[:, 1], goal_pos_r[:, 0])  # angle to goal in robot frame
    goal_displacement_vector = torch.cat([goal_pos_r[:, :2], goal_rot.unsqueeze(-1)], dim=-1)  # (B, 3)

    return torch.clamp(goal_displacement_vector, -1.0, 1.0)

from loguru import logger
import torch
import torch.nn as nn
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.sensors import MultiMeshRayCaster
from cfg.CFG import DEVICE
from lab.vision_encoder import ViTEncoder
from torch_scatter import scatter
import warnings


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
        if self.normalize:
            rgb = rgb.to(dtype=torch.float32) / 255.0
        embedding = self.encoder(rgb)
        return embedding


def get_dummy_embedding(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros((env.num_envs, 768), device=env.device, dtype=torch.float32)


def get_dummy_lidar(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros((env.num_envs, 50), device=env.device, dtype=torch.float32)


def get_lidar(env: RslRlVecEnvWrapper, num_obstacles: int, normalize = True) -> torch.Tensor:
    ''' Returns a (B, num_obstacles) tensor of the closest obstacle distances
        NOTE: This fuction does not filter lidar data.
        If there are floor/body collisions right under the sensor, they will be included in the output.
    '''


    lidar: MultiMeshRayCaster = env.scene["lidar"]
    robot_poses = lidar.data.pos_w # (B, 3)
    scan: torch.Tensor = lidar.data.ray_hits_w # (B, N_rays, 3)

    scan = scan - robot_poses.unsqueeze(1)  # (B, N_rays, 3) in robot frame


    point_dist = torch.norm(scan, dim=-1) # (B, N_rays) lengths of each ray

    ## I voxelize the data, in a volume 5x5x1, centered on the robot
    VOXEL_SIZE = 0.25
    VOLUME_DIMS = torch.tensor((10, 10, 1), device=env.device, dtype=torch.float32)  # x, y, z
    MAX_POINT_DIST = torch.norm(VOLUME_DIMS)

    origin = torch.tensor([
        VOLUME_DIMS[0] / 2,
        VOLUME_DIMS[1] / 2,
        0.5, # slightly above robot base 
    ], device=env.device)  

    voxels = torch.floor((scan + origin) / VOXEL_SIZE).to(torch.int32)
    Nx, Ny, Nz = (VOLUME_DIMS / VOXEL_SIZE).to(torch.int32) # number of voxels per dimension

    mask = (    
        (voxels[..., 0] >= 0) & (voxels[..., 0] < Nx) &
        (voxels[..., 1] >= 0) & (voxels[..., 1] < Ny) &
        (voxels[..., 2] >= 0) & (voxels[..., 2] < Nz)
    ) # binary mask of shape (B, N_rays) with voxels inside the volume

    voxel_hash = (
        voxels[:, :, 0]
        + voxels[:, :, 1] * Nx
        + voxels[:, :, 2] * Nx * Ny 
    ) # (B, N_rays, 3) -> (B, N_rays) hashed voxel coordinates

    voxel_hash.masked_fill_(~mask, 0)   # set out-of-bounds to zero

    # TODO: torch.unique sorts each tensor, monitor speed for scaling
    unique, inverse = torch.unique(voxel_hash, return_inverse=True)

    ## scatter: for each voxel, keep only the point with the minimum distance
    ## This reduces the number of points to process, to O(N_voxels)
    ## ouptut shape: (B, len(unique))
    out = scatter(point_dist, inverse, dim=1 , reduce='min')
    out[out == 0.0] = float('inf') # set padded zeros to inf distance

    # NOTE: if in a given batch there are less than num_obstacles points, the 
    # returned distances will contain inf values. so normalize before passing to NN.
    if out.size(1) < 1:
        warnings.warn(f"Lidar voxelization returned less than one point. Check lidar and voxelization parameters to ensure everything is working properly.")
        warnings.warn(f"Returning vector of zeros")
        return torch.zeros((env.num_envs, num_obstacles), device=env.device, dtype=torch.float32)
    elif out.size(1) < num_obstacles:
        top_k = torch.topk(out, k=out.size(1), dim=1, largest=False).values  
        padding = torch.full((env.num_envs, num_obstacles - out.size(1)), MAX_POINT_DIST, device=env.device)
        res = torch.cat([top_k, padding], dim=1) # (B, k)
    else:
        res = torch.topk(out, k=num_obstacles, dim=1, largest=False).values  # (B, k)

    if normalize:
        res = torch.clamp(res / MAX_POINT_DIST, 0.0, 1.0)
    logger.info(f"Lidar out: {res}")
    return res


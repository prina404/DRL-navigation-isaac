import torch
import torch.nn as nn
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

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
        if self.normalize:
            rgb = rgb.to(dtype=torch.float32) / 255.0
        embedding = self.encoder(rgb)
        return embedding


def get_dummy_embedding(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros((env.num_envs, 768), device=env.device, dtype=torch.float32)


def get_dummy_lidar(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros((env.num_envs, 50), device=env.device, dtype=torch.float32)


def get_lidar(env: RslRlVecEnvWrapper) -> torch.Tensor:
    pass

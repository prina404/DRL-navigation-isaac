import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

class NavPolicyAC(ActorCritic):
    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims,
        critic_hidden_dims,
        activation,
        init_noise_std,
        encoder: nn.Module,
        encoder_out_dim: int,
        freeze_encoder: bool = True,
    ):
        # Build actor/critic for ViT embedding size
        super().__init__(
            encoder_out_dim,
            encoder_out_dim,
            num_actions,
            actor_hidden_dims,
            critic_hidden_dims,
            activation,
            init_noise_std,
        )
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def _encode(self, obs: TensorDict) -> torch.Tensor:
        # Extract RGB tensor from TensorDict  
        rgb_obs = obs["rgb"]  # Shape: [N, H, W, 3]  
        
        # Your encoder expects [N, C, H, W], so transpose if needed  
        if rgb_obs.shape[-1] == 3:  # H, W, C format  
            rgb_obs = rgb_obs.permute(0, 3, 1, 2)  # Convert to N, C, H, W  
        
        emb = self.encoder(rgb_obs)  
        if emb.ndim > 2:  
            emb = emb.flatten(1)  
        return emb

    def act(self, obs, **kwargs):
        return super().act(self._encode(obs), **kwargs)

    def evaluate(self, obs, **kwargs):
        return super().evaluate(self._encode(obs), **kwargs)

    def act_inference(self, obs):
        return super().act_inference(self._encode(obs))

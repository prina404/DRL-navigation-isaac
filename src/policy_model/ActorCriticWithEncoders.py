from typing import Any
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from loguru import logger

from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.modules.actor_critic import ActorCritic
from typing import Literal


class ActorCriticWithEncoders(ActorCritic):
    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        encoders_hidden_dims: dict[str, tuple[int] | list[int]] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticWithEncoders.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )

        nn.Module.__init__(self)  # skip ActorCritic constructor
        self.obs_groups = obs_groups

        # Encoders
        self.actor_encoders, actor_input_dim = self._init_encoder_modules(obs, "policy", activation, encoders_hidden_dims)
        self.critic_encoders, critic_input_dim = self._init_encoder_modules(obs, "critic", activation, encoders_hidden_dims)

        # Actor
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.actor = MLP(actor_input_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)
        
        # Critic
        self.critic = MLP(critic_input_dim, 1, critic_hidden_dims, activation)

        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.actor_obs_normalizer = EmpiricalNormalization(actor_input_dim) if actor_obs_normalization else nn.Identity()
        self.critic_obs_normalizer = EmpiricalNormalization(critic_input_dim) if critic_obs_normalization else nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
            
        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def _init_encoder_modules(
        self, obs: TensorDict, group_name: str, activation: str, encoder_cfg: dict | None
    ) -> tuple[nn.ModuleDict, int]:
        encoders = nn.ModuleDict()
        input_dim = 0
        for obs_name in self.obs_groups[group_name]:
            obs_tensor = obs[obs_name]
            obs_size = obs_tensor.shape[-1]
            if encoder_cfg is not None and obs_name in encoder_cfg:  # initialize an encoder for this observation term
                hidden_dims = encoder_cfg[obs_name]
                encoders[obs_name] = MLP(obs_size, hidden_dims[-1], hidden_dims[:-1], activation, last_activation='sigmoid')
                input_dim += hidden_dims[-1]
            else:
                encoders[obs_name] = nn.Identity()
                input_dim += obs_size
        return encoders, input_dim
    
    def _get_encoded_obs(self, obs: TensorDict, group_name: Literal["policy", "critic"]) -> torch.Tensor:
        encoder_outputs = []
        encoders = self.actor_encoders if group_name == "policy" else self.critic_encoders
        for obs_name in self.obs_groups[group_name]:
            encoder = encoders[obs_name]
            encoder_outputs.append(encoder(obs[obs_name]))
        return torch.cat(encoder_outputs, dim=-1)
    
    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._get_encoded_obs(obs, "policy")
    
    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._get_encoded_obs(obs, "critic")
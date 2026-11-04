from __future__ import annotations

import copy
import math
from typing import Any

from policy.NavPolicyv2 import MLPModelWithEncoders, resolve_legacy_obs_groups

ENCODERS_HIDDEN_DIMS: dict[str, list[int]] = {
    "velocity_buffer": [128, 128],
    "action_buffer": [128, 128],
    "global_plan": [128, 128],
}

go2_policy_cfg: dict[str, Any] = {
    # Training loop parameters
    "num_steps_per_env": math.ceil((15 / (16 * 0.005))),  # 15 seconds per episode, converted to steps (with decimation)
    "save_interval": 50,  # Save a checkpoint every 50 iterations
    "max_iterations": 1000,
    "device": "cuda:0",
    # Observation routing
    "obs_groups": None,  # must be assigned depending on task configuration
    # Actor architecture
    "actor": {
        "class_name": MLPModelWithEncoders,
        "hidden_dims": [1024, 512, 256, 128],
        "activation": "elu",
        "obs_normalization": False,
        "last_activation": None,  # linear: the action term squashes the sampled action into the command ellipse
        "encoders_hidden_dims": ENCODERS_HIDDEN_DIMS,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",  # "HeteroscedasticGaussianDistribution" for a state-dependent std
            "init_std": 1.0,
            "std_type": "log",
        },
    },
    # Critic architecture
    "critic": {
        "class_name": MLPModelWithEncoders,
        "hidden_dims": [1024, 512, 256, 128],
        "activation": "elu",
        "obs_normalization": False,
        "last_activation": None,  # the legacy critic head was linear
        "encoders_hidden_dims": ENCODERS_HIDDEN_DIMS,
        "distribution_cfg": None,  # deterministic value output
    },
    # PPO algorithm parameters
    "algorithm": {
        "class_name": "PPO",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.01,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
    },
    "load_run": "unitree_go2_nav$",  # a regex: unanchored, it also matches ckpts/unitree_go2_nav_jit
    "load_checkpoint": "last_ckpt_vis.pt",
}


def make_go2_policy_cfg(obs_groups: dict[str, list[str]], **overrides: Any) -> dict[str, Any]:
    cfg = copy.deepcopy(go2_policy_cfg)
    cfg["obs_groups"] = resolve_legacy_obs_groups(obs_groups)
    cfg.update(overrides)
    return cfg

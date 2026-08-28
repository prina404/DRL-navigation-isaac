from __future__ import annotations

import copy
import math
from typing import Any

from policy.NavPolicyv2 import MLPModelWithEncoders, resolve_legacy_obs_groups

STUDENT_ENCODERS_HIDDEN_DIMS: dict[str, list[int]] = {
    "velocity_buffer": [128, 128],
    "action_buffer": [128, 128],
    "global_plan": [128, 128],
    "depth": [1024, 1024, 512],
    "vision": [1024, 1024, 512],
}

student_policy_cfg: dict[str, Any] = {
    "num_steps_per_env": math.ceil((15 / (16 * 0.005))),
    "save_interval": 50,
    "max_iterations": 1000,
    "device": "cuda:0",
    "obs_groups": None,
    "actor": {
        "class_name": MLPModelWithEncoders,
        # "hidden_dims": [256, 128, 128],
        "hidden_dims": [1024, 512, 512, 256, 128],
        "activation": "elu",
        "obs_normalization": False,
        "last_activation": None,
        "encoders_hidden_dims": STUDENT_ENCODERS_HIDDEN_DIMS,
        "distribution_cfg": {
            "class_name": "HeteroscedasticGaussianDistribution",
            "init_std": 1.0,
            "std_type": "log",
        },
    },
    "critic": {
        "class_name": MLPModelWithEncoders,
        "hidden_dims": [256, 256, 128],
        "activation": "elu",
        "obs_normalization": False,
        "last_activation": None,
        "encoders_hidden_dims": STUDENT_ENCODERS_HIDDEN_DIMS,
        "distribution_cfg": None,
    },
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
    "load_run": "distillation",
    "load_checkpoint": "student_actor.pt",
}


def make_student_policy_cfg(obs_groups: dict[str, list[str]], **overrides: Any) -> dict[str, Any]:
    cfg = copy.deepcopy(student_policy_cfg)
    cfg["obs_groups"] = resolve_legacy_obs_groups(obs_groups)
    cfg.update(overrides)
    return cfg


def make_student_actor_cfg(init_std: float | None = None) -> dict[str, Any]:
    actor_cfg = copy.deepcopy(student_policy_cfg["actor"])
    actor_cfg.pop("class_name", None)
    if init_std is not None:
        actor_cfg["distribution_cfg"]["init_std"] = init_std
    return actor_cfg

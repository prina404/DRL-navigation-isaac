import torch
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.modules import ActorCritic

from cfg.CFG import CHECKPOINT_DIR


def get_locomotion_policy():
    cfg = GO2_FLAT_CFG["policy"]
    ckpt_path = get_checkpoint_path(
        log_path=str(CHECKPOINT_DIR),
        run_dir=GO2_FLAT_CFG["load_run"],
        checkpoint=GO2_FLAT_CFG["load_checkpoint"],
    )

    obs_dict = {
        "default": torch.zeros((1, cfg["actor_input_dims"])),
    }
    cfg["critic_input_dims"]  # discard unusable variable
    obs_groups = {
        "policy": ["default"],
        "critic": ["default"],
    }
    policy = ActorCritic(obs_dict, obs_groups, cfg["n_actions"], **cfg)

    state_dict = torch.load(ckpt_path, weights_only=False)
    policy.load_state_dict(state_dict["model_state_dict"])

    policy = policy.to(GO2_FLAT_CFG["device"])
    return policy.act_inference  # return only the actor network


GO2_FLAT_CFG = {
    "seed": 42,
    "device": "cuda:0",
    "num_steps_per_env": 24,
    "max_iterations": 1500,
    "empirical_normalization": False,
    "policy": {
        "class_name": "ActorCritic",
        "actor_input_dims": 48,
        "critic_input_dims": 48,
        "n_actions": 12,
        "init_noise_std": 1.0,
        "actor_hidden_dims": [128, 128, 128],
        "critic_hidden_dims": [128, 128, 128],
        "activation": "elu",
    },
    "algorithm": {
        "class_name": "PPO",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.01,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 0.001,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
    },
    "save_interval": 50,
    "experiment_name": "unitree_go2_flat",
    "run_name": "",
    "logger": "tensorboard",
    "neptune_project": "isaaclab",
    "wandb_project": "isaaclab",
    "resume": False,
    "load_run": "unitree_go2_locomotion",
    "load_checkpoint": "flat_model_6800.pt",
}

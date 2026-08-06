import torch
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.models import MLPModel

from cfg.CFG import CHECKPOINT_DIR
from policy.NavPolicyv2 import convert_legacy_checkpoint


def get_locomotion_policy():
    """Build the low-level locomotion controller and load its checkpoint.

    The checkpoint was trained with ``rsl-rl-lib<4.0``, where a single ``ActorCritic`` module held both networks. Only
    the actor is needed here, so the critic parameters of the converted checkpoint are discarded.
    """
    cfg = GO2_FLAT_CFG["policy"]
    ckpt_path = get_checkpoint_path(
        log_path=str(CHECKPOINT_DIR),
        run_dir=GO2_FLAT_CFG["load_run"],
        checkpoint=GO2_FLAT_CFG["load_checkpoint"],
    )

    obs_dict = {
        "default": torch.zeros((1, cfg["actor_input_dims"])),
    }
    obs_groups = {
        "actor": ["default"],
        "critic": ["default"],
    }
    policy = MLPModel(
        obs=obs_dict,
        obs_groups=obs_groups,
        obs_set="actor",
        output_dim=cfg["n_actions"],
        hidden_dims=cfg["actor_hidden_dims"],
        activation=cfg["activation"],
        obs_normalization=False,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": cfg["init_noise_std"],
            "std_type": cfg["noise_std_type"],
        },
    )
    policy.load_state_dict(convert_legacy_checkpoint(ckpt_path)["actor_state_dict"], strict=True)

    policy = policy.to(GO2_FLAT_CFG["device"])
    policy.eval()
    # Calling the model runs the actor deterministically, like ActorCritic.act_inference() did
    return policy


GO2_FLAT_CFG = {
    "seed": 42,
    "device": "cuda:0",
    "num_steps_per_env": 24,
    "max_iterations": 1500,
    "empirical_normalization": False,
    "policy": {
        "class_name": "MLPModel",
        "actor_input_dims": 48,
        "critic_input_dims": 48,
        "n_actions": 12,
        "init_noise_std": 1.0,
        "noise_std_type": "scalar",
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
    "load_checkpoint": "flat_high_speed.pt",
}

import math

from models.policy_model import ActorCriticWithEncoders

go2_policy_cfg = {
    # Training loop parameters
    "num_steps_per_env": math.ceil((15 / (16 * 0.005))),  # 15 seconds per episode, converted to steps (with decimation)
    "save_interval": 50,  # Save a checkpoint every 50 iterations
    "max_iterations": 1000,
    "device": "cuda:0",
    "empirical_normalization": False,
    # Observation routing
    "obs_groups": {
        "policy": ["velocity_buffer", "action_buffer", "global_plan", "lidar"],  # "depth"],
        "critic": ["velocity_buffer", "action_buffer", "global_plan", "lidar"],  # "depth"],
    },
    # Policy architecture
    "policy": {
        "class_name": ActorCriticWithEncoders,
        "actor_hidden_dims": [256, 128, 128],
        "critic_hidden_dims": [256, 128, 128],
        "activation": "elu",
        "init_noise_std": 1.0,
        "noise_std_type": "log",
        "encoders_hidden_dims": {
            "velocity_buffer": [128, 128],
            "action_buffer": [128, 128],
            # "goal_relative_pos": [128, 128],
            "global_plan": [128, 128],
        },
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
    "load_run": "unitree_go2_nav",
    "load_checkpoint": "short_horizon.pt",
}

go2_policy_cfg = {
    # Training loop parameters
    "num_steps_per_env": 24,
    "save_interval": 50,
    "max_iterations": 1000,
    "device": "cuda:0",
    # Observation routing 
    "obs_groups": {  
        "policy": ["rgb"], 
        "critic": ["rgb"],  
    },  
    
    # Policy architecture
    "policy": {
        "class_name": "NavPolicyAC",
        "init_noise_std": 1.0,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
        "encoder_out_dim": 768,
        "encoder": None,
        "freeze_encoder": True,
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
        "learning_rate": 1.e-3,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
    },
    'load_run': 'unitree_go2',
    'load_checkpoint': ''
}
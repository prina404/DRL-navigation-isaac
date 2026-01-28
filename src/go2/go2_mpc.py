import torch
from go2.go2_mpc_cfg import unitree_go2_flat_cfg, unitree_go2_rough_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.modules import ActorCritic
from cfg.CFG import CHECKPOINT_DIR


def get_mpc_policy():
    cfg = unitree_go2_flat_cfg['policy']
    ckpt_path = get_checkpoint_path(log_path=str(CHECKPOINT_DIR), 
                                run_dir=unitree_go2_flat_cfg["load_run"], 
                                checkpoint=unitree_go2_flat_cfg["load_checkpoint"])

    # policy = ActorCritic( # TODO: add IsaacLab version check because ActorCritic API changed
    #     cfg.pop('actor_input_dims'),
    #     cfg.pop('critic_input_dims'),
    #     cfg.pop('n_actions'),
    #     **cfg
    # )
    obs_dict = {
        "default": torch.zeros((1, cfg.pop('actor_input_dims'))),
    }
    cfg.pop('critic_input_dims') # discard unusable variable 
    obs_groups = {
        "policy": ["default"],
        "critic": ["default"],
    }
    policy = ActorCritic(
        obs_dict,
        obs_groups,
        cfg.pop('n_actions'),
        **cfg
    )


    state_dict = torch.load(ckpt_path, weights_only=False)
    policy.load_state_dict(state_dict['model_state_dict'])

    policy = policy.to(unitree_go2_flat_cfg['device'])
    return policy.act_inference # return only the actor network
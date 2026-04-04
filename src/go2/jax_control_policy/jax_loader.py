import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import orbax.checkpoint
from flax.training.train_state import TrainState
from pathlib import Path
import json
import optax

class Policy(nn.Module):
    std_dev: float
    policy_mean_abs_clip: float
    policy_std_min_clip: float
    policy_std_max_clip: float
    action_space: int

    @nn.compact
    def __call__(self, x):
        policy_mean = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        policy_mean = nn.LayerNorm()(policy_mean)
        policy_mean = nn.elu(policy_mean)
        policy_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(policy_mean)
        policy_mean = nn.elu(policy_mean)
        policy_mean = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(policy_mean)
        policy_mean = nn.elu(policy_mean)
        policy_mean = nn.Dense(self.action_space, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(policy_mean)
        policy_mean = jnp.clip(policy_mean, -self.policy_mean_abs_clip, self.policy_mean_abs_clip)

        return policy_mean
    
def get_policy():
    curr_path = Path(__file__).parent
    cfg_dir = curr_path / "model"
    cfg_path = cfg_dir / "config_algorithm.json"
    ckpt_path = cfg_dir / "checkpoint"
    with open(cfg_path, "r") as f:
        algorithm_config = json.load(f)
    
    policy = Policy(
        algorithm_config["std_dev"],
        algorithm_config["policy_mean_abs_clip"],
        algorithm_config["policy_std_min_clip"], algorithm_config["policy_std_max_clip"],
        12
        )
    
    key = jax.random.PRNGKey(0)
    key, policy_key = jax.random.split(key, 2)
    dummy_state = np.zeros((1, 48))
    
    policy_state = TrainState.create(
        apply_fn=policy.apply,
        params=policy.init(policy_key, dummy_state),
        tx=optax.chain(
            optax.clip_by_global_norm(0.0),
            optax.inject_hyperparams(optax.adam)(learning_rate=lambda count: 0.0),
        )
    )

    policy.apply(policy_state.params, dummy_state)
    target = {
        "policy": policy_state,
    }

    check_point_handler = orbax.checkpoint.PyTreeCheckpointHandler(aggregate_filename=str(ckpt_path))
    checkpointer = orbax.checkpoint.Checkpointer(check_point_handler)
    checkpoint = checkpointer.restore(cfg_dir, item=target)
    policy_state = checkpoint["policy"]

    return policy, policy_state.params
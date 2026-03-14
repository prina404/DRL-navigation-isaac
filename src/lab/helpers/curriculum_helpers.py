import torch

from lab.MyEnv import MyEnv


## Curriculum manager calls this only when one or more envs are reset, so the internal
## episode_length_buf contains the lengths of the episodes just before reset.
def update_collision_weight(
    env: MyEnv, env_ids: torch.Tensor, term_name: str, weight_step_size: float, max_weight: float
) -> float:
    if not hasattr(env, "_running_mean_var"):  # ensure attribute exists
        env._running_mean_var = 0.0
        env._last_update_step = 0

    env.path_manager.update_episode_durations(env_ids, env.episode_length_buf[env_ids])

    current_step = env.common_step_counter
    term_cfg = env.reward_manager.get_term_cfg(term_name)

    # check curriculum conditions once every 200 steps
    if current_step - env._last_update_step < 1000 or current_step == 0:
        return term_cfg.weight
    env._last_update_step = current_step

    history = env.path_manager.duration_history
    if torch.any(history == 0.0):
        # not enough data yet to compute variance, skip update
        return term_cfg.weight

    duration_variance = history.var(dim=-1)
    average_variance = duration_variance.mean()

    if env._running_mean_var == 0.0:  # initialize running mean variance on first update
        env._running_mean_var = average_variance
    else:
        env._running_mean_var = 0.8 * env._running_mean_var + 0.2 * average_variance

    if average_variance < env._running_mean_var * 0.9:
        term_cfg.weight += weight_step_size

    # Ensure the weight doesn't exceed the maximum allowed value
    term_cfg.weight = min(term_cfg.weight, max_weight)

    return term_cfg.weight


def collision_termination_threshold(env: MyEnv, env_ids: torch.Tensor) -> float:
    if env.common_step_counter % 5000 == 0:
        env.collision_termination_thresh = max(env.collision_termination_thresh - 0.05, 0.1)
    return env.collision_termination_thresh


def nominal_policy_weight(env: MyEnv, env_ids: torch.Tensor, max_step: int = 150 * 300) -> float:
    # linearly decay nominal policy weight from 1.0 to 0.0 over max_step steps
    progress = env.common_step_counter / max_step
    env.nominal_weight = max(1.0 - progress, 0.0)
    return env.nominal_weight

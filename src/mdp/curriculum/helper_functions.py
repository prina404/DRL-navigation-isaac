import torch

from navigation_env.NavigationEnv import NavEnv


## Curriculum manager calls this only when one or more envs are reset, so the internal
## episode_length_buf contains the lengths of the episodes just before reset.
def update_collision_weight(
    env: NavEnv, env_ids: torch.Tensor, weight_step_size: float, max_weight: float, episode_start: int = 0
) -> float:
    if not hasattr(env, "_running_mean_var"):  # ensure attribute exists
        env._running_mean_var = 0.0
        env._last_update_step = 0

    env.path_manager.update_episode_durations(env_ids, env.episode_length_buf[env_ids])

    current_step = env.common_step_counter
    current_episode = env.episode_counter.mean().item()
    term_cfg = env.reward_manager.get_term_cfg("collision")

    # check curriculum conditions once every 1000 steps
    if current_step - env._last_update_step < 1000 or current_episode < episode_start:
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

    if average_variance < env._running_mean_var * 0.95:
        term_cfg.weight += weight_step_size

    # Ensure the weight doesn't exceed the maximum allowed value
    term_cfg.weight = min(term_cfg.weight, max_weight)

    return term_cfg.weight


def _progress_coeff(env: NavEnv, episode_start: int, episode_end: int) -> float:
    current_episode = env.episode_counter.mean().item() * 0.75  # TODO: check correction factor
    if current_episode < episode_start:
        return 1.0
    elif current_episode > episode_end:
        return 0.0
    else:
        return 1.0 - (current_episode - episode_start) / (episode_end - episode_start)


def collision_termination_threshold(env: NavEnv, env_ids: torch.Tensor, episode_start: int = 50, episode_end: int = 250) -> float:
    termination_coeff = _progress_coeff(env, episode_start, episode_end)
    env.collision_termination_thresh = max(termination_coeff, 0.01)
    return env.collision_termination_thresh


def nominal_policy_weight(env: NavEnv, env_ids: torch.Tensor, episode_start: int = 50, episode_end: int = 250) -> float:
    # linearly decay nominal policy weight from 1.0 to 0.0 over the ep interval
    env.nominal_weight = _progress_coeff(env, episode_start, episode_end)
    return env.nominal_weight

def add_path_obstacles(env: NavEnv, env_ids: torch.Tensor, episode_start: int = 50, episode_end: int = 250) -> None:
    env.obstacle_prob = 1 - _progress_coeff(env, episode_start, episode_end)
    return env.obstacle_prob

from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass
import mdp.terminations.helper_functions as termination
import mdp.rewards.helper_functions as rewards
import isaaclab.envs.mdp as isaac_mdp


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # If the global path is shorter than distance_traveled termination threshold,
    # we terminate on goal reached condition
    goal_reached = TerminationTermCfg(
        func=rewards.is_goal_reached,
        params={"threshold_m": 0.3},
    )

    timeout = TerminationTermCfg(
        func=isaac_mdp.time_out,
        # time_out=True,
    )

    collision_after_threshold = TerminationTermCfg(func=termination.collision_after_threshold_termination)

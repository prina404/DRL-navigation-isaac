from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import mdp.rewards.helper_functions as rewards


@configclass
class RewardsCfg:
    goal_reached = RewTerm(
        func=rewards.is_goal_reached,
        weight=200.0,
        params={"threshold_m": 0.3},
    )

    heading = RewTerm(func=rewards.robot_heading_reward, weight=1.0)

    collision = RewTerm(
        func=rewards.penalty_collision,
        weight=0.2,
        params={"force_thresh": 1.0, "penalty": -5.0},
    )

    action_smoothness = RewTerm(
        func=rewards.action_smoothness_penalty,
        weight=1.0,
    )

    collision_after_threshold = RewTerm(
        func=rewards.collision_after_threshold_reward,
        weight=1.0,
        params={"penalty": -50.0},
    )

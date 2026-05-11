from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass

import mdp.curriculum.helper_functions as curriculum


@configclass
class CurriculumCfg:
    obstacle_weight = CurriculumTermCfg(
        func=curriculum.update_collision_weight,
        params={
            "weight_step_size": 0.05,
            "max_weight": 1.0,
            "episode_start": 50,  # start curriculum at 100 episodes
        },
    )

    collision_thresh = CurriculumTermCfg(
        func=curriculum.collision_termination_threshold,
        params={
            "episode_start": 300,  # start increasing threshold at 300 episodes
            "episode_end": 1000,  # end increasing threshold at 1000 episodes
        },
    )

    nominal_weight = CurriculumTermCfg(
        func=curriculum.nominal_policy_weight,
        params={
            "episode_start": 100,  # start decaying at 100 episodes
            "episode_end": 400,  # end decaying at 400 episodes
        },
    )

@configclass
class PathObstaclesCurriculumCfg(CurriculumCfg):
    """Curriculum config that adds path obstacles after a certain number of episodes"""

    add_obstacles = CurriculumTermCfg(
        func=curriculum.add_path_obstacles,
        params={
            "episode_start": 100,  # add obstacles after n episodes
            "episode_end": 600,  # full obstacle probability
        }
    )
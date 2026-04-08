from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass
import mdp.curriculum.helper_functions as curriculum

@configclass
class CurriculumCfg:
    obstacle_weight = CurriculumTermCfg(
        func=curriculum.update_collision_weight,
        params={
            "weight_step_size": 0.1,
            "max_weight": 5.0,
            "episode_start": 50,  # start curriculum at 50 episodes
        },
    )

    collision_thresh = CurriculumTermCfg(
        func=curriculum.collision_termination_threshold,
        params={
            "episode_start": 50,  # start increasing threshold at 50 episodes
            "episode_end": 250,  # end increasing threshold at 250 episodes
        },
    )

    nominal_weight = CurriculumTermCfg(
        func=curriculum.nominal_policy_weight,
        params={
            "episode_start": 50,  # start decaying at 50 episodes
            "episode_end": 200,  # end decaying at 200 episodes
        },
    )

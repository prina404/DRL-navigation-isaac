from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

import mdp.observations.helper_functions as observations
from tasks.task_common_params import PARAMS


# ------------ GROUPS ------------
@configclass
class _ActionGroup(ObsGroup):
    # The last 5 executed velocity commands (m/s, rad/s), after the squash, the nominal blend and the speed cap, not the
    # raw policy actions. Cleared on reset.
    action_buffer = ObsTerm(func=observations.get_velocity_command, history_length=5)


@configclass
class _MaxSpeedGroup(ObsGroup):
    # the speed cap of the robot [m/s], fed to the policy head unencoded
    max_speed = ObsTerm(func=observations.get_max_speed)


@configclass
class _VelocityGroup(ObsGroup):
    # The last 10 velocities, cleared on reset. Like a state estimate, each one is noisy, but its noise is drawn once,
    # as it enters the history, and then stays the same.
    velocity_buffer = ObsTerm(
        func=observations.get_base_velocity,
        history_length=10,
        noise=GaussianNoiseCfg(std=PARAMS["velocity_noise_std"]),
    )

    def __post_init__(self):
        self.enable_corruption = True


@configclass
class _LidarGroup(ObsGroup):
    lidar = ObsTerm(func=observations.get_lidar, params={"num_obstacles": 256, "noise_std": PARAMS["lidar_noise_std"]})


@configclass
class _GoalGroup(ObsGroup):
    goal_relative_pos = ObsTerm(func=observations.get_goal_relative_position, params={"local_goal_points_forward": 3})


@configclass
class _GlobalPlanGroup(ObsGroup):
    global_plan = ObsTerm(
        func=observations.get_path_obs,
        params={
            "num_points_forward": 10,
            "normalize": True,
        },
    )


# ------------ FULL OBSERVATION CONFIGS ------------
@configclass
class LidarOnlyCfg:
    action_buffer = _ActionGroup()
    velocity_buffer = _VelocityGroup()
    lidar = _LidarGroup()
    global_plan = _GlobalPlanGroup()
    max_speed = _MaxSpeedGroup()

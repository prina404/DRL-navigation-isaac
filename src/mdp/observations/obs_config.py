
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
import mdp.observations.helper_functions as observations

# ------------ GROUPS ------------
@configclass
class _ActionGroup(ObsGroup):
    action_buffer = ObsTerm(func=observations.get_previous_actions)

@configclass
class _VelocityGroup(ObsGroup):
    velocity_buffer = ObsTerm(func=observations.get_previous_velocities)

@configclass
class _VisionGroup(ObsGroup):
    vision = ObsTerm(func=observations.VisionEncoder(encoder="vit"))

@configclass
class _DepthGroup(ObsGroup):
    depth = ObsTerm(func=observations.DepthEncoder(encoder="resnet"))

@configclass
class _LidarGroup(ObsGroup):
    lidar = ObsTerm(func=observations.get_lidar, params={"num_obstacles": 256})

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


@configclass
class LidarVisionCfg:
    action_buffer = _ActionGroup()
    velocity_buffer = _VelocityGroup()
    vision = _VisionGroup()
    lidar = _LidarGroup()
    global_plan = _GlobalPlanGroup()

@configclass
class LidarDepthCfg:
    action_buffer = _ActionGroup()
    velocity_buffer = _VelocityGroup()
    depth = _DepthGroup()
    lidar = _LidarGroup()
    global_plan = _GlobalPlanGroup()
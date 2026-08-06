import isaaclab.envs.mdp as isaac_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_term_cfg import EventTermCfg
from isaaclab.utils import configclass

from mdp.events import helper_functions as events


@configclass
class BaseEventsCfg:
    """Base event config. Can be used for simple environments without doors"""

    reset_pos = EventTermCfg(func=events.teleport_on_reset, mode="reset")

    reset_joints = EventTermCfg(
        func=isaac_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    replan = EventTermCfg(func=events.replan_global_plan, mode="interval", interval_range_s=(2.0, 2.0))

    update_counters = EventTermCfg(func=events.update_episode_counters, mode="reset")


@configclass
class FullEventsCfg(BaseEventsCfg):
    """Full event config with door randomization."""

    randomize_doors = EventTermCfg(
        func=events.randomize_door_positions,
        mode="reset",
    )

    randomize_indoor_lights = EventTermCfg(
        func=events.randomize_lights_on_off,
        mode="reset",
        params={
            "on_probability": 0.75,
        },
    )

    randomize_dome_light = EventTermCfg(
        func=events.randomize_distant_light,
        mode="interval",
        interval_range_s=(10.0, 15.0),
    )

    # TODO: re-enable once events.randomize_chair_positions is finished. It is still a stub
    # (no sampling logic, `_collides` unimplemented) and references `env.scene.env_sizes`,
    # which is not an InteractiveScene attribute, so every reset raises AttributeError.
    # randomize_chairs = EventTermCfg(
    #     func=events.randomize_chair_positions,
    #     mode="reset",
    # )
    
@configclass
class PathObstaclesEventsCfg(FullEventsCfg):
    """Event config that adds path obstacles after a certain number of episodes"""
    
    randomize_obstacles = EventTermCfg(
        func=events.move_obstacle_on_path,
        mode="reset",
    )

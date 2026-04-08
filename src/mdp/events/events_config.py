
from isaaclab.utils import configclass
from isaaclab.managers.manager_term_cfg import EventTermCfg
from mdp.events import helper_functions as events
import isaaclab.envs.mdp as isaac_mdp
from isaaclab.managers import SceneEntityCfg

@configclass
class BaseEventsCfg:
    ''' Base event config. Can be used for simple environments without doors'''

    reset_pos = EventTermCfg(func=events.teleport_on_reset, mode="reset")

    reset_joints = EventTermCfg(
        func=isaac_mdp.reset_joints_by_scale,
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
    ''' Full event config with door randomization.'''
    
    randomize_doors = EventTermCfg(
        func=events.randomize_door_positions,
        mode="reset",
    )

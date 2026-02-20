import time

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.manager_term_cfg import EventTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCfg, RayCasterCfg, patterns
from isaaclab.sensors.camera import TiledCameraCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from isaaclab.sim.schemas import CollisionPropertiesCfg
from isaaclab.utils import configclass
from loguru import logger

import lab.helpers.action_helpers as actions
import lab.helpers.observation_helpers as observations
import lab.helpers.reward_helpers as rewards
import lab.helpers.event_helpers as events

from cfg.CFG import SCENE_USD_PATH

from go2.go2_articulation_cfg import UNITREE_GO2_CFG
import torch


@configclass
class Go2SimCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(500.0, 500.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.01)),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # Go2 Robot
    unitree_go2: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Go2",
    )

    body_collision_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Go2/base",
        history_length=1,
        track_air_time=False,
    )

    # camera = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Go2/base/front_cam",
    #     width=128,
    #     height=128,
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(0.4, 0.0, 0.0),  
    #         convention="world",  
    #     ),
    #     data_types=["rgb"],
    #     spawn=PinholeCameraCfg(focal_length=1.5, horizontal_aperture=4, clipping_range=(0.1, 100.0)),
    # )

    lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Go2/radar",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.4, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=10,
            vertical_fov_range=(0, 30),
            horizontal_fov_range=(0, 360),
            horizontal_res=3.0,
        ),
        debug_vis=False,
        mesh_prim_paths=[
            "/World/ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                # TODO check if we can simplify prim_expr
                prim_expr="{ENV_REGEX_NS}/environment/Meshes/dynamic_objects/other/door_.*/Meshes/door_.*/group_.*",
                is_shared=False,  # door prims have hinge joints, so we need to track their mesh transforms for accurate raycasting
                track_mesh_transforms=True,
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/environment/Meshes/static_objects",
                is_shared=True, 
                track_mesh_transforms=False,
            ),
        ],
    )

    # # TODO: static env loading ok, check if setting a subset of kinematic objects is possible
    environment = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/environment",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(SCENE_USD_PATH),
            # rigid_props=sim_utils.RigidBodyPropertiesCfg( # should be handled in preprocessing
            #     kinematic_enabled=True
            # ),  # Big impact on performance if True, but env is now static
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
        ),
    )


@configclass
class ObservationsCfg:
    @configclass
    class ActionGroup(ObsGroup):
        action_buffer = ObsTerm(func=observations.get_previous_actions)

    @configclass
    class VelocityGroup(ObsGroup):
        velocity_buffer = ObsTerm(func=observations.get_previous_velocities)

    @configclass
    class VisionGroup(ObsGroup):
        vision = ObsTerm(func=observations.VisionEncoder(encoder="vit"))

    @configclass
    class LidarGroup(ObsGroup):
        lidar = ObsTerm(func=observations.get_lidar, params={"num_obstacles": 50})

    @configclass
    class GlobalPlanGroup(ObsGroup):
        global_plan = ObsTerm(
            func=observations.get_path_obs,
            params={
                "num_points_forward": 10,
                "normalize": True,
            },
        )
    
    @configclass
    class GoalGroup(ObsGroup):
        goal_relative_pos = ObsTerm(func=observations.get_goal_relative_position)

    action_buffer = ActionGroup()
    velocity_buffer = VelocityGroup()
    goal_relative_pos = GoalGroup()
    #vision = VisionGroup()
    lidar = LidarGroup()
    # global_plan = GlobalPlanGroup()


@configclass
class ActionsCfg:
    mpc_cmd = ActionTermCfg(
        class_type=actions.Go2MPCPolicyAction,
        asset_name="unitree_go2",
    )


@configclass
class RewardsCfg:
    # goal_reached = RewTerm(
    #     func=rewards.is_goal_reached,
    #     weight=50.0,
    #     params={"threshold_m": 0.3},
    # )

    # distance_to_goal = RewTerm(
    #     func=rewards.reward_distance_to_goal,
    #     weight=1.0,
    # )

    # penalty_still = RewTerm(
    #     func=rewards.penalty_still,
    #     weight=0.5,
    #     params={"speed_thresh": 0.2, "penalty": -0.1},
    # )

    # heading = RewTerm(
    #     func=rewards.robot_heading_reward,
    #     weight=3.0,
    # )

    collision = RewTerm(
        func=rewards.penalty_collision_base,
        weight=1.0,
        params={"force_thresh": 1.0, "penalty": -10.0},
    )

    action_smoothness = RewTerm(
        func=rewards.action_smoothness_penalty,
        weight=0.4,
    )

    time_penalty = RewTerm(
        func=rewards.time_penalty,
        weight=0.2,
    )

    # obstacle_proximity = RewTerm(
    #     func=rewards.penalty_obstacle_proximity,
    #     weight=0.5,
    #     params={"max_penalty": -2.0},
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    goal_reached = DoneTerm(
        func=rewards.is_goal_reached,
        params={"threshold_m": 0.3},
    )
    timeout = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )


@configclass
class EventsCfg:

    reset_pos = EventTermCfg(func=events.teleport_on_reset, mode="reset")

    reset_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="unitree_go2"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_doors = EventTermCfg(
        func=events.randomize_door_positions,
        mode="reset",
    )

    replan = EventTermCfg(
        func=events.replan_global_plan,
        mode="interval",
        interval_range_s=(2.0, 2.0)
    )



@configclass
class Go2EnvCfg(ManagerBasedRLEnvCfg):

    scene = Go2SimCfg(num_envs=1, env_spacing=18.0)

    actions = ActionsCfg()
    observations = ObservationsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventsCfg()

    def __post_init__(self) -> None:
        self.sim.dt = 0.005
        self.sim.device = "cuda:0"
        self.sim.use_fabric = True
        self.sim.render = sim_utils.RenderCfg(
            rendering_mode="performance",
            enable_translucency=True,
            enable_reflections=True,
            enable_shadows=True,
            enable_ambient_occlusion=True,
            dlss_mode="3",  # defaults to 1 in performance mode
        )
        self.episode_length_s = 15.0
        
        self.decimation = 16
        self.sim.render_interval = self.decimation

        logger.info("Go2 EnvCfg initialized")

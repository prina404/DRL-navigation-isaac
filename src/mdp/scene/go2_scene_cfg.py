import isaaclab.sim as sim
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCollectionCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import (
    ContactSensorCfg,
    MultiMeshRayCasterCfg,
    RayCasterCfg,
    patterns,
)
from isaaclab.sensors.camera import TiledCameraCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from isaaclab.utils import configclass

from cfg.CFG import get_scene_usd_path
from mdp.scene.moving_obstacles_cfg import get_obstacles_cfg
from mdp.scene.robot_cfg import UNITREE_GO2_CFG


@configclass
class Go2BaseCfg(InteractiveSceneCfg):
    """Base config. Do not instantiate this"""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(500.0, 500.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.01)),
    )

    # Go2 Robot
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Go2",
    )

    body_collision_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Go2/base",
        history_length=1,
        track_air_time=False,
    )

    hip_collision_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Go2/.*_thigh_protector",
        history_length=1,
        track_air_time=False,
    )



@configclass
class Go2EmptySceneCfg(Go2BaseCfg):
    """Empty plane with robot only"""

    sky_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1000.0),
    )

    lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Go2/radar",
        update_period=0.05,  # 20Hz
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
        ],
    )


@configclass
class Go2FullSceneLidarCfg(Go2BaseCfg):
    """Full indoor scene, robot has lidar only"""

    lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Go2/radar",
        update_period=0.05,  # 20Hz
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
                prim_expr="{ENV_REGEX_NS}/environment/Meshes/dynamic_objects/other/chair_.*/Meshes/chair_.*",
                is_shared=False,  
                track_mesh_transforms=True,
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/environment/Meshes/static_objects",
                is_shared=True,
                track_mesh_transforms=False,
            ),
        ],
    )

    # path_obstacles = RigidObjectCollectionCfg(
    #     rigid_objects=get_obstacles_cfg(),
    # )

    def __post_init__(self):
        # env config in post_init to ensure usd path is updated at runtime
        self.environment = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/environment",
            spawn=sim.UsdFileCfg(
                usd_path=str(get_scene_usd_path()),
            ),
        )


CAMERA_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Go2/base/front_cam",
    width=128,
    height=128,
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.4, 0.0, 0.0),
        convention="world",
    ),
    data_types=["rgb", "depth"],
    spawn=PinholeCameraCfg(focal_length=1.5, horizontal_aperture=4, clipping_range=(0.1, 100.0)),
)


@configclass
class Go2EmptyVisionCfg(Go2EmptySceneCfg):
    """Empty plane with robot only, robot has RGBD camera"""
    #lidar = None
    camera = CAMERA_CFG


@configclass
class Go2FullVisionCfg(Go2FullSceneLidarCfg):
    """Full indoor scene, robot has RGBD camera"""
    #lidar = None
    camera = CAMERA_CFG

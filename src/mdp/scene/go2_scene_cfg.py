from collections.abc import Callable
from dataclasses import MISSING

import isaaclab.sim as sim
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCollectionCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import (
    ContactSensorCfg,
    MultiMeshRayCasterCfg,
    RayCasterCfg,
    patterns,
)
from isaaclab.utils import configclass
from pxr import Usd

from cfg.CFG import get_map_origins, get_split_usd_path
from mdp.scene.moving_obstacles_cfg import get_obstacles_cfg
from mdp.scene.multi_map_ray_caster import MultiMapRayCasterCfg
from mdp.scene.robot_cfg import UNITREE_GO2_CFG


@configclass
class Go2BaseCfg(InteractiveSceneCfg):
    """Base config. Do not instantiate this"""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(500.0, 500.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.01)),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1000.0),
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

    # the knee sphere is a collider of the thigh link, just above the calf joint (the calf's own collider is disabled)
    knee_collision_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Go2/.*_thigh",
        history_length=1,
        track_air_time=False,
    )


@configclass
class Go2EmptySceneCfg(Go2BaseCfg):
    """Empty plane with robot only"""

    lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Go2/radar",
        spawn=None,  # the radar link of the robot; a spawned frame would clash with the robot when cloning per map
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


def spawn_map_objects(prim_path: str, cfg: "MapObjectsCfg", translation=None, orientation=None, **kwargs) -> Usd.Prim:
    """Spawn the doors and chairs of a map at the map's origin, and tag them with the map's name.

    Every robot has its environment origin at the world origin, so the offset that moves the objects into their map
    must come from the spawner (the variants of a `MultiAssetSpawnerCfg` are all spawned with the same translation).
    The tag survives cloning, and is how `NavEnv` finds out which map each robot was assigned to.
    """
    prim = sim.spawn_from_usd(prim_path, cfg, translation=cfg.map_origin, orientation=orientation, **kwargs)
    prim.SetCustomDataByKey("map_name", cfg.map_name)
    return prim


@configclass
class MapObjectsCfg(sim.UsdFileCfg):
    """The dynamic objects of one map, spawned at the origin of that map"""

    func: Callable = spawn_map_objects
    map_name: str = MISSING
    map_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)


@configclass
class Go2FullSceneLidarCfg(Go2BaseCfg):
    """Every loaded map, robot has lidar only.

    The static geometry of each map is a global prim, shared by all the robots navigating it, while each robot has its
    own copy of the doors and chairs of its map. Robots do not collide with, nor see, the other environments.
    """

    lidar = MultiMapRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Go2/radar",
        spawn=None,  # the radar link of the robot; a spawned frame would clash with the robot when cloning per map
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
                prim_expr="{ENV_REGEX_NS}/dynamic_objects/other/door_.*/Meshes/door_.*/group_.*",
                track_mesh_transforms=True,  # door prims have hinge joints, so we need to track their mesh transforms
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/dynamic_objects/other/chair_.*/Meshes/chair_.*",
                track_mesh_transforms=True,
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/maps",  # the static geometry of every map, merged into a single mesh
                track_mesh_transforms=False,
            ),
        ],
    )

    # path_obstacles = RigidObjectCollectionCfg(
    #     rigid_objects=get_obstacles_cfg(),
    # )

    def __post_init__(self):
        # maps are read in post init, once the entry script has picked them
        map_origins = get_map_origins()
        for map_name, map_origin in map_origins.items():
            static_cfg = AssetBaseCfg(
                prim_path=f"/World/maps/{map_name}",
                spawn=sim.UsdFileCfg(usd_path=str(get_split_usd_path(map_name, "static"))),
                init_state=AssetBaseCfg.InitialStateCfg(pos=map_origin),
            )
            setattr(self, f"{map_name}_static", static_cfg)

        self.dynamic_objects = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/dynamic_objects",
            spawn=sim.MultiAssetSpawnerCfg(
                assets_cfg=[
                    MapObjectsCfg(
                        usd_path=str(get_split_usd_path(map_name, "dynamic")),
                        map_name=map_name,
                        map_origin=map_origin,
                    )
                    for map_name, map_origin in map_origins.items()
                ],
                random_choice=False,
            ),
        )

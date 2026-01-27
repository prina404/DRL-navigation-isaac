from typing import Callable

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch
import time
import torch.nn as nn
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.schemas import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.managers.manager_term_cfg import EventTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns, MultiMeshRayCasterCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.sensors.camera import TiledCameraCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg

from isaaclab.utils import configclass
#from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from lab.go2_articulation_cfg import UNITREE_GO2_CFG
from loguru import logger
from rsl_rl.modules import ActorCritic

from lab.vision_encoder import ViTEncoder
from utils.gridmap import generate_ogm_on_reset
from cfg.CFG import SCENE_USD_PATH


@configclass
class Go2SimCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300.0, 300.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),
    )

    # lights
    # # Lights
    # light = AssetBaseCfg(
    #     prim_path="/World/Light",
    #     spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    # )
    sky_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    # dome_light = AssetBaseCfg(
    #     prim_path="/World/DomeLight",
    #     spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    # )

    # Define collision disable config

    # Go2 Robot
    unitree_go2: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Go2",
    )
    # Go2 foot contact sensor
    # contact_forces = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Go2/.*_foot", history_length=3, track_air_time=True
    # )

    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Go2/base/front_cam",
        width =320,
        height=240,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.4, 0.0, 0.0),  # Your desired offset
            convention="world",  # Coordinate frame convention
        ),
        data_types=["rgb"],
        spawn=PinholeCameraCfg(
            focal_length=1.5, horizontal_aperture=4, clipping_range=(0.1, 100.0)
        ),
    )

    # lidar = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Go2/radar",
    #     update_period=0.02,
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.4, 0.0, 0.0)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.LidarPatternCfg(channels=20, vertical_fov_range=(-30, 30), horizontal_fov_range=(0, 360), horizontal_res=3.0),
    #     debug_vis=True,
    #     mesh_prim_paths=["/{ENV_REGEX_NS}/environment"],
    # )

    lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Go2/radar",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.4, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=20,
            vertical_fov_range=(-30, 30),
            horizontal_fov_range=(0, 360),
            horizontal_res=3.0,
        ),
        debug_vis=True,
        mesh_prim_paths=[
            "/World/ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/environment",
                is_shared=True,     # TODO: this reduces VRAM usage significantly, but can we make it work with semi-static objects?
                track_mesh_transforms=False,
            ),
        ],
    )

    # TODO: check if it is possible to preprocess the USD to reduce physx errors during collision baking
    environment = AssetBaseCfg(                     
        prim_path = "{ENV_REGEX_NS}/environment",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(SCENE_USD_PATH),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
        )
    )


class VisionEncoder:
    def __init__(
        self, encoder: nn.Module, device: torch.device, normalize: bool = True
    ):
        self.encoder = encoder.to(device)
        self.normalize = normalize

    @torch.no_grad()
    def __call__(self, env: RslRlVecEnvWrapper) -> torch.Tensor:
        cam = env.scene["camera"]
        rgb = cam.data.output["rgb"]  # (N, H, W, C)
        rgb = rgb.permute(0, 3, 1, 2)  # Convert to N, C, H, W for ViT
        if self.normalize:
            rgb = rgb.to(dtype=torch.float32) / 255.0
        embedding = self.encoder(rgb)
        return embedding

def _get_dummy_embedding(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros((env.num_envs, 768), device=env.device, dtype=torch.float32)

def _get_lidar(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros((env.num_envs, 50), device=env.device, dtype=torch.float32)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        vision = ObsTerm(
            func=VisionEncoder(
                encoder=ViTEncoder(),
                device="cuda:0",
                normalize=True,
            ),
            #func=_get_dummy_embedding,
        )
        lidar_points = ObsTerm(func=_get_lidar)

        def __post_init__(self) -> None:
            self.concatenate_terms = True

    policy = PolicyCfg()
    critic = PolicyCfg()



class Go2MPCPolicyAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._robot_cfg = SceneEntityCfg(name=cfg.asset_name)
        self._cmd = torch.zeros((self.num_envs, 3), device=self.device)
        self._last_joint_action = None
        self._decimation = 1  # Run MPC every 1*dt seconds

        self.mpc: ActorCritic = getattr(cfg, "mpc_policy", None)  # type: ignore
        if self.mpc is None:
            raise ValueError("MPC policy must be provided in the environment config.")
        self.physics_step_counter = 0

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._cmd

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._cmd

    def process_actions(self, actions: torch.Tensor):
        self._cmd[:] = actions

    def _build_low_level_obs(self) -> torch.Tensor:
        base_lin_vel = mdp.base_lin_vel(self._env, asset_cfg=self._robot_cfg)
        base_ang_vel = mdp.base_ang_vel(self._env, asset_cfg=self._robot_cfg)
        projected_gravity = mdp.projected_gravity(self._env, asset_cfg=self._robot_cfg)

        base_vel_cmd = self._cmd + torch.tensor([1, 0, 0.1], device=self._cmd.device)

        joint_pos = mdp.joint_pos_rel(self._env, asset_cfg=self._robot_cfg)
        joint_vel = mdp.joint_vel_rel(self._env, asset_cfg=self._robot_cfg)

        if self._last_joint_action is None:
            self._last_joint_action = torch.zeros_like(joint_pos)

        return torch.cat(
            [
                base_lin_vel,
                base_ang_vel,
                projected_gravity,
                base_vel_cmd,
                joint_pos,
                joint_vel,
                self._last_joint_action,
            ],
            dim=-1,
        )

    def apply_actions(self):
        # Run low-level policy every sim step by default
        if self.physics_step_counter % self._decimation == 0 or self._last_joint_action is None:
            with torch.no_grad():
                mpc_obs = self._build_low_level_obs()
                mpc_obs = {"default": mpc_obs}  # Wrap in dict for ActorCritic
                self._last_joint_action = self.mpc(mpc_obs)
            
        self.physics_step_counter += 1
        mpc_action = self._last_joint_action

        robot = self._env.scene[self.cfg.asset_name]
        q0 = robot.data.default_joint_pos
        q_des = q0 + mpc_action * 0.25
        robot.set_joint_position_target(q_des)


@configclass
class ActionsCfg:
    mpc_cmd = ActionTermCfg(
        class_type=Go2MPCPolicyAction,
        asset_name="unitree_go2",
    )


@configclass
class RewardsCfg:
    pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    pass


@configclass
class EventsCfg:
    create_occupancy_gridmap = EventTermCfg(
        func=generate_ogm_on_reset,
        mode="interval",  # No need to specify other params, env & env_ids are passed by default
        interval_range_s=(3.0, 3.0),
        params={
            "map_resolution": 0.05,
        },
    )


@configclass
class Go2EnvCfg(ManagerBasedRLEnvCfg):

    scene = Go2SimCfg(num_envs=1, env_spacing=15.0)

    actions = ActionsCfg()
    observations = ObservationsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    # events = EventsCfg()

    def __post_init__(self) -> None:
        self.sim.dt = 0.005
        self.sim.device = "cuda:0"
        self.sim.use_fabric = True
        self.episode_length_s = 20.0

        logger.info("Go2 EnvCfg initialized")

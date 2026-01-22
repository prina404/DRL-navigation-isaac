from typing import Callable

from loguru import logger
from isaaclab.managers.manager_term_cfg import EventTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.sensors.camera import TiledCameraCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ActionTerm, SceneEntityCfg, ActionTermCfg
import torch
from utils.gridmap import generate_ogm_on_reset


@configclass
class Go2SimCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300.0, 300.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),
    )

    # lights
    # Lights
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    # dome_light = AssetBaseCfg(
    #     prim_path="/World/DomeLight",
    #     spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    # )

    # Go2 Robot
    unitree_go2: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Go2")
    # Go2 foot contact sensor
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Go2/.*_foot", history_length=3, track_air_time=True
    )

    camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Go2/base/front_cam",
        width=320,
        height=240,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.4, 0.0, 0.0),  # Your desired offset
            convention="world",  # Coordinate frame convention
        ),
        data_types=["rgb"],
        spawn=PinholeCameraCfg(
            focal_length=1.5, horizontal_aperture=4, clipping_range=(0.1, 1000.0)
        ),
    )

    # Go2 height scanner
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Go2/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


def _get_rgb(env: ManagerBasedRLEnv) -> torch.Tensor:
    cam = env.scene["camera"]  # <-- IMPORTANT: access by name, not env.scene.camera

    rgb = cam.data.output.get("policy", None)
    if rgb is None:
        # ObservationManager probes shapes during construction; return a correctly-shaped placeholder.
        # TiledCamera outputs per-env batch, so shape is [num_envs, H, W, 3]
        return torch.zeros(
            (env.num_envs, cam.cfg.height, cam.cfg.width, 3),
            device=env.device,
            dtype=torch.float32,
        )
    # Return a [N, H, W, 3] tensor of RGB images in [0, 1] range
    return rgb.to(dtype=torch.float32) / 255.0


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        policy = ObsTerm(func=_get_rgb)

        def __post_init__(self) -> None:
            self.concatenate_terms = True

    policy = PolicyCfg()


class Go2MPCPolicyAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._robot_cfg = SceneEntityCfg(name=cfg.asset_name)
        self._cmd = torch.zeros((self.num_envs, 3), device=self.device)
        self._last_joint_action = None

        self.mpc: Callable = getattr(cfg, "mpc_policy", None)  # type: ignore
        if self.mpc is None:
            raise ValueError("MPC policy must be provided in the environment config.")

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

        base_vel_cmd = self._cmd + 0.5

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
        with torch.no_grad():
            mpc_obs = self._build_low_level_obs()
            mpc_action = self.mpc(mpc_obs) + 0.1
            self._last_joint_action = mpc_action

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
        }
    )


@configclass
class Go2EnvCfg(ManagerBasedRLEnvCfg):

    scene = Go2SimCfg(num_envs=1, env_spacing=12.0)

    actions = ActionsCfg()
    observations = ObservationsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventsCfg()

    def __post_init__(self) -> None:
        self.sim.dt = 0.005
        self.episode_length_s = 20.0

        logger.info("Go2 EnvCfg initialized")

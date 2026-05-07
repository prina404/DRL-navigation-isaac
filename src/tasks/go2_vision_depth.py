from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from loguru import logger

import mdp.actions.actions_config as actions
import mdp.curriculum.curriculum_config as curriculum
import mdp.events.events_config as events
import mdp.observations.obs_config as obs
import mdp.rewards.rewards_config as rewards
import mdp.scene.go2_scene_cfg as scene
import mdp.terminations.terminations_config as terminations
from tasks.task_common_params import PARAMS


@configclass
class Go2VisionFull(ManagerBasedRLEnvCfg):

    actions = actions.ActionsCfg()
    observations = obs.VisionOnlyCfg()
    rewards = rewards.RewardsCfg()
    terminations = terminations.TerminationsCfg()
    events = events.FullEventsCfg()
    curriculum = curriculum.CurriculumCfg()

    obs_groups = {
        "policy": ["velocity_buffer", "action_buffer", "global_plan", "vision"],
        "critic": ["velocity_buffer", "action_buffer", "global_plan", "vision"],
    }

    def __post_init__(self) -> None:
        # initialized in post init to ensure dataset cfg is loaded and we get the correct scene path
        self.scene = scene.Go2FullVisionCfg(num_envs=1, env_spacing=18.0)

        self.sim.dt = PARAMS["dt"]
        self.sim.device = PARAMS["device"]
        self.sim.use_fabric = PARAMS["use_fabric"]
        self.sim.render = PARAMS["render"]

        self.episode_length_s = PARAMS["episode_length_s"]
        self.decimation = PARAMS["decimation"]
        self.sim.render_interval = self.decimation

        logger.info("Go2 Full environment with Lidar+RGB initialized")


@configclass
class Go2VisionEmpty(ManagerBasedRLEnvCfg):

    scene = scene.Go2EmptyVisionCfg(num_envs=1, env_spacing=18.0)

    actions = actions.ActionsCfg()
    observations = obs.VisionOnlyCfg()
    rewards = rewards.RewardsCfg()
    terminations = terminations.TerminationsCfg()
    events = events.BaseEventsCfg()
    curriculum = curriculum.CurriculumCfg()

    obs_groups = {
        "policy": ["velocity_buffer", "action_buffer", "global_plan", "vision"],
        "critic": ["velocity_buffer", "action_buffer", "global_plan", "vision"],
    }

    def __post_init__(self) -> None:
        self.sim.dt = PARAMS["dt"]
        self.sim.device = PARAMS["device"]
        self.sim.use_fabric = PARAMS["use_fabric"]
        self.sim.render = PARAMS["render"]

        self.episode_length_s = PARAMS["episode_length_s"]
        self.decimation = PARAMS["decimation"]
        self.sim.render_interval = self.decimation

        logger.info("Go2 Empty environment with Lidar+RGB initialized")


@configclass
class Go2DepthFull(ManagerBasedRLEnvCfg):

    actions = actions.ActionsCfg()
    observations = obs.DepthOnlyCfg()
    rewards = rewards.RewardsCfg()
    terminations = terminations.TerminationsCfg()
    events = events.FullEventsCfg()
    curriculum = curriculum.CurriculumCfg()

    obs_groups = {
        "policy": ["velocity_buffer", "action_buffer", "global_plan", "depth"],
        "critic": ["velocity_buffer", "action_buffer", "global_plan", "depth"],
    }

    def __post_init__(self) -> None:
        # initialized in post init to ensure dataset cfg is loaded and we get the correct scene path
        self.scene = scene.Go2EmptyVisionCfg(num_envs=1, env_spacing=18.0)

        self.sim.dt = PARAMS["dt"]
        self.sim.device = PARAMS["device"]
        self.sim.use_fabric = PARAMS["use_fabric"]
        self.sim.render = PARAMS["render"]

        self.episode_length_s = PARAMS["episode_length_s"]
        self.decimation = PARAMS["decimation"]
        self.sim.render_interval = self.decimation

        logger.info("Go2 Full environment with Lidar+RGB initialized")

@configclass
class Go2VisionDepthFull(ManagerBasedRLEnvCfg):

    actions = actions.ActionsCfg()
    observations = obs.VisionDepthCfg()
    rewards = rewards.RewardsCfg()
    terminations = terminations.TerminationsCfg()
    events = events.FullEventsCfg()
    curriculum = curriculum.CurriculumCfg()

    obs_groups = {
        "policy": ["velocity_buffer", "action_buffer", "global_plan", "vision", "depth"],
        "critic": ["velocity_buffer", "action_buffer", "global_plan", "vision", "depth"],
    }

    def __post_init__(self) -> None:
        # initialized in post init to ensure dataset cfg is loaded and we get the correct scene path
        self.scene = scene.Go2FullVisionCfg(num_envs=1, env_spacing=18.0)

        self.sim.dt = PARAMS["dt"]
        self.sim.device = PARAMS["device"]
        self.sim.use_fabric = PARAMS["use_fabric"]
        self.sim.render = PARAMS["render"]

        self.episode_length_s = PARAMS["episode_length_s"]
        self.decimation = PARAMS["decimation"]
        self.sim.render_interval = self.decimation

        logger.info("Go2 Full environment with Lidar+RGB initialized")
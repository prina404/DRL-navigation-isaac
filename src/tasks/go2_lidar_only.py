from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg
import mdp.scene.go2_scene_cfg as scene
import mdp.actions.actions_config as actions
import mdp.observations.obs_config as obs
import mdp.rewards.rewards_config as rewards
import mdp.terminations.terminations_config as terminations
import mdp.events.events_config as events
import mdp.curriculum.curriculum_config as curriculum
import isaaclab.sim as sim_utils
from tasks.task_common_params import PARAMS
from loguru import logger


@configclass
class Go2LidarFull(ManagerBasedRLEnvCfg):

    scene = scene.Go2FullSceneLidarCfg(num_envs=1, env_spacing=18.0)

    actions = actions.ActionsCfg()
    observations = obs.LidarOnlyCfg()
    rewards = rewards.RewardsCfg()
    terminations = terminations.TerminationsCfg()
    events = events.FullEventsCfg()
    curriculum = curriculum.CurriculumCfg()

    obs_groups = {
        "policy": ["velocity_buffer", "action_buffer", "global_plan", "lidar"],
        "critic": ["velocity_buffer", "action_buffer", "global_plan", "lidar"],
    }

    def __post_init__(self) -> None:
        self.sim.dt = PARAMS["dt"]
        self.sim.device = PARAMS["device"]
        self.sim.use_fabric = PARAMS["use_fabric"]
        self.sim.render = PARAMS["render"]

        self.episode_length_s = PARAMS["episode_length_s"]
        self.decimation = PARAMS["decimation"]
        self.sim.render_interval = self.decimation

        logger.info("Go2 Full environment with Lidar only initialized")


@configclass
class Go2LidarEmpty(ManagerBasedRLEnvCfg):

    scene = scene.Go2EmptySceneCfg(num_envs=1, env_spacing=18.0)

    actions = actions.ActionsCfg()
    observations = obs.LidarOnlyCfg()
    rewards = rewards.RewardsCfg()
    terminations = terminations.TerminationsCfg()
    events = events.BaseEventsCfg()
    curriculum = curriculum.CurriculumCfg()

    obs_groups = {
        "policy": ["velocity_buffer", "action_buffer", "global_plan", "lidar"],
        "critic": ["velocity_buffer", "action_buffer", "global_plan", "lidar"],
    }
    def __post_init__(self) -> None:
        self.sim.dt = PARAMS["dt"]
        self.sim.device = PARAMS["device"]
        self.sim.use_fabric = PARAMS["use_fabric"]
        self.sim.render = PARAMS["render"]

        self.episode_length_s = PARAMS["episode_length_s"]
        self.decimation = PARAMS["decimation"]
        self.sim.render_interval = self.decimation


        logger.info("Go2 Empty environment with Lidar only initialized")
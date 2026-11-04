from collections.abc import Sequence
from pathlib import Path

import torch
from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs.ui import ViewportCameraController

from navigation_env.managers.camera_manager import CameraManager
from navigation_env.managers.map_manager import MapManager
from navigation_env.managers.path_manager import PathManager


class NavEnv(ManagerBasedRLEnv):
    # Reward terms logged as their mean value per step, before the weight, instead of IsaacLab's episode sum over
    # max_episode_length_s, which shrinks with the episode even when every step earns the full reward.
    PER_STEP_LOGGED_REWARDS = ("heading",)

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        assert "scene_path" in kwargs, "Must provide scene_path as a keyword argument"
        scene_path = Path(kwargs["scene_path"])
        map_img_path = str(scene_path.parent / (scene_path.stem + "_map.png"))
        map_yaml_path = str(scene_path.parent / (scene_path.stem + "_map.yaml"))
        self._map_manager = MapManager(map_img_path, map_yaml_path, self.num_envs, device=self.device)

        self.path_manager = PathManager(
            scene_path=scene_path, map_mgr=self._map_manager, num_envs=self.num_envs, device=self.device
        )

        camera_controller = ViewportCameraController(self, cfg=ViewerCfg(origin_type="world"))
        self.camera_manager = CameraManager(
            camera_controller,
            camera_relative_pos=torch.tensor([-0.6, 0.0, 0.8]),
            camera_lookat=torch.tensor([1.6, 0.0, -1.0]),
        )

        self.long_horizon = kwargs.get("use_long_horizon", False)
        self.sample_voronoi_probability = kwargs.get("sample_voronoi_probability", 0.25)
        self.debug_vis = kwargs.get("debug_vis", False)
        self.collision_termination_thresh = 1.0
        self.nominal_weight = 0.0  # scale for nominal action

        self.episode_counter = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        if "log" in self.extras:  # Fixes wrong logging of rewards and dones in the first step after reset
            self.extras.pop("log", None)

        retVal = super().step(action)

        # the action and velocity histories are observation terms (see obs_config.py), cleared on reset
        self.update_follow_camera()
        self._update_lidar_buffer()

        return retVal

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        # read before the reward manager clears the episode sums and the episode lengths are zeroed
        rewards = self.reward_manager
        per_step = {
            name: self._mean_reward_per_step(name, env_ids)
            for name in self.PER_STEP_LOGGED_REWARDS
            if name in rewards.active_terms and rewards.get_term_cfg(name).weight != 0.0
        }
        super()._reset_idx(env_ids)
        for name, value in per_step.items():
            self.extras["log"][f"Episode_Reward/{name}"] = value

    def _mean_reward_per_step(self, name: str, env_ids: Sequence[int]) -> torch.Tensor:
        """Value of reward term `name` per step, before its weight, averaged over the episodes ending in `env_ids`."""
        rewards = self.reward_manager
        scale = rewards.get_term_cfg(name).weight * self.step_dt  # what the reward manager multiplies each step by
        steps = self.episode_length_buf[env_ids].clamp(min=1)
        return (rewards._episode_sums[name][env_ids] / (scale * steps)).mean()

    def teleport_robots(self, env_ids: torch.Tensor) -> None:
        ## set cartesian position, quaternion orientation in (x, y, z, w), and linear and angular velocity
        ## reset positions only for specified env indices
        robot: Articulation = self.scene["robot"]

        state = self.path_manager.compute_robot_teleport_state(
            env_ids, self.scene.env_origins, robot.data.default_root_state.torch[0, 2]
        )

        robot.write_root_com_state_to_sim(state, env_ids)

        # reset joints to default state after teleporting root poses
        default_pos = robot.data.default_joint_pos.torch[env_ids]
        default_vel = robot.data.default_joint_vel.torch[env_ids]
        robot.set_joint_position_target(default_pos, None, env_ids)
        robot.write_joint_state_to_sim(default_pos, default_vel, None, env_ids)

        self._map_manager.reset_costmaps(env_ids)  # clear costmaps on teleport to avoid stale obstacle data

        if env_ids[0] == 0:  # only update camera on the first env for now
            self.update_follow_camera(smoothing=1.0)  # snap camera to new position on reset

    def sample_task_on_reset(self, env_ids: torch.Tensor) -> None:
        self.path_manager.sample_nav_task(env_ids, sample_voronoi_prob=self.sample_voronoi_probability)

    def manual_replan(self, env_ids: torch.Tensor) -> None:
        robot = self.scene["robot"]
        self.path_manager.compute_global_plan(
            env_ids, robot.data.root_com_pos_w.torch[env_ids], self.scene.env_origins[env_ids]
        )

    def _update_lidar_buffer(self) -> None:
        if "lidar" in self.scene.sensors:
            lidar = self.scene.sensors["lidar"]
            self._map_manager.update_local_costmap(lidar.data, self.scene.env_origins)

    def update_follow_camera(self, smoothing: float = 0.3):
        if not self.debug_vis and not self.sim.has_gui and self.render_mode != "rgb_array":
            return

        robot_pos = self.scene["robot"].data.root_pos_w.torch[0]
        robot_quat = self.scene["robot"].data.root_quat_w.torch[0]

        self.camera_manager.update(robot_pos, robot_quat, smoothing_factor=smoothing)

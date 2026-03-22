import isaaclab.sim as sim_utils
import torch
import warp as wp
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_angle_axis

from lab.helpers.observation_helpers import get_lidar
from lab.MyEnv import MyEnv


class MyEnvDebuggingVis(MyEnv):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.viz_marker = self._define_markers()

    def step(self, action):
        retVal = super().step(action)
        self._visualize_markers()
        return retVal

    def _visualize_markers(self):
        marker_data = [
            self._build_heading_markers(),
            self._build_goal_markers(),
            self._build_path_markers(),
            # self._build_lidar_markers(),
        ]

        loc = torch.vstack([data[0] for data in marker_data if data[0].shape[0] > 0])
        rots = torch.vstack([data[1] for data in marker_data if data[1].shape[0] > 0])
        indices = torch.hstack([data[2] for data in marker_data if data[2].shape[0] > 0])

        self.viz_marker.visualize(loc, rots, marker_indices=indices)

    def _empty_marker_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.empty((0, 3), device=self.device),
            torch.empty((0, 4), device=self.device),
            torch.empty((0,), device=self.device, dtype=torch.int64),
        )

    def _build_heading_markers(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        robot = self.scene["robot"]

        # robot position in local (env) frame for path math
        robot_pos_local = wp.to_torch(robot.data.root_link_pos_w)[:, :2] - self.scene.env_origins[:, :2]  # (B, 2)

        # reward heading: 3rd point forward along path (closer to goal than robot)
        reward_yaw = torch.zeros((self.num_envs,), device=self.device)
        null_target = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        nominal_disabled = self.nominal_weight == 0.0
        for i in range(self.num_envs):
            path_points = self.path_manager.path_tensors[i]

            if path_points is None or path_points.size(0) <= 1 or nominal_disabled:
                null_target[i] = True
            else:
                # Find closest point on trajectory
                distances = torch.norm(path_points - robot_pos_local[i], dim=-1)
                closest_idx = torch.argmin(distances)

                # Pick third point forward on trajectory with boundary check
                target_idx = min(closest_idx + 3, path_points.shape[0] - 1)
                target = path_points[target_idx]

                delta = target - robot_pos_local[i]
                reward_yaw[i] = torch.atan2(delta[1], delta[0])

        self.marker_locations = wp.to_torch(robot.data.root_pos_w)  # (B, 3)
        self.reward_marker_orientations = quat_from_angle_axis(reward_yaw, self.up_dir)

        # command heading: from action (x, y) only
        cmd_xy = self._action_buffer[:, 0, :2]  # (B, 2)
        cmd_yaw = torch.atan2(cmd_xy[:, 1], cmd_xy[:, 0])

        # rotate command to world frame using robot yaw
        _, _, robot_yaw = euler_xyz_from_quat(wp.to_torch(robot.data.root_link_quat_w))  # (B, 1)
        command_world_yaw = robot_yaw.squeeze(-1) + cmd_yaw
        self.command_marker_orientations = quat_from_angle_axis(command_world_yaw, self.up_dir)

        # offset markers above robot and render both prototypes
        target_loc = self.marker_locations + self.marker_offset  # (B, 3)
        cmd_loc = target_loc + torch.tensor([0.0, 0.0, 0.1], device=self.device)  # (B, 3)

        target_loc[null_target] += torch.tensor([0.0, 0.0, -1.0], device=self.device)  # if no target, hide the arrow

        loc = torch.vstack((target_loc, cmd_loc))  # (2B, 3)
        rots = torch.vstack((self.reward_marker_orientations, self.command_marker_orientations))  # (2B, 4)

        all_envs = torch.arange(self.num_envs, device=self.device)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))  # (2B,)
        return loc, rots, indices

    def _build_goal_markers(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        goal_locs = torch.cat(
            (
                self.path_manager.goal_pos_local + self.scene.env_origins[:, :2],
                torch.zeros((self.num_envs, 1), device=self.device),
            ),
            dim=-1,
        )
        goal_rots = torch.zeros((self.num_envs, 4), device=self.device)
        goal_rots[:, 0] = 1.0  # identity quaternion
        goal_indices = torch.full((self.num_envs,), 3, device=self.device, dtype=torch.int64)
        return goal_locs, goal_rots, goal_indices

    def _build_path_markers(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # --- path points (one every 30cm) ---
        step = 0.3
        path_loc_list = []
        for i in range(self.num_envs):
            path_points = self.path_manager.path_tensors[i]
            if path_points is None or path_points.numel() == 0:
                continue

            kept = [path_points[0]]
            last = path_points[0]
            for p in path_points[1:]:
                if torch.norm(p - last) >= step:
                    kept.append(p)
                    last = p

            kept = torch.stack(kept, dim=0)  # (K, 2)
            # convert to world coords
            kept_w = kept + self.scene.env_origins[i, :2]
            z = torch.full((kept_w.shape[0], 1), 0.05, device=self.device)
            kept_w = torch.cat([kept_w, z], dim=-1)  # (K, 3)
            path_loc_list.append(kept_w)

        if len(path_loc_list) == 0:
            return self._empty_marker_data()

        path_locs = torch.cat(path_loc_list, dim=0)
        path_rots = torch.zeros((path_locs.shape[0], 4), device=self.device)
        path_rots[:, 0] = 1.0  # identity quaternion
        path_indices = torch.full((path_locs.shape[0],), 2, device=self.device, dtype=torch.int64)
        return path_locs, path_rots, path_indices

    def _build_lidar_markers(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        robot = self.scene["robot"]

        # render lidar points from last observation
        lidar_emb = get_lidar(self, 128, normalize=False)
        lidar_dist = lidar_emb[:, :128]
        lidar_angles = lidar_emb[:, 128:]

        # reconstruct lidar points in robot frame
        lidar_x_local = lidar_dist * torch.cos(lidar_angles)
        lidar_y_local = -lidar_dist * torch.sin(lidar_angles)

        # rotate to world frame using robot yaw
        _, _, robot_yaw = euler_xyz_from_quat(wp.to_torch(robot.data.root_link_quat_w))  # (B,)
        robot_yaw = robot_yaw.squeeze(-1) if robot_yaw.ndim > 1 else robot_yaw

        cos_yaw = torch.cos(robot_yaw).unsqueeze(1)  # (B, 1)
        sin_yaw = torch.sin(robot_yaw).unsqueeze(1)  # (B, 1)

        lidar_x_world = wp.to_torch(robot.data.root_link_pos_w)[:, 0:1] + lidar_x_local * cos_yaw - lidar_y_local * sin_yaw
        lidar_y_world = wp.to_torch(robot.data.root_link_pos_w)[:, 1:2] + lidar_x_local * sin_yaw + lidar_y_local * cos_yaw
        lidar_z_world = torch.full_like(lidar_x_world, 0.10)

        lidar_points = torch.stack((lidar_x_world, lidar_y_world, lidar_z_world), dim=-1)  # (B, 128, 3)

        # hide padded / invalid lidar returns
        max_point_dist = torch.norm(torch.tensor((4.0, 4.0, 1.0), device=self.device))
        valid_mask = lidar_dist < (max_point_dist - 1e-3)

        if not valid_mask.any():
            return self._empty_marker_data()

        lidar_points_flat = lidar_points[valid_mask]  # (N, 3)
        lidar_rots = torch.zeros((lidar_points_flat.shape[0], 4), device=self.device)
        lidar_rots[:, 0] = 1.0
        lidar_indices = torch.full(
            (lidar_points_flat.shape[0],),
            4,
            device=self.device,
            dtype=torch.int64,
        )
        return lidar_points_flat, lidar_rots, lidar_indices

    def _define_markers(self) -> VisualizationMarkers:
        heading_marker = VisualizationMarkersCfg(
            prim_path="/Visuals/myMarkers",
            markers={
                "reward_heading": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.075, 0.075, 0.25),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.3)),
                ),
                "command": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.075, 0.075, 0.25),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                "path_point": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Shapes/sphere.usd",
                    scale=(0.05, 0.05, 0.05),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
                "goal": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Shapes/sphere.usd",
                    scale=(0.15, 0.15, 0.15),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                # "lidar": sim_utils.UsdFileCfg(
                #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Shapes/sphere.usd",
                #     scale=(0.03, 0.03, 0.03),
                #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                # ),
            },
        )
        # setting aside useful variables for later
        self.up_dir = torch.tensor([0.0, 0.0, 1.0]).cuda()
        self.yaws = torch.zeros((self.num_envs, 1)).cuda()
        self.commands = torch.randn((self.num_envs, 3)).cuda()
        self.commands[:, -1] = 0.0
        self.commands = self.commands / torch.linalg.norm(self.commands, dim=1, keepdim=True)

        # offsets to account for atan range and keep things on [-pi, pi]
        ratio = self.commands[:, 1] / (self.commands[:, 0] + 1e-8)
        gzero = torch.where(self.commands > 0, True, False)
        lzero = torch.where(self.commands < 0, True, False)
        plus = lzero[:, 0] * gzero[:, 1]
        minus = lzero[:, 0] * lzero[:, 1]
        offsets = torch.pi * plus - torch.pi * minus
        self.yaws = torch.atan(ratio).reshape(-1, 1) + offsets.reshape(-1, 1)

        self.marker_locations = torch.zeros((self.num_envs, 3)).cuda()
        self.marker_offset = torch.zeros((self.num_envs, 3)).cuda()
        self.marker_offset[:, -1] = 0.2
        self.reward_marker_orientations = torch.zeros((self.num_envs, 4)).cuda()
        self.command_marker_orientations = torch.zeros((self.num_envs, 4)).cuda()
        return VisualizationMarkers(heading_marker)

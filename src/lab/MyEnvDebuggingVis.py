from isaaclab.utils.math import quat_from_angle_axis, euler_xyz_from_quat
from lab.MyEnv import MyEnv
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import torch

class MyEnvDebuggingVis(MyEnv):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.viz_marker = self._define_markers()

    def step(self, action):
        retVal = super().step(action)
        self._visualize_markers()
        return retVal
    
    def _visualize_markers(self):
        robot = self.scene["unitree_go2"]
        self.marker_locations = robot.data.root_pos_w  # (B, 3)

        # robot position in local (env) frame for path math
        robot_pos_local = robot.data.root_link_pos_w[:, :2] - self.scene.env_origins[:, :2]  # (B, 2)

        # reward heading: 3rd point forward along path (closer to goal than robot)
        reward_yaw = torch.zeros((self.num_envs,), device=self.device)
        for i in range(self.num_envs):
            goal_pos = self.goal_pos[i]
            path_points = self._path_tensors[i]

            if path_points is None or path_points.numel() == 0:
                target = goal_pos
            else:
                dist_to_goal = torch.norm(robot_pos_local[i] - goal_pos)
                forward_points = path_points[torch.norm(path_points - goal_pos, dim=-1) <= dist_to_goal]
                if forward_points.shape[0] == 0:
                    forward_points = path_points
                idx = min(2, forward_points.shape[0] - 1)
                target = forward_points[idx]

            delta = target - robot_pos_local[i]
            reward_yaw[i] = torch.atan2(delta[1], delta[0])

        self.reward_marker_orientations = quat_from_angle_axis(reward_yaw, self.up_dir)

        # command heading: from action (x, y) only
        cmd_xy = self._action_buffer[:, 0, :2]  # (B, 2)
        cmd_yaw = torch.atan2(cmd_xy[:, 1], cmd_xy[:, 0])

        # rotate command to world frame using robot yaw
        _, _, robot_yaw = euler_xyz_from_quat(robot.data.root_link_quat_w)  # (B, 1)
        command_world_yaw = robot_yaw.squeeze(-1) + cmd_yaw
        self.command_marker_orientations = quat_from_angle_axis(command_world_yaw, self.up_dir)

        # offset markers above robot and render both prototypes
        loc = self.marker_locations + self.marker_offset  # (B, 3)
        loc = torch.vstack((loc, loc + torch.tensor([0.0, 0.0, 0.15], device=self.device)))  # (2B, 3)
        rots = torch.vstack((self.reward_marker_orientations, self.command_marker_orientations))  # (2B, 4)

        all_envs = torch.arange(self.cfg.scene.num_envs, device=self.device)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))  # (2B,)
        # goal marker
        goal_locs = torch.cat((self.goal_pos + self.scene.env_origins[:, :2], torch.zeros((self.cfg.scene.num_envs, 1), device=self.device)), dim=-1)
        goal_rots = torch.zeros((self.cfg.scene.num_envs, 4), device=self.device)
        goal_rots[:, 0] = 1.0  # identity quaternion
        goal_indices = torch.full((self.cfg.scene.num_envs,), 3, device=self.device, dtype=torch.int64)

        # --- path points (one every 30cm) ---
        step = 0.3
        path_loc_list = []
        for i in range(self.num_envs):
            path_points = self._path_tensors[i]
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

        if len(path_loc_list) > 0:
            path_locs = torch.cat(path_loc_list, dim=0)
            path_rots = torch.zeros((path_locs.shape[0], 4), device=self.device)
            path_rots[:, 0] = 1.0  # identity quaternion
            path_indices = torch.full((path_locs.shape[0],), 2, device=self.device, dtype=torch.int64)

            loc = torch.vstack((loc, path_locs, goal_locs))
            rots = torch.vstack((rots, path_rots, goal_rots))
            indices = torch.hstack((indices, path_indices, goal_indices))

        self.viz_marker.visualize(loc, rots, marker_indices=indices)


    def _define_markers(self) -> VisualizationMarkers:
        heading_marker = VisualizationMarkersCfg(
            prim_path="/Visuals/myMarkers",
            markers={
                "reward_heading": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.1, 0.1, 0.35),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.3)),
                ),
                "command": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.1, 0.1, 0.35),
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
            }
        )
        # setting aside useful variables for later
        self.up_dir = torch.tensor([0.0, 0.0, 1.0]).cuda()
        self.yaws = torch.zeros((self.cfg.scene.num_envs, 1)).cuda()
        self.commands = torch.randn((self.cfg.scene.num_envs, 3)).cuda()
        self.commands[:,-1] = 0.0
        self.commands = self.commands/torch.linalg.norm(self.commands, dim=1, keepdim=True)

        # offsets to account for atan range and keep things on [-pi, pi]
        ratio = self.commands[:,1]/(self.commands[:,0]+1E-8)
        gzero = torch.where(self.commands > 0, True, False)
        lzero = torch.where(self.commands < 0, True, False)
        plus = lzero[:,0]*gzero[:,1]
        minus = lzero[:,0]*lzero[:,1]
        offsets = torch.pi*plus - torch.pi*minus
        self.yaws = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

        self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset[:,-1] = 0.5
        self.reward_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()
        self.command_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()
        return VisualizationMarkers(heading_marker)


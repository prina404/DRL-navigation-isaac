import torch
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.utils.math import quat_apply


class CameraManager:
    def __init__(self, controller: ViewportCameraController, camera_relative_pos: torch.Tensor, camera_lookat: torch.Tensor):
        self.camera = controller
        self.old_eye_pos = None
        self.old_lookat_pos = None
        # Camera and lookat settings for video recording
        self.camera_offset = camera_relative_pos.to(controller._env.device)
        self.lookat_offset = camera_lookat.to(controller._env.device)

    def update(self, robot_pos_w: torch.Tensor, robot_rot_w: torch.Tensor, smoothing_factor: float = 0.3):
        camera_offset_w = quat_apply(robot_rot_w, self.camera_offset)
        lookat_offset_w = quat_apply(robot_rot_w, self.lookat_offset)

        if self.old_eye_pos is None:
            self.old_eye_pos = robot_pos_w + camera_offset_w
            self.old_lookat_pos = robot_pos_w + lookat_offset_w
        else:  # smooth camera movement by interpolating between old and new positions
            new_eye_pos = robot_pos_w + camera_offset_w
            new_lookat_pos = robot_pos_w + lookat_offset_w

            alpha = smoothing_factor
            smoothed_eye_pos = (1 - alpha) * self.old_eye_pos + alpha * new_eye_pos
            smoothed_lookat_pos = (1 - alpha) * self.old_lookat_pos + alpha * new_lookat_pos

            self.camera.update_view_location(
                eye=smoothed_eye_pos.cpu().numpy(),
                lookat=smoothed_lookat_pos.cpu().numpy(),
            )

            self.old_eye_pos = smoothed_eye_pos
            self.old_lookat_pos = smoothed_lookat_pos

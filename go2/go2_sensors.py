from turtle import width
import omni
import numpy as np
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from pxr import Gf
import omni.replicator.core as rep
import torch


class SensorManager:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.tiled_camera = None
        self._camera_resolution = None

    def add_rtx_lidar(self):
        lidar_annotators = []
        for env_idx in range(self.num_envs):
            _, sensor = omni.kit.commands.execute(
                "IsaacSensorCreateRtxLidar",
                path="/lidar",
                parent=f"/World/envs/env_{env_idx}/Go2/base",
                config="Hesai_XT32_SD10",
                # config="Velodyne_VLS128",
                translation=(0.2, 0, 0.2),
                orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),  # Gf.Quatd is w,i,j,k
            )

            annotator = rep.AnnotatorRegistry.get_annotator(
                "RtxSensorCpuIsaacCreateRTXLidarScanBuffer"
            )
            hydra_texture = rep.create.render_product(sensor.GetPath(), [1, 1], name="Isaac")
            annotator.attach(hydra_texture.path)
            lidar_annotators.append(annotator)
        return lidar_annotators
    
    # # TODO: set realistic aperture and focal length based on Go2 camera specs
    # def add_camera(self, freq, resolution=(640, 480)):
    #     """Create a single TiledCamera that captures all env cameras in one batch."""
    #     self._camera_resolution = resolution
    #     tiled_camera: TiledCameraCfg = TiledCameraCfg(
    #         prim_path="/World/envs/env_.*/Go2/base/front_cam",
    #         width=resolution[0],
    #         height=resolution[1],
    #         offset=TiledCameraCfg.OffsetCfg(  
    #             pos=(0.4, 0.0, 0.0),  # Your desired offset  
    #             convention="world"       # Coordinate frame convention  
    #         ),  
    #         data_types=["rgb"],
    #         spawn=PinholeCameraCfg(
    #             focal_length=1.5,
    #             horizontal_aperture=4,
    #             clipping_range=(0.1, 1000.0)
    #         )
    #     )
    #     self.tiled_camera = TiledCamera(tiled_camera)
    #     return self.tiled_camera

    # def get_camera_frames(self) -> torch.Tensor:
    #     """Return a tensor of shape (num_envs, H, W, C) containing all camera frames."""
    #     if self.tiled_camera is None:
    #         raise RuntimeError("Tiled camera not initialized. Call add_camera() first.")

    #     return self.tiled_camera.data.output["rgb"]

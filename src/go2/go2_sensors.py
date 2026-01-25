import omni
from pxr import Gf
import omni.replicator.core as rep


class SensorManager:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.tiled_camera = None
        self._camera_resolution = None
        self.lidar_sensors = []
        self.lidar_render_products = []
        self.lidar_annotators = []

    def add_rtx_lidar(self):
        self.lidar_sensors.clear()
        self.lidar_render_products.clear()
        self.lidar_annotators.clear()

        for env_idx in range(self.num_envs):
            # One LiDAR prim per env (unique path, attached under that env's robot base)
            sensor_attributes = {'omni:sensor:Core:scanRateBaseHz': 60}
            _, sensor = omni.kit.commands.execute(
                "IsaacSensorCreateRtxLidar",
                path=f"lidar_{env_idx}",  # relative path under parent -> unique per env
                parent=f"/World/envs/env_{env_idx}/Go2/base",
                config="Hesai_XT32_SD10",
                # config="Velodyne_VLS128",
                translation=(0.2, 0, 0.2),
                orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),  # w,i,j,k
                visibility=True,
                **sensor_attributes
            )
            self.lidar_sensors.append(sensor)

            annotator = rep.AnnotatorRegistry.get_annotator("IsaacCreateRTXLidarScanBuffer")
            rp = rep.create.render_product(sensor.GetPath(), [1, 1], name=f"RtxLidarRP_{env_idx}")

            annotator.attach(rp.path)

            self.lidar_annotators.append(annotator)
            self.lidar_render_products.append(rp)

        # Debug-draw all env LiDARs
            dd_writer = rep.writers.get("RtxLidarDebugDrawPointCloudBuffer")
            dd_writer.attach(rp)

        return self.lidar_sensors
    

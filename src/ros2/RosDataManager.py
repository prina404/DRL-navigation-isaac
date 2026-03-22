# Adapted from original code by Zhefan-Xu
# https://github.com/Zhefan-Xu/isaac-go2-ros2


import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import OccupancyGrid
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import omni
import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata._syntheticdata as sd
import subprocess
import time
import torch
from torch import Tensor
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.sensors import MultiMeshRayCaster, TiledCamera
import warp as wp
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from lab.MyEnv import MyEnv

MAP_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,  # ← critical
    reliability=ReliabilityPolicy.RELIABLE
)


class RosDataManager(Node):
    def __init__(self, env: RslRlVecEnvWrapper, lidar_annotators: MultiMeshRayCaster, cameras: TiledCamera, cfg):
        super().__init__("robot_data_manager")
        self.cfg = cfg
        self.create_ros_time_graph()
        sim_time_set = False
        while rclpy.ok() and sim_time_set == False:
            sim_time_set = self.use_sim_time()

        self.env: MyEnv = env.unwrapped
        self.num_envs = self.env.scene.num_envs
        self.lidar = lidar_annotators
        self.cameras = cameras

        # ROS2 Broadcaster
        self.broadcaster = TransformBroadcaster(self)
        self.static_broadcaster = StaticTransformBroadcaster(self)

        # ROS2 Publisher
        self.odom_pub = []
        self.pose_pub = []
        self.lidar_pub = []
        self.color_pub = []
        self.depth_pub = []
        self.camera_info_pub = []
        
        self.map_pub = []
        for i in range(self.num_envs):
            pub = self.create_publisher(
                    OccupancyGrid,
                    f"{self.robot_ns(i)}/map",
                    MAP_QOS
                )
            msg = self.create_map_msg(i)
            msg.header.stamp = self.get_clock().now().to_msg()
            self.map_pub.append(pub)
            pub.publish(msg)

        # ROS2 Subscriber
        self.cmd_vel_sub = []

        for i in range(self.num_envs):
            self.odom_pub.append(self.create_publisher(Odometry, self.odom_topic(i), 10))
            self.pose_pub.append(self.create_publisher(PoseStamped, self.pose_topic(i), 10))
            self.lidar_pub.append(self.create_publisher(PointCloud2, self.lidar_topic(i), 10))

            if self.cfg.sensor.enable_camera:
                self.color_pub.append(self.create_publisher(Image, self.color_topic(i), 10))
                self.depth_pub.append(self.create_publisher(Image, self.depth_topic(i), 10))
                self.camera_info_pub.append(self.create_publisher(CameraInfo, self.camera_info_topic(i), 10))

            self.cmd_vel_sub.append(
                self.create_subscription(
                    Twist,
                    self.cmd_vel_topic(i),
                    lambda msg, env_idx=i: self.cmd_vel_callback(msg, env_idx),
                    10,
                )
            )

        # use wall time for lidar and odom pub
        self.odom_pose_freq = 50.0
        self.lidar_freq = 15.0
        self.camera_freq = float(getattr(self.cfg.sensor, "camera_freq", self.lidar_freq))
        self.odom_pose_pub_time = time.time()
        self.lidar_pub_time = time.time()
        self.camera_pub_time = time.time()

        self.create_static_transform()
        self.create_camera_publisher()

        self.base_vel_cmd_input = torch.zeros((self.num_envs, 3), dtype=torch.float32).cpu()

    def robot_ns(self, env_idx: int) -> str:
        return "robot" if self.num_envs == 1 else f"robot_{env_idx}"

    def map_frame(self, env_idx: int) -> str:
        return "map" if self.num_envs == 1 else f"{self.robot_ns(env_idx)}/map"

    def odom_frame(self, env_idx: int) -> str:
        return "odom" if self.num_envs == 1 else f"{self.robot_ns(env_idx)}/odom"

    def base_frame(self, env_idx: int) -> str:
        return "base_link" if self.num_envs == 1 else f"{self.robot_ns(env_idx)}/base_link"

    def lidar_frame(self, env_idx: int) -> str:
        return "lidar_frame" if self.num_envs == 1 else f"{self.robot_ns(env_idx)}/lidar_frame"

    def camera_frame(self, env_idx: int) -> str:
        return "front_cam" if self.num_envs == 1 else f"{self.robot_ns(env_idx)}/front_cam"

    def odom_topic(self, env_idx: int) -> str:
        return f"{self.robot_ns(env_idx)}/odom"

    def pose_topic(self, env_idx: int) -> str:
        return f"{self.robot_ns(env_idx)}/pose"

    def lidar_topic(self, env_idx: int) -> str:
        return f"{self.robot_ns(env_idx)}/lidar"

    def color_topic(self, env_idx: int) -> str:
        return f"{self.robot_ns(env_idx)}/front_cam/color_image"

    def depth_topic(self, env_idx: int) -> str:
        return f"{self.robot_ns(env_idx)}/front_cam/depth_image"

    def camera_info_topic(self, env_idx: int) -> str:
        return f"{self.robot_ns(env_idx)}/front_cam/camera_info"

    def cmd_vel_topic(self, env_idx: int) -> str:
        return f"{self.robot_ns(env_idx)}/cmd_vel"
    
    def env_origin(self, env_idx: int) -> Tensor:
        return self.env.scene.env_origins[env_idx]

    def create_ros_time_graph(self):
        og.Controller.edit(
            {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                    # ("OnPlayBack", "omni.graph.action.OnImpulseEvent"),
                ],
                og.Controller.Keys.CONNECT: [
                    # Connecting execution of OnImpulseEvent node to PublishClock so it will only publish when an impulse event is triggered
                    # ("OnPlayBack.outputs:tick", "PublishClock.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                    # Connecting simulationTime data of ReadSimTime to the clock publisher node
                    ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    # Assigning topic name to clock publisher
                    ("PublishClock.inputs:topicName", "/clock"),
                ],
            },
        )

    def use_sim_time(self):
        # Define the command as a list
        command = ["ros2", "param", "set", "/robot_data_manager", "use_sim_time", "true"]

        # Run the command in a non-blocking way
        subprocess.Popen(command)
        return True

    def create_map_msg(self, env_idx: int):
        msg = OccupancyGrid()

        msg.header.frame_id = self.map_frame(env_idx)
        map_mgr = self.env._map_manager

        # map metadata
        msg.info.resolution = map_mgr.resolution
        msg.info.width = map_mgr.shape[1]          
        msg.info.height = map_mgr.shape[0]         
        msg.info.origin.position.x = map_mgr.origin[0]
        msg.info.origin.position.y = map_mgr.origin[1]
        msg.info.origin.orientation.w = 1.0
        
        ogm = np.fliplr(map_mgr.occupancy_gridmap)
        msg.data = ogm.flatten().tolist()

        return msg

    def create_static_transform(self):
        static_transforms = []

        for i in range(self.num_envs):
            # map -> odom (identity for now; later this can come from localization)
            map_to_odom = TransformStamped()
            map_to_odom.header.stamp = self.get_clock().now().to_msg()
            map_to_odom.header.frame_id = self.map_frame(i)
            map_to_odom.child_frame_id = self.odom_frame(i)
            map_to_odom.transform.translation.x = 0.0
            map_to_odom.transform.translation.y = 0.0
            map_to_odom.transform.translation.z = 0.0
            map_to_odom.transform.rotation.x = 0.0
            map_to_odom.transform.rotation.y = 0.0
            map_to_odom.transform.rotation.z = 0.0
            map_to_odom.transform.rotation.w = 1.0
            static_transforms.append(map_to_odom)

            # base_link -> lidar_frame
            base_lidar_transform = TransformStamped()
            base_lidar_transform.header.stamp = self.get_clock().now().to_msg()
            base_lidar_transform.header.frame_id = self.base_frame(i)
            base_lidar_transform.child_frame_id = self.lidar_frame(i)
            base_lidar_transform.transform.translation.x = 0.2
            base_lidar_transform.transform.translation.y = 0.0
            base_lidar_transform.transform.translation.z = 0.2
            base_lidar_transform.transform.rotation.x = 0.0
            base_lidar_transform.transform.rotation.y = 0.0
            base_lidar_transform.transform.rotation.z = 0.0
            base_lidar_transform.transform.rotation.w = 1.0
            static_transforms.append(base_lidar_transform)

            # base_link -> front_cam
            base_cam_transform = TransformStamped()
            base_cam_transform.header.stamp = self.get_clock().now().to_msg()
            base_cam_transform.header.frame_id = self.base_frame(i)
            base_cam_transform.child_frame_id = self.camera_frame(i)
            base_cam_transform.transform.translation.x = 0.4
            base_cam_transform.transform.translation.y = 0.0
            base_cam_transform.transform.translation.z = 0.2
            base_cam_transform.transform.rotation.x = -0.5
            base_cam_transform.transform.rotation.y = 0.5
            base_cam_transform.transform.rotation.z = -0.5
            base_cam_transform.transform.rotation.w = 0.5
            static_transforms.append(base_cam_transform)

        self.static_broadcaster.sendTransform(static_transforms)

    def create_camera_publisher(self):
        if self.cfg.sensor.enable_camera:
            self.publish_camera_info()

    def publish_odom(
        self, base_pos: Tensor, base_rot: Tensor, base_lin_vel_b: Tensor, base_ang_vel_b: Tensor, env_idx: int
    ):
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = self.odom_frame(env_idx)
        odom.child_frame_id = self.base_frame(env_idx)

        odom.pose.pose.position.x = base_pos[0].item()
        odom.pose.pose.position.y = base_pos[1].item()
        odom.pose.pose.position.z = base_pos[2].item()
        odom.pose.pose.orientation.x = base_rot[1].item()
        odom.pose.pose.orientation.y = base_rot[2].item()
        odom.pose.pose.orientation.z = base_rot[3].item()
        odom.pose.pose.orientation.w = base_rot[0].item()

        odom.twist.twist.linear.x = base_lin_vel_b[0].item()
        odom.twist.twist.linear.y = base_lin_vel_b[1].item()
        odom.twist.twist.linear.z = base_lin_vel_b[2].item()
        odom.twist.twist.angular.x = base_ang_vel_b[0].item()
        odom.twist.twist.angular.y = base_ang_vel_b[1].item()
        odom.twist.twist.angular.z = base_ang_vel_b[2].item()
        self.odom_pub[env_idx].publish(odom)

        odom_tf = TransformStamped()
        odom_tf.header.stamp = self.get_clock().now().to_msg()
        odom_tf.header.frame_id = self.odom_frame(env_idx)
        odom_tf.child_frame_id = self.base_frame(env_idx)
        odom_tf.transform.translation.x = base_pos[0].item()
        odom_tf.transform.translation.y = base_pos[1].item()
        odom_tf.transform.translation.z = base_pos[2].item()
        odom_tf.transform.rotation.x = base_rot[1].item()
        odom_tf.transform.rotation.y = base_rot[2].item()
        odom_tf.transform.rotation.z = base_rot[3].item()
        odom_tf.transform.rotation.w = base_rot[0].item()
        self.broadcaster.sendTransform(odom_tf)

    def publish_pose(self, base_pos: Tensor, base_rot: Tensor, env_idx: int):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.map_frame(env_idx)
        pose_msg.pose.position.x = base_pos[0].item()
        pose_msg.pose.position.y = base_pos[1].item()
        pose_msg.pose.position.z = base_pos[2].item()
        pose_msg.pose.orientation.x = base_rot[1].item()
        pose_msg.pose.orientation.y = base_rot[2].item()
        pose_msg.pose.orientation.z = base_rot[3].item()
        pose_msg.pose.orientation.w = base_rot[0].item()
        self.pose_pub[env_idx].publish(pose_msg)

    def publish_lidar_data(self, points: Tensor, env_idx: int):
        point_cloud = PointCloud2()
        point_cloud.header.frame_id = self.map_frame(env_idx) # lidar data is in isaac local frame, which corresponds to the map frame in ROS2.
        point_cloud.header.stamp = self.get_clock().now().to_msg()

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        point_cloud = point_cloud2.create_cloud(point_cloud.header, fields, points)
        self.lidar_pub[env_idx].publish(point_cloud)

    def pub_ros2_data(self):
        pub_odom_pose = False
        pub_lidar = False
        pub_camera = False
        dt_odom_pose = time.time() - self.odom_pose_pub_time
        dt_lidar = time.time() - self.lidar_pub_time
        dt_camera = time.time() - self.camera_pub_time
        if dt_odom_pose >= 1.0 / self.odom_pose_freq:
            pub_odom_pose = True

        if dt_lidar >= 1.0 / self.lidar_freq:
            pub_lidar = True

        if dt_camera >= 1.0 / self.camera_freq:
            pub_camera = True

        if pub_odom_pose:
            self.odom_pose_pub_time = time.time()
            robot_data = self.env.unwrapped.scene["robot"].data
            for i in range(self.num_envs):
                base_pose_local = wp.to_torch(robot_data.root_state_w)[i, :3] - self.env_origin(i)  # convert to local coordinates
                self.publish_odom(
                    base_pose_local,
                    wp.to_torch(robot_data.root_state_w)[i, 3:7],
                    wp.to_torch(robot_data.root_lin_vel_b)[i],
                    wp.to_torch(robot_data.root_ang_vel_b)[i],
                    i,
                )
                self.publish_pose(
                    base_pose_local, wp.to_torch(robot_data.root_state_w)[i, 3:7], i
                )
        if self.cfg.sensor.enable_lidar:
            if pub_lidar:
                self.lidar_pub_time = time.time()
                scan_local = self.lidar.data.ray_hits_w - self.env.scene.env_origins.unsqueeze(1)  # convert to local coordinates
                for i in range(self.num_envs):
                    self.publish_lidar_data(scan_local[i], i)

        if self.cfg.sensor.enable_camera and pub_camera:
            self.camera_pub_time = time.time()
            if self.cfg.sensor.color_image:
                self.pub_color_image()
            if self.cfg.sensor.depth_image:
                self.pub_depth_image()

    def cmd_vel_callback(self, msg: Twist, env_idx: int):
        self.base_vel_cmd_input[env_idx][0] = msg.linear.x
        self.base_vel_cmd_input[env_idx][1] = msg.linear.y
        self.base_vel_cmd_input[env_idx][2] = msg.angular.z

    def pub_image_graph(self):
        for i in range(self.num_envs):
            keys = og.Controller.Keys
            og.Controller.edit(
                {
                    "graph_path": f"/CameraROS2Graph_{i}",
                    "evaluator_name": "execution",
                },
                {
                    keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("IsaacCreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                        ("ROS2CameraHelperColor", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("ROS2CameraHelperDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ],
                    keys.SET_VALUES: [
                        ("IsaacCreateRenderProduct.inputs:cameraPrim", f"/World/envs/env_{i}/Go2/base/front_cam"),
                        ("IsaacCreateRenderProduct.inputs:enabled", True),
                        ("IsaacCreateRenderProduct.inputs:height", 480),
                        ("IsaacCreateRenderProduct.inputs:width", 640),

                        ("ROS2CameraHelperColor.inputs:type", "rgb"),
                        ("ROS2CameraHelperColor.inputs:topicName", self.color_topic(i)),
                        ("ROS2CameraHelperColor.inputs:frameId", self.camera_frame(i)),
                        ("ROS2CameraHelperColor.inputs:useSystemTime", False),

                        ("ROS2CameraHelperDepth.inputs:type", "depth"),
                        ("ROS2CameraHelperDepth.inputs:topicName", self.depth_topic(i)),
                        ("ROS2CameraHelperDepth.inputs:frameId", self.camera_frame(i)),
                        ("ROS2CameraHelperDepth.inputs:useSystemTime", False),
                    ],
                    keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "IsaacCreateRenderProduct.inputs:execIn"),
                        ("IsaacCreateRenderProduct.outputs:execOut", "ROS2CameraHelperColor.inputs:execIn"),
                        ("IsaacCreateRenderProduct.outputs:renderProductPath", "ROS2CameraHelperColor.inputs:renderProductPath"),
                        ("IsaacCreateRenderProduct.outputs:execOut", "ROS2CameraHelperDepth.inputs:execIn"),
                        ("IsaacCreateRenderProduct.outputs:renderProductPath", "ROS2CameraHelperDepth.inputs:renderProductPath"),
                    ],
                },
            )

    def pub_color_image(self):
        data = self.cameras.data.output.get("rgb")
        if data is None:
            return
        for i in range(self.num_envs):
            if self.num_envs == 1:
                frame_id = "unitree_go2/front_cam"
            else:
                frame_id = f"unitree_go2_{i}/front_cam"
            self.publish_image(data[i], frame_id, "rgb8", self.color_pub[i])

    def pub_depth_image(self):
        data = self.cameras.data.output.get("depth")
        if data is None:
            return
        for i in range(self.num_envs):
            if self.num_envs == 1:
                frame_id = "unitree_go2/front_cam"
            else:
                frame_id = f"unitree_go2_{i}/front_cam"
            depth = data[i]
            if depth.ndim == 2:
                depth = depth.unsqueeze(-1)
            self.publish_image(depth, frame_id, "32FC1", self.depth_pub[i])

    def pub_cam_depth_cloud(self):
        for i in range(self.num_envs):
            # The following code will link the camera's render product and publish the data to the specified topic name.
            render_product = self.cameras.render_product_paths[i]
            step_size = 1
            if self.num_envs == 1:
                topic_name = "unitree_go2/front_cam/depth_cloud"
                frame_id = "unitree_go2/front_cam"
            else:
                topic_name = f"unitree_go2_{i}/front_cam/depth_cloud"
                frame_id = f"unitree_go2_{i}/front_cam"
            node_namespace = ""
            queue_size = 1

            # Note, this pointcloud publisher will simply convert the Depth image to a pointcloud using the Camera intrinsics.
            # This pointcloud generation method does not support semantic labelled objects.
            rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                sd.SensorType.DistanceToImagePlane.name
            )

            writer = rep.writers.get(rv + "ROS2PublishPointCloud")
            writer.initialize(
                frameId=frame_id,
                nodeNamespace=node_namespace,
                queueSize=queue_size,
                topicName=topic_name,
            )
            writer.attach([render_product])

            # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
            gate_path = omni.syntheticdata.SyntheticData._get_node_path(rv + "IsaacSimulationGate", render_product)

            og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    def publish_camera_info(self):
        for i in range(self.num_envs):
            if self.num_envs == 1:
                frame_id = "unitree_go2/front_cam"
            else:
                frame_id = f"unitree_go2_{i}/front_cam"

            height, width = self.cameras.image_shape
            intrinsic = self.cameras.data.intrinsic_matrices[i].cpu()
            k = intrinsic.flatten().tolist()
            p = [
                k[0],
                k[1],
                k[2],
                0.0,
                k[3],
                k[4],
                k[5],
                0.0,
                k[6],
                k[7],
                k[8],
                0.0,
            ]

            msg = CameraInfo()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = frame_id
            msg.height = height
            msg.width = width
            msg.distortion_model = "plumb_bob"
            msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            msg.k = k
            msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            msg.p = p
            self.camera_info_pub[i].publish(msg)

    def publish_image(self, tensor: Tensor, frame_id: str, encoding: str, publisher):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        height, width, channels = tensor.shape
        msg.height = int(height)
        msg.width = int(width)
        msg.encoding = encoding
        msg.is_bigendian = False
        msg.step = int(width) * int(channels) * int(tensor.element_size())
        msg.data = tensor.detach().cpu().contiguous().numpy().tobytes()
        publisher.publish(msg)

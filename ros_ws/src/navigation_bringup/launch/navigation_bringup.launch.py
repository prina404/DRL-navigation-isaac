import os
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml

ROOT_DIR = Path(__file__).resolve().parents[6]
DATASET_CFG = yaml.safe_load((ROOT_DIR / "dataset_cfg.yaml").open())
MAP_FOLDER = (ROOT_DIR / DATASET_CFG["dataset_folder"]).resolve() / DATASET_CFG["env_folder"]


def _create_robot_nav2_nodes(context):
    params_file = LaunchConfiguration("params_file").perform(context)
    robot_prefix = LaunchConfiguration("robot_prefix").perform(context)
    num_robots = int(LaunchConfiguration("num_robots").perform(context))
    use_sim_time_str = LaunchConfiguration("use_sim_time").perform(context)
    use_sim_time = use_sim_time_str.lower() in ("true", "1")

    actions = []

    for i in range(num_robots):
        ns = f"{robot_prefix}_{i}"
        map_frame = f"{ns}/map"
        odom_frame = f"{ns}/odom"
        base_frame = f"{ns}/base_link"

        # Keep /tf and /tf_static independent per robot namespace.
        remappings = []

        configured_params = RewrittenYaml(
            source_file=params_file,
            root_key=ns,
            param_rewrites={
                "use_sim_time": use_sim_time_str,
                "bt_navigator.ros__parameters.global_frame": map_frame,
                "bt_navigator.ros__parameters.robot_base_frame": base_frame,
                "controller_server.ros__parameters.odom_topic": "odom",
                "global_costmap.global_costmap.ros__parameters.global_frame": map_frame,
                "global_costmap.global_costmap.ros__parameters.robot_base_frame": base_frame,
                "local_costmap.local_costmap.ros__parameters.global_frame": odom_frame,
                "local_costmap.local_costmap.ros__parameters.robot_base_frame": base_frame,
                "behavior_server.ros__parameters.global_frame": map_frame,
                "behavior_server.ros__parameters.local_frame": odom_frame,
                "behavior_server.ros__parameters.robot_base_frame": base_frame,
                # "amcl.ros__parameters.global_frame_id": map_frame,
                # "amcl.ros__parameters.odom_frame_id": odom_frame,
                # "amcl.ros__parameters.base_frame_id": base_frame,
                # Relative names resolve under each namespace: /robot_i/...
                "odom_topic": "odom",
                # "scan_topic": "lidar",
                "local_costmap.local_costmap.ros__parameters.voxel_layer.point_cloud.topic": f"/{ns}/lidar",
                "global_costmap.global_costmap.ros__parameters.obstacle_layer.point_cloud.topic": f"/{ns}/lidar",
                "local_costmap.local_costmap.ros__parameters.voxel_layer.point_cloud.sensor_frame": f"{ns}/base_link",
                "global_costmap.global_costmap.ros__parameters.obstacle_layer.point_cloud.sensor_frame": f"{ns}/base_link",
                "collision_monitor.ros__parameters.base_frame_id": base_frame,
                "collision_monitor.ros__parameters.odom_frame_id": odom_frame,
            },
            convert_types=True,
        )
        node_params = [configured_params, {"use_sim_time": use_sim_time}]

        lifecycle_nodes = [
            # "amcl",
            "planner_server",
            "controller_server",
            'smoother_server',
            "bt_navigator",
            "behavior_server",
            #'velocity_smoother',
            # 'collision_monitor',
        ]

        actions.extend(
            [
                # Node(
                #     package="nav2_amcl",
                #     executable="amcl",
                #     name="amcl",
                #     namespace=ns,
                #     output="screen",
                #     parameters=[configured_params],
                #     remappings=remappings,
                # ),
                Node(
                    package="nav2_planner",
                    executable="planner_server",
                    name="planner_server",
                    namespace=ns,
                    respawn=True,
                    respawn_delay=1.0,
                    output="screen",
                    # arguments=["--ros-args", "--log-level", "debug"],
                    parameters=node_params,
                    remappings=remappings,
                ),
                Node(
                    package="nav2_controller",
                    executable="controller_server",
                    name="controller_server",
                    namespace=ns,
                    respawn=True,
                    respawn_delay=1.0,
                    output="screen",
                    arguments=["--ros-args", "--log-level", "info"],
                    parameters=node_params,
                    remappings=remappings,
                ),
                Node(
                    package='nav2_smoother',
                    executable='smoother_server',
                    name='smoother_server',
                    output='screen',
                    namespace=ns,
                    respawn=True,
                    respawn_delay=1.0,
                    # arguments=['--ros-args', '--log-level', "debug"],
                    parameters=node_params,
                    remappings=remappings,
                ),
                Node(
                    package="nav2_bt_navigator",
                    executable="bt_navigator",
                    name="bt_navigator",
                    namespace=ns,
                    respawn=True,
                    respawn_delay=1.0,
                    output="screen",
                    arguments=["--ros-args", "--log-level", "info"],
                    parameters=node_params,
                    remappings=remappings,
                ),
                Node(
                    package="nav2_lifecycle_manager",
                    executable="lifecycle_manager",
                    name="lifecycle_manager_navigation",
                    namespace=ns,
                    respawn=True,
                    respawn_delay=1.0,
                    output="screen",
                    # arguments=["--ros-args", "--log-level", "debug"],
                    parameters=[
                        configured_params,
                        {
                            "use_sim_time": use_sim_time,
                            "autostart": True,
                            "node_names": lifecycle_nodes,
                        },
                    ],
                ),
                Node(
                    package="nav2_behaviors",
                    executable="behavior_server",
                    name="behavior_server",
                    namespace=ns,
                    respawn=True,
                    respawn_delay=1.0,
                    output="screen",
                    # arguments=["--ros-args", "--log-level", "debug"],
                    parameters=node_params,
                    remappings=remappings,
                ),
                # Node(
                #     package='nav2_velocity_smoother',
                #     executable='velocity_smoother',
                #     name='velocity_smoother',
                #     output='screen',
                #     namespace=ns,
                #     respawn=True,
                #     respawn_delay=1.0,
                #     parameters=node_params,
                #     # arguments=['--ros-args', '--log-level', "info"],
                #     remappings=remappings,
                # ),
                # Node(
                #     package='nav2_collision_monitor',
                #     executable='collision_monitor',
                #     name='collision_monitor',
                #     output='screen',
                #     namespace=ns,
                #     respawn=True,
                #     respawn_delay=1.0,
                #     parameters=node_params,
                #     # arguments=['--ros-args', '--log-level', "info"],
                #     remappings=remappings,
                # ),
            ]
        )

    return actions


def rviz_node():
    return Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )


def generate_launch_description():
    ws_folder = get_package_share_directory("navigation_bringup")
    config_dir = os.path.join(ws_folder, "config")
    nav2_params = os.path.join(config_dir, "nav2_params.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument("num_robots", default_value="2"),
            DeclareLaunchArgument("robot_prefix", default_value="robot"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("params_file", default_value=nav2_params),
            OpaqueFunction(function=_create_robot_nav2_nodes),
            #rviz_node(),
        ]
    )

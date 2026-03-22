import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from nav2_common.launch import RewrittenYaml
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[6]
DATASET_CFG = yaml.safe_load((ROOT_DIR / "dataset_cfg.yaml").open())
MAP_FOLDER = (ROOT_DIR / DATASET_CFG["dataset_folder"]).resolve() / DATASET_CFG["env_folder"] 


def _create_robot_nav2_nodes(context):
    params_file = LaunchConfiguration("params_file").perform(context)
    map_file = LaunchConfiguration("map").perform(context)
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
                "global_frame": map_frame,
                "global_frame_id": map_frame,
                "odom_frame_id": odom_frame,
                "robot_base_frame": base_frame,
                "base_frame_id": base_frame,
                # Relative names resolve under each namespace: /robot_i/...
                "odom_topic": "odom",
                #"scan_topic": "lidar",
                "local_costmap.local_costmap.ros__parameters.voxel_layer.point_cloud.topic": f"/{ns}/lidar",
                "global_costmap.global_costmap.ros__parameters.obstacle_layer.point_cloud.topic": f"/{ns}/lidar",
                "local_costmap.local_costmap.ros__parameters.voxel_layer.point_cloud.sensor_frame": f"{ns}/base_link",
                "global_costmap.global_costmap.ros__parameters.obstacle_layer.point_cloud.sensor_frame": f"{ns}/base_link",
           
            },
            convert_types=True,
        )
        node_params = [configured_params]

        lifecycle_nodes = [
            # "map_server",
            # "amcl",
            "planner_server",
            "controller_server",
            "bt_navigator",
            "behavior_server",
        ]

        actions.extend(
            [
                # Node(
                #     package="nav2_map_server",
                #     executable="map_server",
                #     name="map_server",
                #     namespace=ns,
                #     output="screen",
                #     parameters=[
                #         configured_params,
                #         {
                #             "yaml_filename": map_file,
                #             "use_sim_time": use_sim_time,
                #             "frame_id": map_frame,
                #         },
                #     ],
                #     remappings=remappings,
                # ),
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
                    output="screen",
                    parameters=node_params,
                    remappings=remappings,
                ),
                Node(
                    package="nav2_controller",
                    executable="controller_server",
                    name="controller_server",
                    namespace=ns,
                    output="screen",
                    parameters=node_params,
                    remappings=remappings,
                ),
                Node(
                    package="nav2_bt_navigator",
                    executable="bt_navigator",
                    name="bt_navigator",
                    namespace=ns,
                    output="screen",
                    parameters=node_params,
                    remappings=remappings,
                ),
                Node(
                    package="nav2_lifecycle_manager",
                    executable="lifecycle_manager",
                    name="lifecycle_manager_navigation",
                    namespace=ns,
                    output="screen",
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
                    output="screen",
                    parameters=node_params,
                    remappings=remappings,
                ),
            ]
        )

    return actions

def generate_launch_description():
    ws_folder = get_package_share_directory('navigation_bringup')
    config_dir = os.path.join(ws_folder, 'config')
    nav2_params = os.path.join(config_dir, 'nav2_params.yaml')

    map_file = str(MAP_FOLDER / f"{DATASET_CFG['env_name']}_map.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("num_robots", default_value="2"),
        DeclareLaunchArgument("robot_prefix", default_value="robot"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("params_file", default_value=nav2_params),
        DeclareLaunchArgument("map", default_value=map_file),
        OpaqueFunction(function=_create_robot_nav2_nodes),
    ])

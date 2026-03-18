import os
import yaml

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[6]
DATASET_CFG = yaml.safe_load((ROOT_DIR / "dataset_cfg.yaml").open())
MAP_FOLDER = (ROOT_DIR / DATASET_CFG["dataset_folder"]).resolve() / DATASET_CFG["env_folder"] 

def generate_launch_description():
    ws_folder = get_package_share_directory('navigation_bringup')
    config_dir = os.path.join(ws_folder, 'config')
    nav2_params = os.path.join(config_dir, 'nav2_params.yaml')

    map_file = str(MAP_FOLDER / f"{DATASET_CFG['env_name']}_map.yaml")
    use_sim_time = True

    lifecycle_nodes = [
        "map_server",
        "amcl",
        "planner_server",
        "controller_server",
        "bt_navigator",
    ]

    return LaunchDescription([
        Node(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            output="screen",
            parameters=[
                {"yaml_filename": map_file},
                {"use_sim_time": use_sim_time}
            ],
        ),

        Node(
            package="nav2_amcl",
            executable="amcl",
            name="amcl",
            output="screen",
            parameters=[nav2_params],
        ),

        Node(
            package="nav2_planner",
            executable="planner_server",
            name="planner_server",
            output="screen",
            parameters=[nav2_params],
        ),

        Node(
            package="nav2_controller",
            executable="controller_server",
            name="controller_server",
            output="screen",
            parameters=[nav2_params],
        ),

        Node(
            package="nav2_bt_navigator",
            executable="bt_navigator",
            name="bt_navigator",
            output="screen",
            parameters=[nav2_params],
        ),

        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_navigation",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "autostart": True,
                "node_names": lifecycle_nodes,
            }],
        ),

    ])

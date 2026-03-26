import subprocess
import time

from typing import Sequence
from loguru import logger

from cfg.CFG import ROOT_DIR

from nav2_simple_commander.robot_navigator import BasicNavigator
from lab.MyEnv import MyEnv
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time
import torch

class MultiEnvNavigator():
    def __init__(self, env: MyEnv):
        self.env = env

        self.ns = robot_namespaces(env.num_envs)
        self.navigators = [BasicNavigator(namespace=ns) for ns in self.ns]

        self.current_goal_poses = env.path_manager.goal_pos_local.clone().cpu()
        self.goal_queue = list(range(env.num_envs))

    def _send_goals(self):
        retry_env_ids = []

        robot_pos_map = self.env.path_manager.start_pos_map
        robot_pos_local = self.env._map_manager.map_to_local_coords(robot_pos_map).cpu()
        logger.info(f"robot local positions: {robot_pos_local[self.goal_queue].tolist()}, goal positions: {self.current_goal_poses[self.goal_queue].tolist()}")
        logger.debug(f"Current robot pose: {self.env.scene['robot'].data.root_com_pos_w}")
        for env_id in self.goal_queue:
            new_goal = self.current_goal_poses[env_id]
            goal_msg = coord_to_pose(new_goal.tolist(), self.ns[env_id])

            # pos_msg = coord_to_pose(robot_pos_local[env_id].tolist(), self.ns[env_id])
            # self.navigators[env_id].setInitialPose(pos_msg) # reset initial pose after teleport

            if not self.navigators[env_id].goToPose(goal_msg):
                retry_env_ids.append(env_id) # store for later to avoid blocking others
        
        max_retry = 3
        for id in retry_env_ids:
            logger.debug(f"Retrying to send goal for env {id}...")
            for _ in range(max_retry):
                new_goal = self.current_goal_poses[id]
                goal_msg = coord_to_pose(new_goal.tolist(), self.ns[id])
                if self.navigators[id].goToPose(goal_msg):
                    break
                time.sleep(0.2)
        
    def step(self):
        # send goals first, then compute new ones, this creates a 1-step delay needed for correct 
        # teleport coordinate publishing
        if len(self.goal_queue) > 0:
            self._send_goals()
            self.goal_queue = []
        
        # check if any env goal has been updated, if so cancel previous goal and send new one
        diff_goal = self.current_goal_poses != self.env.path_manager.goal_pos_local.cpu() # (N, 2)
        diff_goal = diff_goal.any(dim=-1)  # (N, )
        if diff_goal.any():
            env_ids = torch.where(diff_goal)[0].tolist()
            logger.debug(f"Updating goals for envs: {env_ids}")
            logger.debug(f"Current robot pose: {self.env.scene['robot'].data.root_com_pos_w}")
            for env_id in env_ids:
                self.navigators[env_id].cancelTask()
                #self.navigators[env_id].clearAllCostmaps()
                        
            self.current_goal_poses = self.env.path_manager.goal_pos_local.clone().cpu() # update stored goals to new ones
            self.goal_queue.extend(env_ids)
    


def coord_to_pose(coord: Sequence[float], namespace: str) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = namespace + "/map"
    pose.pose.position.x = coord[0]
    pose.pose.position.y = coord[1]
    pose.pose.position.z = 0.0
    return pose
            

def robot_namespaces(num_envs: int):
    # Keep this consistent with RosDataManager + launch file naming
    return [f"robot_{i}" for i in range(num_envs)]


def wait_for_nav2_ready(ros2_dm, num_envs: int, robot_prefix: str = "robot", timeout: float = 120.0) -> None:
    """Block until every robot's planner_server reports 'active',
    or raise RuntimeError on timeout."""

    nodes_to_check = (
        [f"/{robot_prefix}_{i}/planner_server" for i in range(num_envs)]
    )
    deadline = time.monotonic() + timeout
    pending = set(nodes_to_check)
    logger.info(f"Waiting for Nav2 nodes to become active: {pending}")
    while pending:
        ros2_dm.pub_ros2_data(ros2_dm.get_time())

        if time.monotonic() > deadline:
            raise RuntimeError(f"Nav2 nodes did not become active within {timeout}s: {pending}")
        for node in list(pending):
            try:
                result = subprocess.run(
                    ["bash", "-lc", f"source {ROOT_DIR}/ros_ws/install/setup.bash && " f"ros2 lifecycle get {node}"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if "active" in result.stdout:
                    logger.info(f"  {node} is active ✓")
                    pending.discard(node)
            except subprocess.TimeoutExpired:
                pass
        if pending:
            time.sleep(1.0)
    logger.info("All Nav2 nodes active — starting eval loop.")


def kill_nav2_lifecycle():
    out = subprocess.check_output(
        ["bash", "-lc", f"source {ROOT_DIR}/ros_ws/install/setup.bash && ros2 lifecycle nodes"],
        text=True,
    )
    prefixes = set()
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("/"):
            continue
        parts = line.strip("/").split("/")
        if len(parts) >= 2:
            prefixes.add(parts[0])  # robot_0, robot_1, ...
    if len(prefixes) > 0:
        logger.debug(f"killing all lifecycle nodes under: {prefixes}")
        subprocess.Popen(["pkill -f robot_"], shell=True)
    subprocess.Popen(["pkill -f rviz2"], shell=True)

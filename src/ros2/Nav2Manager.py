import subprocess
import time

from typing import Sequence
from loguru import logger

from cfg.CFG import ROOT_DIR

from nav2_simple_commander.robot_navigator import BasicNavigator
from lab.MyEnv import MyEnv
from geometry_msgs.msg import PoseStamped
import torch

class MultiEnvNavigator():
    def __init__(self, env: MyEnv):
        self.env = env

        self.ns = robot_namespaces(env.num_envs)
        self.navigators = [BasicNavigator(namespace=ns) for ns in self.ns]

        self.current_goal_poses = env.path_manager.goal_pos_local.clone().cpu()
        self._send_goals(list(range(env.num_envs)))

    def _send_goals(self, env_ids: Sequence[int]):
        for env_id in env_ids:
            new_goal = self.current_goal_poses[env_id]
            goal_msg = coord_to_pose(new_goal.tolist(), self.ns[env_id])
            self.navigators[env_id].goToPose(goal_msg)

    def step(self):
        # check if any env goal has been updated, if so cancel previous goal and send new one
        diff_goal = self.current_goal_poses != self.env.path_manager.goal_pos_local.cpu() # (N, 2)
        diff_goal = diff_goal.any(dim=-1)  # (N, )
        if diff_goal.any():
            env_ids = torch.where(diff_goal)[0].tolist()
            logger.debug(f"Updating goals for envs: {env_ids}")
            for env_id in env_ids:
                self.navigators[env_id].cancelTask()
                #self.navigators[env_id].clearAllCostmaps()
                        
            self.current_goal_poses = self.env.path_manager.goal_pos_local.clone().cpu() # update stored goals to new ones
            self._send_goals(env_ids)


def coord_to_pose(coord: Sequence[float], namespace: str) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = namespace + "/map"
    pose.pose.position.x = coord[0]
    pose.pose.position.y = coord[1]
    pose.pose.position.z = 0.0
    return pose
            

LIFECYCLE_NODES = ["planner_server", "controller_server", "bt_navigator", "behavior_server"]


def robot_namespaces(num_envs: int):
    # Keep this consistent with RosDataManager + launch file naming
    return ["robot"] if num_envs == 1 else [f"robot_{i}" for i in range(num_envs)]


def wait_for_nav2_ready(ros2_dm, num_envs: int, robot_prefix: str = "robot", timeout: float = 120.0) -> None:
    """Block until every robot's planner_server reports 'active',
    or raise RuntimeError on timeout."""
    kill_nav2_lifecycle()  # ensure a clean slate before waiting

    nodes_to_check = (
        [f"/{robot_prefix}/planner_server"] if num_envs == 1 else [f"/{robot_prefix}_{i}/planner_server" for i in range(num_envs)]
    )
    deadline = time.monotonic() + timeout
    pending = set(nodes_to_check)
    logger.info(f"Waiting for Nav2 nodes to become active: {pending}")
    while pending:
        ros2_dm.pub_ros2_data()

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

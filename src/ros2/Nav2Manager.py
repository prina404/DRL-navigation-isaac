import subprocess
import time

from loguru import logger

from cfg.CFG import ROOT_DIR

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

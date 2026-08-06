import subprocess
import time

from functools import partial
from typing import Sequence
from loguru import logger

from cfg.CFG import ROOT_DIR

from nav2_msgs.action import NavigateToPose
from nav2_simple_commander.robot_navigator import BasicNavigator
from navigation_env.NavigationEnv import NavEnv
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time
from rclpy.executors import SingleThreadedExecutor
import torch

MAX_GOAL_ATTEMPTS = 5
"""Give up on an env after this many rejected goals; it gets a fresh one on the next episode."""

SPIN_BUDGET = 16
"""Callbacks drained per step while a goal/cancel response is in flight."""


class MultiEnvNavigator():
    """Drives one Nav2 stack per environment from inside the Isaac stepping loop.

    Every method here must be non-blocking. The simulator is the only publisher of the
    ``odom -> base_link`` transform, so stalling this class stops TF, which stops the
    costmaps from activating, which stops the action servers from ever appearing --
    a deadlock. That rules out the ``BasicNavigator`` helpers (``goToPose``,
    ``cancelTask``, ...): they wait on the action server and then call
    ``rclpy.spin_until_future_complete``. We drive the underlying action clients
    directly and pump them with our own executor instead.
    """

    def __init__(self, env: NavEnv):
        self.env = env

        self.ns = robot_namespaces(env.num_envs)
        self.navigators = [BasicNavigator(namespace=ns) for ns in self.ns]

        # The BasicNavigator nodes belong to this executor. Nothing else may spin them:
        # rclpy refuses to add a node to a second executor, so a stray
        # rclpy.spin_until_future_complete(nav, ...) would block forever.
        self._executor = SingleThreadedExecutor()
        for nav in self.navigators:
            self._executor.add_node(nav)

        self.current_goal_poses = env.path_manager.goal_pos_local.clone().cpu()
        self.goal_queue = list(range(env.num_envs))
        self._goal_handles = [None] * env.num_envs
        self._goal_attempts = [0] * env.num_envs
        self._waiting_on = None  # last set of envs waiting for a server, for throttled logging
        self._pending_responses = 0  # in-flight goal/cancel requests, see _track()

    def _track(self, future, done_cb=None):
        """Count an in-flight request so _spin_some() knows to drain aggressively."""
        self._pending_responses += 1

        def _on_done(fut):
            self._pending_responses -= 1
            if done_cb is not None:
                done_cb(fut)

        future.add_done_callback(_on_done)

    def _spin_some(self):
        """Drain pending ROS work for the navigator nodes without ever blocking.

        spin_once() handles at most one callback and costs ~0.2 ms even when idle, so a
        fixed large drain would burn several ms of every sim step. Pay for the full drain
        only while a goal or cancel response is actually in flight; otherwise one tick is
        enough to keep the action clients ticking over. Nothing reads the feedback topic,
        so letting it drop on a KEEP_LAST queue is fine.
        """
        budget = SPIN_BUDGET if self._pending_responses else 1
        for _ in range(budget):
            self._executor.spin_once(timeout_sec=0.0)

    def _queue(self, env_id: int):
        if env_id not in self.goal_queue:
            self.goal_queue.append(env_id)

    def _send_goals(self):
        """Fire off a goal for every queued env whose action server is already up.

        Envs whose server is not ready yet stay queued and are retried on the next step.
        This replaces the old blocking wait_for_server + sleep retry loop, which stalled
        the whole stepping loop until Nav2 came up -- and Nav2 could not come up while
        the loop was stalled.
        """
        still_queued = []

        robot_pos_map = self.env.path_manager.start_pos_map
        robot_pos_local = self.env._map_manager.map_to_local_coords(robot_pos_map).cpu()

        for env_id in self.goal_queue:
            nav = self.navigators[env_id]
            if not nav.nav_to_pose_client.server_is_ready():
                still_queued.append(env_id)
                continue

            if self._goal_attempts[env_id] >= MAX_GOAL_ATTEMPTS:
                logger.error(f"Env {env_id}: Nav2 rejected the goal {MAX_GOAL_ATTEMPTS} times, dropping it.")
                continue

            new_goal = self.current_goal_poses[env_id]
            logger.info(
                f"Env {env_id}: robot local position {robot_pos_local[env_id].tolist()}, "
                f"goal position {new_goal.tolist()}"
            )

            # pos_msg = coord_to_pose(robot_pos_local[env_id].tolist(), self.ns[env_id])
            # self.navigators[env_id].setInitialPose(pos_msg) # reset initial pose after teleport

            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = coord_to_pose(new_goal.tolist(), self.ns[env_id])
            self._goal_attempts[env_id] += 1
            future = nav.nav_to_pose_client.send_goal_async(goal_msg, nav._feedbackCallback)
            self._track(future, partial(self._on_goal_response, env_id))

        if still_queued != self._waiting_on:
            self._waiting_on = list(still_queued)  # copy: still_queued is about to become self.goal_queue
            if still_queued:
                logger.info(f"NavigateToPose server not up yet for envs {still_queued}, retrying next step.")

        self.goal_queue = still_queued

    def _on_goal_response(self, env_id: int, future):
        """Record the accepted goal handle so it can be cancelled later.

        Runs inside _spin_some(), so an exception here would escape step() and kill the
        eval loop -- swallow it and let the env retry instead.
        """
        try:
            goal_handle = future.result()
        except Exception as exc:  # noqa: BLE001 - never let a goal response kill the run
            logger.warning(f"Env {env_id}: NavigateToPose goal request failed ({exc}), re-queueing.")
            self._queue(env_id)
            return
        if goal_handle is None or not goal_handle.accepted:
            logger.warning(f"Env {env_id}: Nav2 rejected the goal, re-queueing.")
            self._queue(env_id)
            return
        self._goal_handles[env_id] = goal_handle

    def _cancel_goal(self, env_id: int):
        """Cancel the in-flight goal for one env, fire-and-forget.

        The cancel response is drained by our executor on a later step; waiting for it
        here would stall the simulator.
        """
        goal_handle = self._goal_handles[env_id]
        self._goal_handles[env_id] = None
        if goal_handle is not None:
            self._track(goal_handle.cancel_goal_async())

    def step(self):
        # Process goal responses / cancellations from the previous step before looking
        # at the queue, so accepted goals are recorded and rejected ones are re-queued.
        self._spin_some()

        # send goals first, then compute new ones, this creates a 1-step delay needed for correct
        # teleport coordinate publishing
        if len(self.goal_queue) > 0:
            self._send_goals()

        # check if any env goal has been updated, if so cancel previous goal and send new one
        diff_goal = self.current_goal_poses != self.env.path_manager.goal_pos_local.cpu() # (N, 2)
        diff_goal = diff_goal.any(dim=-1)  # (N, )
        if diff_goal.any():
            env_ids = torch.where(diff_goal)[0].tolist()
            logger.debug(f"Updating goals for envs: {env_ids}")
            logger.debug(f"Current robot pose: {self.env.scene['robot'].data.root_com_pos_w}")
            for env_id in env_ids:
                self._cancel_goal(env_id)
                #self.navigators[env_id].clearAllCostmaps()
                self._goal_attempts[env_id] = 0
                self._queue(env_id)

            self.current_goal_poses = self.env.path_manager.goal_pos_local.clone().cpu() # update stored goals to new ones

    def shutdown(self):
        for nav in self.navigators:
            self._executor.remove_node(nav)
            nav.destroy_node()
        self._executor.shutdown()



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
                # `ros2 lifecycle get` prints "<label> [<id>]". Match the label on a word
                # boundary: a plain `"active" in stdout` also matches "inactive", which
                # let the caller run before Nav2 had actually activated.
                if any(line.strip().startswith("active") for line in result.stdout.splitlines()):
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

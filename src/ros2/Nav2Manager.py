import subprocess
import time

import rclpy

from functools import partial
from typing import Sequence
from loguru import logger

from cfg.CFG import ROOT_DIR

from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap
from nav2_simple_commander.robot_navigator import BasicNavigator
from navigation_env.NavigationEnv import NavEnv
from geometry_msgs.msg import PoseStamped
from rclpy.executors import SingleThreadedExecutor
from std_srvs.srv import Trigger
import torch

MAX_GOAL_ATTEMPTS = 5
"""Give up on an env after this many rejected goals; it gets a fresh one on the next episode."""

SPIN_BUDGET = 4
"""Multiplier on the per-step drain budget while a goal/cancel response is in flight."""

REQUEST_TIMEOUT_S = 3.0
"""Give up on an is_active reply after this long and ask again; responses are dropped while the graph settles."""

ASK_INTERVAL_S = 0.5
"""Gap between is_active polls of a stack that answered but is not active yet."""

DOOR_SWING_TIME_S = 0.5
"""How long the doors take to reach the angle `randomize_door_positions` drives them to on reset."""


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

    def __init__(self, env: NavEnv, ros2_dm=None):
        self.env = env
        self.ros2_dm = ros2_dm  # optional, used to drop the stale cmd_vel of an env that gets a new goal

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
        self._spin_budget = max(4, 2 * env.num_envs)
        # steps still to wait before the post-reset costmap clear, see `step`
        self._reclear_in = [0] * env.num_envs
        self._reclear_delay = max(1, round(DOOR_SWING_TIME_S / env.step_dt))

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
        budget = self._spin_budget * (SPIN_BUDGET if self._pending_responses else 1)
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

            new_goal = self.env.path_manager.goal_pos_local[env_id].cpu()
            logger.info(
                f"Env {env_id}: robot local position {robot_pos_local[env_id].tolist()}, "
                f"goal position {new_goal.tolist()}"
            )

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
        goal_handle.get_result_async().add_done_callback(partial(self._on_goal_result, env_id, goal_handle))

    def _on_goal_result(self, env_id: int, goal_handle, future):
        """Re-send the goal when the BT gives up on it, so an env is not left idle for the rest of the episode.

        """
        if self._goal_handles[env_id] is not goal_handle:
            return  # superseded by a cancel or by the next episode's goal
        try:
            status = future.result().status
        except Exception as exc:  # noqa: BLE001 - never let a result callback kill the run
            logger.warning(f"Env {env_id}: NavigateToPose result failed ({exc}).")
            status = GoalStatus.STATUS_ABORTED
        self._goal_handles[env_id] = None
        if status != GoalStatus.STATUS_SUCCEEDED:
            logger.debug(f"Env {env_id}: navigation ended with status {status}, re-queueing the goal.")
            self._queue(env_id)

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

        for env_id, remaining in enumerate(self._reclear_in):
            if remaining <= 0:
                continue
            self._reclear_in[env_id] = remaining - 1
            if remaining == 1:
                self._clear_costmaps(env_id)

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
                self._clear_costmaps(env_id)
                self._reclear_in[env_id] = self._reclear_delay
                if self.ros2_dm is not None:
                    self.ros2_dm.reset_command(env_id)
                self._goal_attempts[env_id] = 0
                self._queue(env_id)

            self.current_goal_poses = self.env.path_manager.goal_pos_local.clone().cpu() # update stored goals to new ones

    def _clear_costmaps(self, env_id: int):
        """Wipe both costmaps of one env
        """
        nav = self.navigators[env_id]
        for client in (nav.clear_costmap_global_srv, nav.clear_costmap_local_srv):
            if client.service_is_ready():
                self._track(client.call_async(ClearEntireCostmap.Request()))

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
    pose.pose.orientation.w = 1.0
    return pose
            

def robot_namespaces(num_envs: int):
    # Keep this consistent with RosDataManager + launch file naming
    return [f"robot_{i}" for i in range(num_envs)]


def wait_for_nav2_ready(ros2_dm, num_envs: int, robot_prefix: str = "robot", timeout: float = 120.0) -> None:
    """Block until every robot's Nav2 stack reports active, or raise RuntimeError on timeout.

    Keeps publishing while it waits: the simulator is the only source of `/clock` and of the
    `odom -> base_link` transform, and the costmaps cannot activate without either.
    """
    node = rclpy.create_node("nav2_readiness_probe")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    clients = {
        env_id: node.create_client(Trigger, f"/{robot_prefix}_{env_id}/lifecycle_manager_navigation/is_active")
        for env_id in range(num_envs)
    }
    pending, inflight, next_ask = set(clients), {}, dict.fromkeys(clients, 0.0)
    logger.info(f"Waiting for {num_envs} Nav2 stacks to become active")

    deadline = time.monotonic() + timeout
    try:
        while pending:
            ros2_dm.pub_ros2_data()
            time.sleep(ros2_dm.env.step_dt)
            for _ in range(num_envs + 4):
                executor.spin_once(timeout_sec=0.0)

            now = time.monotonic()
            for env_id, (future, sent_at) in list(inflight.items()):
                if future.done():
                    del inflight[env_id]
                    result = future.result()
                    if result is not None and result.success:
                        pending.discard(env_id)
                        logger.info(f"  {len(pending)} Nav2 stacks still inactive")
                    else:
                        next_ask[env_id] = now + ASK_INTERVAL_S
                elif now - sent_at > REQUEST_TIMEOUT_S:
                    # Service responses do get dropped in transit while the graph is still settling.
                    # Waiting on a reply that will never arrive would mark a healthy stack as inactive,
                    # so retire the request and ask again.
                    del inflight[env_id]
                    clients[env_id].remove_pending_request(future)
                    next_ask[env_id] = now

            if now > deadline:
                raise RuntimeError(f"Nav2 stacks did not become active within {timeout}s: {sorted(pending)}")
            for env_id in pending:
                if env_id not in inflight and now >= next_ask[env_id] and clients[env_id].service_is_ready():
                    inflight[env_id] = (clients[env_id].call_async(Trigger.Request()), now)
        logger.info("All Nav2 stacks active -- starting eval loop.")
    finally:
        executor.remove_node(node)
        node.destroy_node()


def pump_ros_data(ros2_dm, seconds: float) -> None:
    """Publish `/clock`, TF and sensor data for `seconds` of simulated time without stepping the simulator.

    Used before the Nav2 launch. 
    """
    for _ in range(max(1, int(seconds / ros2_dm.env.step_dt))):
        ros2_dm.pub_ros2_data()
        time.sleep(ros2_dm.env.step_dt)


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
    # `robot[_]` matches the same processes as `robot_` but not the shell running this command, which would
    # otherwise kill itself before reaching the second pkill
    subprocess.run("pkill -f 'robot[_]' ; pkill -f rviz2", shell=True)

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if subprocess.run("pgrep -f 'robot[_]' >/dev/null", shell=True).returncode != 0:
            return
        time.sleep(0.2)
    logger.warning("Nav2 nodes from a previous run are still alive; the new stack may clash with them.")

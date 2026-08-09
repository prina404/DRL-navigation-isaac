import random
import re

import numpy as np
import omni.usd
import torch
import warp as wp
from isaaclab.assets import RigidObjectCollection
from loguru import logger
from pxr import Gf, PhysicsSchemaTools, Usd, UsdGeom, UsdLux, UsdPhysics
from omni.physx import get_physx_scene_query_interface
from scipy.spatial import ConvexHull

from navigation_env.NavigationEnv import NavEnv


def teleport_on_reset(env: NavEnv, env_ids: torch.Tensor) -> None:
    # on reset, sample new start/goal positions,
    env.sample_task_on_reset(env_ids)
    env.teleport_robots(env_ids)
    env.manual_replan(env_ids)


def replan_global_plan(env: NavEnv, env_ids: torch.Tensor) -> None:
    env.manual_replan(env_ids)


def update_episode_counters(env: NavEnv, env_ids: torch.Tensor) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    env.episode_counter[env_ids] += 1


def _collect_door_prims(stage: Usd.Stage, env_ns: str) -> list[Usd.Prim]:
    root_path = f"{env_ns}/environment/Meshes/dynamic_objects/other"
    root_prim = stage.GetPrimAtPath(root_path)

    joint_prims = []
    for prim in Usd.PrimRange(root_prim):
        if prim.IsValid() and prim.GetTypeName() == "PhysicsRevoluteJoint":
            joint_prims.append(prim)
    return joint_prims

def door_distribution(num_envs: int, num_doors: int, p_open: float = 0.80, p_closed: float = 0.15) -> torch.Tensor:
    assert p_open + p_closed <= 1.0, "Probabilities must sum to 1 or less"

    randoms = torch.rand((num_envs, num_doors))
    is_open = randoms < p_open
    is_closed = (randoms >= p_open) & (randoms < p_open + p_closed)
    is_half = randoms >= (p_open + p_closed)

    door_positions = torch.empty((num_envs, num_doors), dtype=torch.float32)
    door_positions[is_open] = 0.85 + 0.15 * torch.rand(is_open.sum())
    door_positions[is_closed] = 0.0 + 0.15 * torch.rand(is_closed.sum())
    door_positions[is_half] = 0.15 + 0.70 * torch.rand(is_half.sum())
    return door_positions * 90.0  # scale to [0, 90] degrees


def randomize_door_positions(env: NavEnv, env_ids: torch.Tensor) -> None:
    stage = omni.usd.get_context().get_stage()

    if not hasattr(env, "_door_prim_paths"):
        env._door_prim_paths = {}

    if not hasattr(env, "num_doors"):
        env_ns = "/World/envs/env_0"
        env.num_doors = len(_collect_door_prims(stage, env_ns))

    door_positions = door_distribution(env.num_envs, env.num_doors, p_open=0.8, p_closed=0.15)

    for env_id in env_ids.cpu().numpy():
        env_ns = f"/World/envs/env_{env_id}"
        door_prims = env._door_prim_paths.get(env_id, None)
        if door_prims is None:
            door_prims = _collect_door_prims(stage, env_ns)
            env._door_prim_paths[env_id] = door_prims

        for door_id, door_prim in enumerate(door_prims):
            angle = door_positions[env_id, door_id].item()
            door_drive = UsdPhysics.DriveAPI.Apply(door_prim, UsdPhysics.Tokens.angular)
            door_drive.CreateTargetPositionAttr().Set(angle)
            door_drive.CreateTargetVelocityAttr().Set(0)


def collect_light_prims(stage: Usd.Stage, env_ns: str) -> list[Usd.Prim]:
    root_prim = stage.GetPrimAtPath(f"{env_ns}/environment/Meshes/static_objects")

    light_prims = []
    for prim in Usd.PrimRange(root_prim):
        if prim.IsValid() and prim.GetTypeName() in ("SphereLight", "RectLight"):
            light_prims.append(prim)
    return light_prims


def randomize_lights_on_off(env: NavEnv, env_ids: torch.Tensor, on_probability: float = 0.5) -> None:
    stage = omni.usd.get_context().get_stage()

    if not hasattr(env, "_light_prim_paths"):  # cache light prim paths to avoid repeated USD queries
        env._light_prim_paths = {}

    for env_id in env_ids.cpu().numpy():
        env_ns = f"/World/envs/env_{env_id}"
        light_prims = env._light_prim_paths.get(env_id, None)
        if light_prims is None:
            light_prims = collect_light_prims(stage, env_ns)
            env._light_prim_paths[env_id] = light_prims

        light_temperature = random.uniform(3500.0, 8000.0)  # simulate different types of indoor lighting
        for light_prim in light_prims:
            light = UsdLux.LightAPI.Apply(light_prim)
            light.GetColorTemperatureAttr().Set(light_temperature)
            if random.random() < on_probability:
                light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))  # turn on
            else:
                light.GetColorAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))  # turn off


def randomize_distant_light(*args, **kwargs):
    stage = omni.usd.get_context().get_stage()

    random_intensity = random.uniform(1.0, 3000.0)
    light_temperature = random.uniform(2500.0, 8000.0)  # set white point to simulate different times of day

    env_ns = "/World/envs/env_0/environment/Rendering/Lights"
    for prim in Usd.PrimRange(stage.GetPrimAtPath(env_ns)):
        if prim.IsValid() and prim.GetTypeName() in ("DistantLight", "DomeLight"):
            light = UsdLux.DistantLight(prim)
            light.GetIntensityAttr().Set(random_intensity)
            light.GetEnableColorTemperatureAttr().Set(True)
            light.GetColorTemperatureAttr().Set(light_temperature)


def _sample_env_ids_obstacles(env: NavEnv, env_ids: torch.Tensor) -> torch.Tensor:
    # sample a subset of envs to add obstacles to, based on obstacle_prob
    if not hasattr(env, "obstacle_prob"):
        env.obstacle_prob = 0.0  # default to no obstacles

    mask = torch.rand(len(env_ids), device=env.device) < env.obstacle_prob
    return env_ids[mask]


def move_obstacle_on_path(env: NavEnv, env_ids: torch.Tensor) -> None:
    obstacles: RigidObjectCollection = env.scene["path_obstacles"]

    # Hide all objects below floor plane for selected envs.
    # Note: only the position part of the pose is touched; the quaternion (indices 3:7, now in
    # (x, y, z, w) order since IsaacLab 3.0) is carried through untouched.
    state_0 = obstacles.data.body_link_pose_w.torch[env_ids].clone()  # (num_env, num_obj, 7)
    state_0[..., 2] = -1.0
    obstacles.write_body_link_pose_to_sim_index(body_poses=state_0, env_ids=env_ids)

    # Resample env_ids to determine which envs get obstacles this episode based on obstacle_prob.
    env_ids = _sample_env_ids_obstacles(env, env_ids)
    if len(env_ids) == 0:
        return  # no envs selected for obstacles this episode

    # Select exactly one obstacle per env with pairwise indexing shape (num_env, 1).
    random_obstacle_ids = torch.randint(0, obstacles.num_bodies, (len(env_ids), 1), device=env.device)

    # Pairwise gather: (num_env, 1, 7), not (num_env, 7).
    state_1 = obstacles.data.body_link_pose_w.torch[env_ids[:, None], random_obstacle_ids].clone()

    sampled_positions = env.path_manager.sample_random_obstacle_on_path(env_ids)  # (num_env, 2)
    state_1[..., :2] = (sampled_positions + env.scene.env_origins[env_ids, :2]).unsqueeze(1)
    state_1[..., 2] = 0.0

    # Pairwise scatter with matching (num_env, 1) object ids.
    obstacles.write_body_link_pose_to_sim_index(body_poses=state_1, env_ids=env_ids, body_ids=random_obstacle_ids)




CHAIR_BODY_PATTERN = "/World/envs/env_*/environment/Meshes/dynamic_objects/other/chair_*/Meshes/chair_*"
CHAIR_QUERY_PROXY_ROOT = "/World/chair_query_proxies"

_CHAIR_BODY_PATH_RE = re.compile(r"^/World/envs/env_(\d+)/.*/other/(chair_\d+)/Meshes/")


def _pose_matrix(pose: np.ndarray) -> Gf.Matrix4d:
    """Transform of a PhysX pose, stored as (x, y, z) plus a quaternion in (x, y, z, w) order."""
    matrix = Gf.Matrix4d().SetRotate(Gf.Quatd(float(pose[6]), float(pose[3]), float(pose[4]), float(pose[5])))
    matrix.SetTranslateOnly(Gf.Vec3d(float(pose[0]), float(pose[1]), float(pose[2])))
    return matrix


def _collision_points_in_actor_frame(body_prim: Usd.Prim, pose: np.ndarray) -> np.ndarray:
    """Every collision mesh vertex of a chair, expressed in its PhysX actor frame.

    The vertices are authored in centimetres with the unit scale sitting on an ancestor of the rigid
    body prim, so they are gathered in world space, which applies that scale, and then mapped back.
    """
    xform_cache = UsdGeom.XformCache()
    world_to_actor = np.asarray(_pose_matrix(pose).GetInverse(), dtype=np.float64)

    points = []
    for prim in Usd.PrimRange(body_prim):
        if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
            continue
        local = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not local:
            continue
        # USD uses the row vector convention, so transforms are applied on the right
        to_world = np.asarray(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        world = np.asarray(local, dtype=np.float64) @ to_world[:3, :3] + to_world[3, :3]
        points.append(world @ world_to_actor[:3, :3] + world_to_actor[3, :3])

    if not points:
        raise RuntimeError(f"No collision meshes under chair body {body_prim.GetPath()}")
    return np.concatenate(points)


def _setup_chair_randomization(env: NavEnv) -> None:
    """Cache the chair rigid bodies, their authored poses, and one collision proxy per chair.

    Two behaviours of the running simulation force this setup, both measured rather than assumed:

    * Authoring ``xformOp:translate`` does **not** move a chair. With ``use_fabric=True`` PhysX owns
      the pose of a rigid body, so poses have to be written through the rigid body view.
    * The ``omni.physx`` scene query answers from the stage as it was parsed at load time and never
      follows the simulation, so a chair's own collider is stuck at its authored pose and cannot
      serve as the query shape. Each chair instead gets an invisible, physics free mesh carrying its
      exact convex hull, which ``overlap_mesh`` evaluates at whatever transform we author on it.
    """
    stage = omni.usd.get_context().get_stage()
    view = env.sim.physics_sim_view.create_rigid_body_view(CHAIR_BODY_PATTERN)
    if view is None or view._backend is None or view.count == 0:
        raise RuntimeError(f"no chair rigid bodies matched '{CHAIR_BODY_PATTERN}'")

    env._chair_view = view
    env._chair_home = wp.to_torch(view.get_transforms()).view(-1, 7).clone().cpu().numpy()
    env._chair_rows = {}  # (env id, chair name) -> row in the rigid body view
    for row, path in enumerate(view.prim_paths):
        match = _CHAIR_BODY_PATH_RE.match(path)
        if match is not None:
            env._chair_rows[(int(match.group(1)), match.group(2))] = row

    # the chairs are clones of each other, so one proxy per chair name serves every environment
    env._chair_proxies = {}
    stage.DefinePrim(CHAIR_QUERY_PROXY_ROOT, "Scope")
    for (_, name), row in sorted(env._chair_rows.items()):
        if name in env._chair_proxies:
            continue

        points = _collision_points_in_actor_frame(stage.GetPrimAtPath(view.prim_paths[row]), env._chair_home[row])
        hull = ConvexHull(points)
        vertices = points[hull.vertices]
        renumbered = {old: new for new, old in enumerate(hull.vertices)}

        mesh = UsdGeom.Mesh.Define(stage, f"{CHAIR_QUERY_PROXY_ROOT}/{name}")
        mesh.CreatePointsAttr([Gf.Vec3f(*vertex) for vertex in vertices])
        mesh.CreateFaceVertexCountsAttr([3] * len(hull.simplices))
        mesh.CreateFaceVertexIndicesAttr([int(renumbered[i]) for simplex in hull.simplices for i in simplex])
        mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        mesh.MakeMatrixXform()

        # widest reach of the hull around the chair's origin in the ground plane, for the robot check
        world_vertices = vertices @ np.asarray(_pose_matrix(env._chair_home[row]), dtype=np.float64)[:3, :3]
        env._chair_proxies[name] = (
            mesh.GetPrim().GetAttribute("xformOp:transform"),
            PhysicsSchemaTools.encodeSdfPath(mesh.GetPath()),
            float(np.linalg.norm(world_vertices[:, :2], axis=1).max()),
        )

    # the chair hull rests on the floor, and the robot is at its stale spawn pose in the query scene
    # (candidates are kept clear of where it actually is with a distance check instead)
    robot_name = re.escape(env.scene["robot"].cfg.prim_path.rstrip("/").rsplit("/", 1)[-1])
    env._chair_ignored_re = re.compile(rf"^/World/ground|/floor_\d+(/|$)|^/World/envs/env_\d+/{robot_name}/")

    logger.info(f"Chair randomization: {len(env._chair_proxies)} chairs over {view.count} rigid bodies")


def _chair_collides(env: NavEnv, chair_name: str, env_id: int, pose: np.ndarray) -> bool:
    """Whether the chair's convex hull, placed at pose, touches geometry it has to stay clear of."""
    xform_attr, encoded_path, _ = env._chair_proxies[chair_name]
    xform_attr.Set(_pose_matrix(pose))

    own_prefix = f"/World/envs/env_{env_id}/environment/Meshes/dynamic_objects/other/{chair_name}/"
    blocked = False

    def report(hit) -> bool:
        nonlocal blocked
        if hit.collision.startswith(own_prefix) or env._chair_ignored_re.search(hit.collision):
            return True  # keep traversing
        blocked = True
        return False  # the first real hit is enough

    get_physx_scene_query_interface().overlap_mesh(*encoded_path, report, False)
    return blocked


def randomize_chair_positions(
    env: NavEnv,
    env_ids: torch.Tensor,
    position_std: float = 0.25,
    max_retries: int = 25,
    robot_clearance: float = 0.15,
) -> None:
    """Teleport every chair to a collision free Gaussian sample around its authored position.

    Args:
        position_std: standard deviation of the x/y offset, in meters.
        max_retries: samples to try per chair before falling back to the authored position.
        robot_clearance: margin kept between the chair's hull and the robot, in meters.

    Candidates are tested against the scene as it was authored, so a chair never lands inside a wall,
    a table or a door frame. Its neighbours are only known at their authored poses though, so two
    chairs can occasionally overlap once both have moved; PhysX pushes them apart in a few steps.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    if not hasattr(env, "_chair_view"):
        try:
            _setup_chair_randomization(env)
        except Exception as error:  # scenes without chairs must not break the reset
            logger.warning(f"Chair randomization disabled: {error}")
            env._chair_view = None

    if env._chair_view is None:
        return

    poses = wp.to_torch(env._chair_view.get_transforms()).view(-1, 7).clone().cpu().numpy()
    robot_xy = env.scene["robot"].data.root_pos_w.torch[:, :2].cpu().numpy()

    rows = []
    for env_id in env_ids.cpu().tolist():
        for chair_name, (_, _, hull_radius_xy) in env._chair_proxies.items():
            row = env._chair_rows.get((env_id, chair_name))
            if row is None:
                continue

            rows.append(row)
            # start from the authored pose, so a chair knocked over during the previous episode is
            # put back upright and an unlucky chair simply stays where the scene put it
            home = env._chair_home[row]
            poses[row] = home

            for offset in np.random.normal(scale=position_std, size=(max_retries, 2)):
                candidate = home.copy()
                candidate[:2] += offset  # x/y only, the chair keeps its height and its rotation
                if np.linalg.norm(candidate[:2] - robot_xy[env_id]) < hull_radius_xy + robot_clearance:
                    continue
                if not _chair_collides(env, chair_name, env_id, candidate):
                    poses[row] = candidate
                    break

    if not rows:
        return

    row_indices = wp.array(np.asarray(rows, dtype=np.int32), dtype=wp.int32, device=env.device)
    pose_buffer = torch.from_numpy(poses).to(env.device).contiguous()
    env._chair_view.set_transforms(wp.from_torch(pose_buffer), indices=row_indices)
    # a teleported chair must not keep the momentum it had when the previous episode ended
    velocities = torch.zeros((env._chair_view.count, 6), dtype=torch.float32, device=env.device)
    env._chair_view.set_velocities(wp.from_torch(velocities), indices=row_indices)

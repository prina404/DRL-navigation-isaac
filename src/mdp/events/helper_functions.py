import random
import re

import omni.usd
import torch
from isaaclab.assets import RigidObjectCollection
from pxr import Gf, Usd, UsdLux, UsdPhysics
from omni.physx import get_physx_scene_query_interface

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

def _collect_chair_prims(stage: Usd.Stage, env_ns: str) -> list[Usd.Prim]:
    root_path = f"{env_ns}/environment/Meshes/dynamic_objects/other"
    root_prim = stage.GetPrimAtPath(root_path)

    chair_prims = []
    for prim in Usd.PrimRange(root_prim):
        if prim.IsValid() and re.search(r"/other/chair_\d+[/]*$", prim.GetPath().pathString) is not None:
            chair_prims.append(prim)
    return chair_prims

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
    state_0 = obstacles.data.object_link_pose_w[env_ids].clone()  # (num_env, num_obj, 7)
    state_0[..., 2] = -1.0
    obstacles.write_object_link_pose_to_sim(state_0, env_ids)

    # Resample env_ids to determine which envs get obstacles this episode based on obstacle_prob.
    env_ids = _sample_env_ids_obstacles(env, env_ids)
    if len(env_ids) == 0:
        return  # no envs selected for obstacles this episode

    # Select exactly one obstacle per env with pairwise indexing shape (num_env, 1).
    random_obstacle_ids = torch.randint(0, obstacles.num_objects, (len(env_ids), 1), device=env.device)

    # Pairwise gather: (num_env, 1, 7), not (num_env, 7).
    state_1 = obstacles.data.object_link_pose_w[env_ids[:, None], random_obstacle_ids].clone()

    sampled_positions = env.path_manager.sample_random_obstacle_on_path(env_ids)  # (num_env, 2)
    state_1[..., :2] = (sampled_positions + env.scene.env_origins[env_ids, :2]).unsqueeze(1)
    state_1[..., 2] = 0.0

    # Pairwise scatter with matching (num_env, 1) object ids.
    obstacles.write_object_link_pose_to_sim(state_1, env_ids, random_obstacle_ids)


def randomize_chair_positions(env: NavEnv, env_ids: torch.Tensor):
    stage = omni.usd.get_context().get_stage()

    if not hasattr(env, "_chair_prim_paths"):
        env._chair_prim_paths = {}
        env._chair_start_positions = {}

    for env_id in env_ids.cpu().numpy():
        env_ns = f"/World/envs/env_{env_id}"
        chair_prims = env._chair_prim_paths.get(env_id, None)
        if chair_prims is None:
            chair_prims = _collect_chair_prims(stage, env_ns)
            env._chair_prim_paths[env_id] = chair_prims
            env._chair_start_positions[env_id] = [chair_prim.GetAttribute("xformOp:translate").Get() for chair_prim in chair_prims]

        for i, chair_prim in enumerate(chair_prims):
            start_pos = env._chair_start_positions[env_id][i]
            _rand_chair_position(chair_prim, start_pos, env_id, env)

def _collides(chair_prim: Usd.Prim, env_id: int, env: NavEnv) -> bool:
    query = get_physx_scene_query_interface()
    pass # TODO


def _rand_chair_position(chair_prim: Usd.Prim, start_pos: torch.Tensor, env_id: int, env: NavEnv):
    retries = 50
    old_pos = chair_prim.GetAttribute("xformOp:translate").Get()
    while retries > 0:
        # Sample a random position within the environment bounds
        env_origin = env.scene.env_origins[env_id, :2]
        env_size = env.scene.env_sizes[env_id, :2]
        # TODO: SAMPLING LOGIC
        #new_pos = Gf.Vec3f(rand_x, rand_y, start_pos[2])  # Keep the original Z position
        new_pos = None
        chair_prim.GetAttribute("xformOp:translate").Set(new_pos)
        # Check if the new position is valid (not colliding with walls or other objects)
        if not _collides(chair_prim, env_id, env):
            return

        # reset position and resample if collision detected    
        chair_prim.GetAttribute("xformOp:translate").Set(old_pos)  # Reset to old position
        retries -= 1
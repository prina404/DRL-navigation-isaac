import omni.usd
import torch
from pxr import Usd, UsdPhysics

from lab.MyEnv import MyEnv


def teleport_on_reset(env: MyEnv, env_ids: torch.Tensor) -> None:
    # on reset, sample new start/goal positions,
    env.sample_task_on_reset(env_ids)
    env.teleport_robots(env_ids)
    env.manual_replan(env_ids)


def replan_global_plan(env: MyEnv, env_ids: torch.Tensor) -> None:
    env.manual_replan(env_ids)


def update_episode_counters(env: MyEnv, env_ids: torch.Tensor) -> None:
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


def randomize_door_positions(env: MyEnv, env_ids: torch.Tensor) -> None:
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

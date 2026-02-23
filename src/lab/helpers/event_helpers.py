import random

import omni.usd
import torch
from isaaclab.assets.articulation.articulation import Articulation
from pxr import Usd, UsdPhysics

from lab.MyEnv import MyEnv


def teleport_on_reset(env: MyEnv, env_ids: torch.Tensor) -> None:
    # on reset, sample new start positions,
    env.teleport_robots(env_ids)
    env.compute_goals_on_reset(env_ids)

    # Reset joints to default state after teleporting root poses.
    robot: Articulation = env.scene["robot"]
    default_pos = robot.data.default_joint_pos[env_ids]
    default_vel = robot.data.default_joint_vel[env_ids]
    robot.set_joint_position_target(default_pos, None, env_ids)
    robot.write_joint_state_to_sim(default_pos, default_vel, None, env_ids)


def clear_costmaps_on_reset(env: MyEnv, env_ids: torch.Tensor) -> None:
    env._map_manager.reset_costmaps(env_ids)


def replan_global_plan(env: MyEnv, env_ids: torch.Tensor) -> None:
    env.manual_replan(env_ids)


def _collect_door_prims(stage: Usd.Stage, env_ns: str) -> list[Usd.Prim]:
    root_path = f"{env_ns}/environment/Meshes/dynamic_objects/other"
    root_prim = stage.GetPrimAtPath(root_path)

    joint_prims = []
    for prim in Usd.PrimRange(root_prim):
        if prim.IsValid() and prim.GetTypeName() == "PhysicsRevoluteJoint":
            joint_prims.append(prim)
    return joint_prims


def randomize_door_positions(env: MyEnv, env_ids: torch.Tensor) -> None:
    stage = omni.usd.get_context().get_stage()

    if not hasattr(env, "_door_prim_paths"):
        env._door_prim_paths = {}

    for env_id in env_ids.cpu().numpy():
        env_ns = f"/World/envs/env_{env_id}"
        door_prims = env._door_prim_paths.get(env_id, None)
        if door_prims is None:
            door_prims = _collect_door_prims(stage, env_ns)
            env._door_prim_paths[env_id] = door_prims

        for door_prim in door_prims:
            angle = random.uniform(0, 90)
            door_drive = UsdPhysics.DriveAPI.Apply(door_prim, UsdPhysics.Tokens.angular)
            door_drive.CreateTargetPositionAttr().Set(angle)
            door_drive.CreateTargetVelocityAttr().Set(0)

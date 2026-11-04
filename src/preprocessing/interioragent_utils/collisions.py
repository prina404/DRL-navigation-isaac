import numpy as np
from loguru import logger
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

import preprocessing.interioragent_utils.doors as doors

# Nothing on the quadruped reaches higher than this (world frame, meters), so geometry entirely
# above it can never be touched and does not need a cooked collision mesh.
QUADRUPED_REACH_HEIGHT = 0.7

# A static piece no taller than this above a door leaf's bottom edge is a saddle or a threshold, which the
# quadruped walks over anyway (meters).
DOOR_SADDLE_HEIGHT = 0.05

# Gap a door leaf has to keep from the static geometry below it (meters).
DOOR_FLOOR_CLEARANCE = 0.002

# How far a doorway piece may stick out of the footprint of its door frame (meters).
DOOR_FRAME_MARGIN = 0.05

STATIC_OBJECTS_PATH = "/Root/Meshes/static_objects"

_BBOX_PURPOSES = [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy, UsdGeom.Tokens.render]


def _world_z_of_points(mesh_prim: Usd.Prim, xform_cache: UsdGeom.XformCache) -> np.ndarray | None:
    points = UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
    if points is None or len(points) == 0:
        return None

    pts = np.asarray(points, dtype=np.float64)
    mat = np.asarray(xform_cache.GetLocalToWorldTransform(mesh_prim), dtype=np.float64)
    # USD uses the row-vector convention (world = point * matrix), so z only needs the third column
    return pts @ mat[:3, 2] + mat[3, 2]


def _world_points(prim: Usd.Prim, xform_cache: UsdGeom.XformCache) -> np.ndarray:
    """Every mesh vertex below prim, in world coordinates, shape (N, 3)."""
    points = []
    for mesh_prim in Usd.PrimRange(prim):
        if not mesh_prim.IsValid() or not mesh_prim.IsA(UsdGeom.Mesh):
            continue
        local = UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
        if local is None or len(local) == 0:
            continue
        mat = np.asarray(xform_cache.GetLocalToWorldTransform(mesh_prim), dtype=np.float64)
        points.append(np.asarray(local, dtype=np.float64) @ mat[:3, :3] + mat[3, :3])
    return np.concatenate(points) if points else np.empty((0, 3))


def _is_entirely_above(
    prim: Usd.Prim,
    height: float,
    bbox_cache: UsdGeom.BBoxCache,
    xform_cache: UsdGeom.XformCache,
) -> bool:
    """Whether every vertex below prim sits strictly above height, in world coordinates."""
    found_geometry = False
    for mesh_prim in Usd.PrimRange(prim):
        if not mesh_prim.IsValid() or not mesh_prim.IsA(UsdGeom.Mesh):
            continue

        # cheap pre-filter: the aligned world bound never overestimates the lowest vertex, so a bound
        # that already sits above the threshold means the mesh does too, without touching its points
        aligned = bbox_cache.ComputeWorldBound(mesh_prim).ComputeAlignedRange()
        if not aligned.IsEmpty() and aligned.GetMin()[2] > height:
            found_geometry = True
            continue

        world_z = _world_z_of_points(mesh_prim, xform_cache)
        if world_z is None:
            continue

        found_geometry = True
        if world_z.min() <= height:
            return False

    return found_geometry


def _is_part_of_rigid_body(prim: Usd.Prim) -> bool:
    """Rigid bodies need their colliders to stay in place, otherwise they fall through the scene."""
    current = prim
    while current.IsValid() and not current.IsPseudoRoot():
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            return True
        current = current.GetParent()
    return False


def _disable_collision(prim: Usd.Prim):
    for api in (
        PhysxSchema.PhysxMeshMergeCollisionAPI,
        PhysxSchema.PhysxCollisionAPI,
        UsdPhysics.MeshCollisionAPI,
        UsdPhysics.CollisionAPI,
    ):
        if prim.HasAPI(api):
            prim.RemoveAPI(api)

    if prim.HasAPI(UsdPhysics.CollisionAPI):
        # the schema is applied by a layer this edit target cannot remove it from, so at least keep
        # the collider itself switched off
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
        logger.debug(f"Could not remove CollisionAPI from {prim.GetPath()}, disabled the collider instead.")


def get_static_root(root: Usd.Prim) -> Usd.Prim:
    static_root = root.GetStage().GetPrimAtPath(STATIC_OBJECTS_PATH)
    if not static_root.IsValid():
        logger.warning(f"Could not find {STATIC_OBJECTS_PATH}, pruning collisions over the whole stage instead.")
        return root
    return static_root


def prune_unreachable_collisions(root: Usd.Prim, height: float = QUADRUPED_REACH_HEIGHT) -> list[Usd.Prim]:
    """Remove the collision APIs of every static object whose geometry lies entirely above height.

    Ceilings, ceiling lights, wall cabinets and similar props are out of reach for a ground robot, so
    cooking their collision meshes only costs simulation startup time and memory.
    """
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=_BBOX_PURPOSES, useExtentsHint=False)
    xform_cache = UsdGeom.XformCache()

    pruned = []
    for prim in Usd.PrimRange(get_static_root(root)):
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue

        if _is_part_of_rigid_body(prim):
            continue

        if _is_entirely_above(prim, height, bbox_cache, xform_cache):
            _disable_collision(prim)
            pruned.append(prim)

    return pruned


def release_door_saddles(root: Usd.Prim, dynamic_doors: list[Usd.Prim]) -> list[Usd.Prim]:
    """Remove the collision APIs of the static doorway pieces that a dynamic door leaf is authored into.

    Some scenes lay a raised saddle across a doorway as a separate floor piece. The leaf of door_0004 in kujiale_0031
    starts 1 mm inside one: PhysX pushes the leaf out against its hinge, and the friction of that contact pins the door
    shut whatever its drive target. Only pieces within the footprint of the door frame are considered, so the floor of
    a room or a wall is never touched, and of those only the low ones are released; a taller one is reported instead.
    """
    xform_cache = UsdGeom.XformCache()
    colliders = []  # (prim, lower corner, upper corner) of every static collider
    for prim in Usd.PrimRange(get_static_root(root)):
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI) or _is_part_of_rigid_body(prim):
            continue
        points = _world_points(prim, xform_cache)
        if len(points):
            colliders.append((prim, points.min(axis=0), points.max(axis=0)))

    released = []
    for door in dynamic_doors:
        frame = next((prim for prim in Usd.PrimRange(door) if doors.is_door_frame(prim)), None)
        if frame is None:
            continue
        frame_points = _world_points(frame, xform_cache)
        frame_lo, frame_hi = frame_points.min(axis=0) - DOOR_FRAME_MARGIN, frame_points.max(axis=0) + DOOR_FRAME_MARGIN

        for leaf in (prim for prim in Usd.PrimRange(door) if doors.is_door_body(prim)):
            leaf_points = _world_points(leaf, xform_cache)
            leaf_lo, leaf_hi = leaf_points.min(axis=0), leaf_points.max(axis=0)
            for prim, lo, hi in colliders:
                in_doorway = (lo[:2] >= frame_lo[:2]).all() and (hi[:2] <= frame_hi[:2]).all()
                overlaps_leaf = (lo[:2] < leaf_hi[:2]).all() and (hi[:2] > leaf_lo[:2]).all() and lo[2] < leaf_hi[2]
                if not in_doorway or not overlaps_leaf or hi[2] <= leaf_lo[2] - DOOR_FLOOR_CLEARANCE or prim in released:
                    continue
                # only the top counts: the saddle of kujiale_0031 is a flat quad, 1 mm above the leaf's bottom edge
                if hi[2] > leaf_lo[2] + DOOR_SADDLE_HEIGHT:
                    logger.warning(f"{prim.GetPath()} ({lo[2]:.3f} to {hi[2]:.3f} m) overlaps {leaf.GetPath()}, left as is")
                    continue
                _disable_collision(prim)
                released.append(prim)
                logger.info(
                    f"Released {prim.GetPath()}: top at {hi[2]:.4f} m, under {leaf.GetPath()} whose bottom is at {leaf_lo[2]:.4f} m"
                )

    return released

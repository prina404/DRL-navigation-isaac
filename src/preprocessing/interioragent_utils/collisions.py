import numpy as np
from loguru import logger
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

# Nothing on the quadruped reaches higher than this (world frame, meters), so geometry entirely
# above it can never be touched and does not need a cooked collision mesh.
QUADRUPED_REACH_HEIGHT = 0.7

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

import re
from pxr import Gf, Usd,  UsdPhysics
import preprocessing.interioragent_utils.physics as physics

def is_chair_xform(prim: Usd.Prim) -> bool:
    return re.search(r"/chair_\d+[/]*$", prim.GetPath().pathString) is not None

def is_chair_root_mesh_xform(prim: Usd.Prim) -> bool:
    return re.search(r"/Meshes/chair_\d+[/]*$", prim.GetPath().pathString) is not None

def set_chair_physics(root: Usd.Prim):
    dynamic_chairs = []
    for prim in Usd.PrimRange(root):
        if not prim.IsValid():
            continue

        # if is_chair_root_mesh_xform(prim):
        #     physics.set_rigid_body_API(prim, False)  # disable rigid body API on root xform, making it static

        if is_chair_root_mesh_xform(prim):
            physics.set_rigid_body_API(prim, True, kinematic=False)  # body is dynamic
            physics.set_mesh_merge_collision(prim)
            UsdPhysics.CollisionAPI.Apply(prim)
            mc = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mc.GetApproximationAttr().Set(UsdPhysics.Tokens.convexHull)

            dynamic_chairs.append(prim)

    return dynamic_chairs
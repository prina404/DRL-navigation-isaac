from isaaclab.app import AppLauncher

simulation_app = AppLauncher({"headless": True,}).app

from pxr import Usd, UsdPhysics, Sdf, UsdGeom, PhysxSchema, Gf
import re
from cfg.CFG import INTERIOR_AGENT_DIR
from loguru import logger
import os
import argparse
parser = argparse.ArgumentParser(description="Preprocess USD for InteriorAgent.")
parser.add_argument("--file", type=str, default=None, help="Path to the USD file to preprocess. If not provided, will preprocess all USDs in the INTERIOR_AGENT_DIR.")
parser.add_argument("--all", action="store_true", default=False, help="Preprocess all USD files in the INTERIOR_AGENT_DIR.")

def set_rigid_body_API(prim: Usd.Prim, enabled: bool, kinematic: bool = False):
    if not enabled:
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        return
    prim.ApplyAPI(UsdPhysics.RigidBodyAPI)
    rb = UsdPhysics.RigidBodyAPI(prim)
    attr = rb.CreateRigidBodyEnabledAttr()
    attr.Set(True)
    k_attr = rb.CreateKinematicEnabledAttr()
    k_attr.Set(kinematic)

def set_stage_static(root: Usd.Prim):
    for prim in Usd.PrimRange(root):
        if not prim.IsValid() or not prim.IsA(UsdGeom.Xform):
            continue

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            set_rigid_body_API(prim, False)


def set_mesh_merge_collision(prim: Usd.Prim):
    meshMergeCollision = PhysxSchema.PhysxMeshMergeCollisionAPI.Apply(prim)
    meshMergeCollection = meshMergeCollision.GetCollisionMeshesCollectionAPI()
    for name in prim.GetAllChildrenNames():
        meshMergeCollection.GetIncludesRel().AddTarget(name)

def is_door_root_xform(prim: Usd.Prim) -> bool:
    return re.search(r"/Meshes/door_\d+[/]*$", prim.GetPath().pathString) is not None

def is_door_frame(prim: Usd.Prim) -> bool:
    return re.search(r"/door_\d+/group_0000[/]*$", prim.GetPath().pathString) is not None

def is_door_body(prim: Usd.Prim) -> bool:
    return re.search(r"/door_\d+/group_0001[/]*$", prim.GetPath().pathString) is not None

def is_lower_hinge(prim: Usd.Prim) -> bool:
    return re.search(r"/door_\d+/physics_constraint_0001[/]*$", prim.GetPath().pathString) is not None

def _local_translation(prim: Usd.Prim) -> Gf.Vec3d:
    local_xform = UsdGeom.Xformable(prim).GetLocalTransformation()
    if isinstance(local_xform, tuple):
        local_xform = local_xform[0]
    return local_xform.ExtractTranslation()

def _matrix_to_quatf(matrix: Gf.Matrix4d) -> Gf.Quatf:
    quat = matrix.ExtractRotation().GetQuat()
    imag = quat.GetImaginary()
    return Gf.Quatf(quat.GetReal(), imag[0], imag[1], imag[2])

def create_hinge_drive(joint_prim: Usd.Prim, target_angle_deg: float):
    drive = UsdPhysics.DriveAPI.Apply(joint_prim, UsdPhysics.Tokens.angular)
    drive.CreateTargetVelocityAttr().Set(0.0)
    drive.CreateDampingAttr().Set(200.0)
    drive.CreateStiffnessAttr().Set(1000.0)
    drive.CreateMaxForceAttr().Set(1e6)
    drive.CreateTargetPositionAttr().Set(target_angle_deg)

def set_hinge_joint_state(joint_prim: Usd.Prim, angle_deg: float):
    joint_state = PhysxSchema.PhysxJointStateAPI.Apply(joint_prim, UsdPhysics.Tokens.angular)
    joint_state.CreatePositionAttr().Set(angle_deg)
    joint_state.CreateVelocityAttr().Set(0.0)

def create_hinge_constraint(hinge_xform: Usd.Prim, frame_prim: Usd.Prim, body_prim: Usd.Prim) -> Usd.Prim:
    # TODO: joint position has to be inherited from hinge_xform
    joint_prim = hinge_xform.GetStage().DefinePrim(hinge_xform.GetPath().AppendChild("hinge_constraint"), "PhysicsRevoluteJoint")

    constraint_api = UsdPhysics.RevoluteJoint(joint_prim)

    constraint_api.CreateBody0Rel().AddTarget(frame_prim.GetPath())
    constraint_api.CreateBody1Rel().AddTarget(body_prim.GetPath())

    hinge_t = _local_translation(hinge_xform)
    body_t = _local_translation(body_prim)

    local_pos0 = Gf.Vec3f(hinge_t[0], hinge_t[1], hinge_t[2])
    local_pos1 = Gf.Vec3f(hinge_t[0] - body_t[0], hinge_t[1] - body_t[1], hinge_t[2] - body_t[2])

    xform_cache = UsdGeom.XformCache()
    hinge_world = xform_cache.GetLocalToWorldTransform(hinge_xform)
    frame_world = xform_cache.GetLocalToWorldTransform(frame_prim)
    body_world = xform_cache.GetLocalToWorldTransform(body_prim)

    joint_in_frame = frame_world.GetInverse() * hinge_world
    joint_in_body = body_world.GetInverse() * hinge_world

    local_rot0 = _matrix_to_quatf(joint_in_frame)
    local_rot1 = _matrix_to_quatf(joint_in_body)

    constraint_api.CreateLocalPos0Attr().Set(local_pos0)
    constraint_api.CreateLocalPos1Attr().Set(local_pos1)
    constraint_api.CreateLocalRot0Attr().Set(local_rot0)
    constraint_api.CreateLocalRot1Attr().Set(local_rot1)

    # set angular limit for joint
    constraint_api.CreateLowerLimitAttr().Set(0.0)
    constraint_api.CreateUpperLimitAttr().Set(90.0)
    return joint_prim

def set_door_physics(root: Usd.Prim):
    for prim in Usd.PrimRange(root):
        if not prim.IsValid() or not is_door_root_xform(prim):
            continue

        door_frame = None 
        door_body = None
        hinge_xform = None
        door_root = prim
        set_rigid_body_API(door_root, False)  # root xform should not have rigid body API
        for child_prim in Usd.PrimRange(door_root):
            if is_door_frame(child_prim):
                set_rigid_body_API(child_prim, True, kinematic=True)  
                UsdPhysics.CollisionAPI.Apply(child_prim)
                mc = UsdPhysics.MeshCollisionAPI.Apply(child_prim)  
                mc.GetApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)
                last_frame_mesh = child_prim.GetAllChildren()[-1]
                last_frame_mesh.SetActive(False)  # last mesh in group is floor, and should not have collisions
                door_frame = child_prim
            elif is_door_body(child_prim):
                child_prim.GetAttribute("visibility").Set("inherited")  # ensure body is visible
                set_rigid_body_API(child_prim, True, kinematic=False)
                set_mesh_merge_collision(child_prim)
                UsdPhysics.CollisionAPI.Apply(child_prim) 
                mc = UsdPhysics.MeshCollisionAPI.Apply(child_prim)  
                mc.GetApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
                door_body = child_prim
            elif is_lower_hinge(child_prim):
                hinge_xform = child_prim
        
        joint_prim = create_hinge_constraint(hinge_xform, door_frame, door_body)
        create_hinge_drive(joint_prim, 45)
            



def main(scene_dir: str, usd_path: str):
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise RuntimeError(f"Failed to open USD: {usd_path}")
    root = stage.GetPseudoRoot()
    
    ## Set all stage geometry to static by disabling rigid body API
    set_stage_static(root)

    # ## Enable door physics by setting up rigid body and hinge constraint APIs on door prims
    set_door_physics(root)

    ## Save preprocessed USD
    src_dir, src_name = os.path.split(usd_path)
    base, ext = os.path.splitext(src_name)
    dst_path = os.path.join(src_dir, f"{base}_baked{ext}")
    stage.GetRootLayer().Export(dst_path)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.file:
        usd_path = args.file
        scene_dir = os.path.dirname(usd_path)
        main(scene_dir, usd_path)
    else:
        for directory in os.listdir(INTERIOR_AGENT_DIR):
            if directory.startswith("kujiale_"):
                scene_dir = os.path.join(INTERIOR_AGENT_DIR, directory)
                usd_path = os.path.join(scene_dir, f"{directory}.usda")
                if not os.path.exists(usd_path):
                    logger.error(f"Could not find expected USD file: {usd_path}")
                    continue
            
            main(scene_dir, usd_path)

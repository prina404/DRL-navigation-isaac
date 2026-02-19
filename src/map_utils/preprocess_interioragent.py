from isaaclab.app import AppLauncher

simulation_app = AppLauncher({
    "headless": True,
}).app

from pxr import Usd, UsdPhysics, Sdf, UsdGeom, PhysxSchema, Gf
from omni.usd.commands import MovePrimCommand, MovePrimsCommand
import omni.usd
import re
from cfg.CFG import INTERIOR_AGENT_DIR, CFG_DIR
from loguru import logger
import os
import yaml
import argparse
import isaaclab.sim as sim_utils

parser = argparse.ArgumentParser(description="Preprocess USD for InteriorAgent.")
parser.add_argument(
    "--file",
    type=str,
    default=None,
    help="Path to the USD file to preprocess. If not provided, will preprocess all USDs in the INTERIOR_AGENT_DIR.",
)
parser.add_argument(
    "--all", action="store_true", default=False, help="Preprocess all USD files in the INTERIOR_AGENT_DIR."
)


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
    joint_prim = hinge_xform.GetStage().DefinePrim(
        hinge_xform.GetPath().AppendChild("hinge_constraint"), "PhysicsRevoluteJoint"
    )

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


def is_door_root_xform(prim: Usd.Prim) -> bool:
    return re.search(r"/other/door_\d+[/]*$", prim.GetPath().pathString) is not None


def is_door_root_mesh_xform(prim: Usd.Prim) -> bool:
    return re.search(r"/Meshes/door_\d+[/]*$", prim.GetPath().pathString) is not None


def is_door_frame(prim: Usd.Prim) -> bool:
    return re.search(r"/door_\d+/group_0000[/]*$", prim.GetPath().pathString) is not None


def is_door_body(prim: Usd.Prim) -> bool:
    return re.search(r"/door_\d+/group_000[1-9][/]*$", prim.GetPath().pathString) is not None


def is_hinge(prim: Usd.Prim) -> bool:
    return re.search(r"/door_\d+/physics_constraint_000[0-9][/]*$", prim.GetPath().pathString) is not None


def load_door_cfg(env_name: str = None) -> dict:
    with open(CFG_DIR / "interioragent_doors.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)

    if env_name not in cfg_dict:
        raise RuntimeError(f"No door configuration found for {env_name} in interioragent_doors.yaml.")

    cfg = cfg_dict[env_name]
    assert "entrance" in cfg, "Door config must specify entrance group index."
    if "normal_doors" not in cfg:
        logger.warning(
            "No normal door configuration found in interioragent_doors.yaml, no dynamic door will be set up."
        )
        cfg["normal_doors"] = []
    if "disabled_doors" not in cfg:
        logger.warning("No disabled door configuration found in interioragent_doors.yaml, no doors will be disabled.")
        cfg["disabled_doors"] = []
    return cfg


def _process_disabled_doors(root: Usd.Prim):
    for prim in Usd.PrimRange(root):
        if not prim.IsValid() or not is_door_root_mesh_xform(prim):
            continue
        # prim is root door xform under /Meshes
        set_rigid_body_API(prim, False)  # disable rigid body API on root xform, making it static
        prim.SetActive(False)  # disable root xform, making it invisible and non-collidable


def _process_normal_doors(root: Usd.Prim):
    # under root xform I may have multiple groups and constraints if the door has multiple panels,
    # so I collect them and set up a joint from frame to body for each group
    door_frame = None
    constraints, bodies = [], []
    for prim in Usd.PrimRange(root):
        if not prim.IsValid():
            continue
        if is_door_root_mesh_xform(prim):
            set_rigid_body_API(prim, False)  # disable rigid body API on root xform, making it static

        elif is_door_frame(prim):
            set_rigid_body_API(prim, True, kinematic=True)  # frame is static
            UsdPhysics.CollisionAPI.Apply(prim)
            mc = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mc.GetApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)
            last_frame_mesh = prim.GetAllChildren()[-1]
            last_frame_mesh.SetActive(False)  # last mesh in group is floor, and should not have collisions
            door_frame = prim

        elif is_door_body(prim):
            prim.GetAttribute("visibility").Set("inherited")  # ensure body is visible
            set_rigid_body_API(prim, True, kinematic=False)  # body is dynamic
            set_mesh_merge_collision(prim)
            UsdPhysics.CollisionAPI.Apply(prim)
            mc = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mc.GetApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
            bodies.append(prim)

        elif is_hinge(prim):
            constraints.append(prim)

    for hinge, body in zip(constraints, bodies):
        joint_prim = create_hinge_constraint(hinge, door_frame, body)
        create_hinge_drive(joint_prim, 90)

def _process_entrance_door(root: Usd.Prim):
    # I can leave everything as static, but I have to enable RigidBodyAPI on the door body, to have collisions
    for prim in Usd.PrimRange(root):
        if is_door_root_mesh_xform(prim):
            set_rigid_body_API(prim, False)  # disable rigid body API on root xform, to avoid nested api errors

        if is_door_body(prim):
            set_rigid_body_API(prim, True, kinematic=True)  
            set_mesh_merge_collision(prim)
            UsdPhysics.CollisionAPI.Apply(prim)
            mc = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mc.GetApproximationAttr().Set(UsdPhysics.Tokens.convexHull)

def set_door_physics(root: Usd.Prim, door_cfg: dict) -> list[Usd.Prim]:
    dynamic_doors = []
    for prim in Usd.PrimRange(root):
        if not prim.IsValid() or not is_door_root_xform(prim):
            continue
        # prim is a door root xform
        door_name = prim.GetName()
        if door_name in door_cfg["normal_doors"]:
            _process_normal_doors(prim)
            dynamic_doors.append(prim)

        elif door_name in door_cfg["disabled_doors"]:
            _process_disabled_doors(prim)

        elif door_name in door_cfg["entrance"]:
            _process_entrance_door(prim)

        else:
            logger.warning(
                f"Found door {door_name} in USD but it is not specified in the config. It will be left as static by"
                " default."
            )

    return dynamic_doors


def init_static_dynamic_folders(
    usd_path: str,
    door_cfg: dict,
    root_path: str = "/Root/Meshes",
    static_folder_name: str = "static_objects",
    dynamic_folder_name: str = "dynamic_objects",
):
    # I need to init an omni.usd context + stage in order to use the MovePrimsCommand.
    context = omni.usd.get_context()
    context.open_stage(usd_path)
    stage = context.get_stage()
    if not stage:
        raise RuntimeError(f"Failed to open USD: {usd_path}")

    # Create two subfolders under /Root/Meshes. This is needed to differentiate lidar collisions in an efficient manner
    mesh_prim = stage.GetPrimAtPath(root_path)

    static_path = f"{root_path}/{static_folder_name}"
    dynamic_path = f"{root_path}/{dynamic_folder_name}"

    static_folder:Usd.Prim = stage.DefinePrim(static_path, "Xform")
    dynamic_folder = stage.DefinePrim(dynamic_path, "Xform")

    sim_utils.standardize_xform_ops(static_folder) # needed for isaaclab xform validation
    sim_utils.standardize_xform_ops(dynamic_folder)

    # Move all prims into static folder by default
    for child in mesh_prim.GetChildren():
        if child.GetName() in [static_folder_name, dynamic_folder_name]:
            continue
        src = child.GetPath()
        dst = f"{static_path}/{child.GetName()}"
        MovePrimCommand(src, dst, destructive=False, stage_or_context=stage).do()

    dynamic_other_path = f"{dynamic_path}/other"
    stage.DefinePrim(dynamic_other_path, "Xform")
    # Move only enabled doors into dynamic folder
    prims = list(Usd.PrimRange(static_folder))
    for prim in prims:
        if prim.IsValid() and is_door_root_xform(prim) and prim.GetName() in door_cfg["normal_doors"]:
            src = prim.GetPath()
            dst = f"{dynamic_other_path}/{prim.GetName()}"
            MovePrimCommand(src, dst, destructive=False, stage_or_context=stage).do()

    # Save a temp .usda file
    src_dir, src_name = os.path.split(usd_path)
    base, ext = os.path.splitext(src_name)
    dst_path = os.path.join(src_dir, f"{base}_temp{ext}")
    stage.GetRootLayer().Export(dst_path)
    context.close_stage()
    logger.info(f"Initialized static/dynamic folders and saved temp USD to {dst_path}")
    return dst_path


def main(scene_dir: str, usd_path: str):
    env_name = os.path.basename(scene_dir)
    door_cfg = load_door_cfg(env_name)

    temp_usd_file = init_static_dynamic_folders(usd_path, door_cfg)

    # stage = Usd.Stage.Open(usd_path)
    stage = Usd.Stage.Open(temp_usd_file)  # reopen stage to ensure moves are registered
    root = stage.GetPseudoRoot()

    ## Set all geometry to static by disabling rigid body API
    set_stage_static(root)

    ## Enable door physics by setting up rigid body and hinge constraint APIs on enabled door prims
    dynamic_doors = set_door_physics(root, door_cfg)

    ## Save preprocessed USD
    src_dir, src_name = os.path.split(usd_path)
    base, ext = os.path.splitext(src_name)
    dst_path = os.path.join(src_dir, f"{base}_baked{ext}")
    stage.GetRootLayer().Export(dst_path)

    os.remove(temp_usd_file)  # clean up temp file
    logger.info(f"Preprocessed USD saved to {dst_path} with {len(dynamic_doors)} dynamic doors.")

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

import re

from loguru import logger
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

import preprocessing.interioragent_utils.physics as physics


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


def is_doorsill(prim: Usd.Prim) -> bool:
    # usually mesh_0017 is the doorsill, TODO: check if this applies to all doors
    return re.search(r"/group_0000/mesh_0017[/]*$", prim.GetPath().pathString) is not None


def _process_disabled_doors(root: Usd.Prim):
    for prim in Usd.PrimRange(root):
        if not prim.IsValid() or not is_door_root_mesh_xform(prim):  # or not is_doorsill(prim):
            continue
        # prim is root door xform under /Meshes, disable also doorsills to avoid collisions with invisible geometry
        physics.set_rigid_body_API(prim, False)  # disable rigid body API on root xform, making it static
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
            physics.set_rigid_body_API(prim, False)  # disable rigid body API on root xform, making it static

        elif is_door_frame(prim):
            physics.set_rigid_body_API(prim, True, kinematic=True)  # frame is static
            UsdPhysics.CollisionAPI.Apply(prim)
            mc = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mc.GetApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)
            last_frame_mesh = prim.GetAllChildren()[-1]
            last_frame_mesh.SetActive(False)  # last mesh in group is floor, and should not have collisions

            doorsill = [mesh for mesh in prim.GetAllChildren() if is_doorsill(mesh)]
            if len(doorsill) > 0:
                doorsill[0].SetActive(False)  # disable doorsill mesh to avoid collisions with invisible geometry

            door_frame = prim

        elif is_door_body(prim):
            prim.GetAttribute("visibility").Set("inherited")  # ensure body is visible
            for subprim in Usd.PrimRange(prim, Usd.PrimAllPrimsPredicate):
                if subprim.IsA(UsdGeom.Mesh):
                    subprim.SetActive(True)  # needed because some environments have disabled meshes

            physics.set_rigid_body_API(prim, True, kinematic=False)  # body is dynamic
            physics.set_mesh_merge_collision(prim)
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
            physics.set_rigid_body_API(prim, False)  # disable rigid body API on root xform, to avoid nested api errors

        if is_door_body(prim):
            physics.set_rigid_body_API(prim, True, kinematic=True)
            physics.set_mesh_merge_collision(prim)
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
                f"Found door {door_name} in USD but it is not specified in the config. It will be left as static by" " default."
            )

    return dynamic_doors

from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics


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
    for child in prim.GetAllChildren():
        meshMergeCollection.GetIncludesRel().AddTarget(child.GetPath())

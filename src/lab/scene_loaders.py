import os
import omni
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from pxr import Usd, UsdGeom, UsdPhysics, Sdf
import omni.replicator.core as rep
from omegaconf import DictConfig
from loguru import logger

def _disable_rigid_bodies(root_prim_path: str) -> int:
    # Disable all rigid bodies under a prim path.
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return 0

    root = stage.GetPrimAtPath(root_prim_path)
    if not root or not root.IsValid():
        return 0

    disabled = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsValid():
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue

        rb = UsdPhysics.RigidBodyAPI(prim)
        attr = rb.GetRigidBodyEnabledAttr()
        if not attr:
            attr = rb.CreateRigidBodyEnabledAttr()
        if attr.Get() is not False:
            attr.Set(False)
            disabled += 1
    for prim in Usd.PrimRange(root):
        if not prim.IsValid():
            continue

        meshCollisionAPI = UsdPhysics.MeshCollisionAPI.Apply(prim)
        meshCollisionAPI.GetApproximationAttr().Set(UsdPhysics.Tokens.meshSimplification)
        
    return disabled


def _is_under_looks(prim: Usd.Prim) -> bool:
    """Skip materials/shaders subtree; never apply physics APIs there."""
    p = str(prim.GetPath())
    return "/Looks/" in p or p.endswith("/Looks")


def _is_root_xform_under_category(prim: Usd.Prim, category_root: Usd.Prim) -> bool:
    """True if prim is an Xform and there is no other Xform between it and category_root."""
    if prim.GetTypeName() != "Xform":
        return False
    # Walk up until category_root; if we encounter another Xform, then prim isn't the root Xform.
    p = prim.GetParent()
    while p and p.IsValid() and p != category_root:
        if p.GetTypeName() == "Xform":
            return False
        p = p.GetParent()
    return True


def _iter_category_root_xforms(scene_root_path: str, categories: list[str]):
    """
    Yield root Xform prims under:
      {scene_root_path}/Meshes/{category}/...
    selecting only highest Xforms to avoid nested rigid bodies.
    """
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    for cat in categories:
        cat_root_path = f"{scene_root_path}/Meshes/{cat}"
        cat_root = stage.GetPrimAtPath(cat_root_path)
        if not cat_root or not cat_root.IsValid():
            continue

        it = iter(Usd.PrimRange(cat_root))
        for prim in it:
            if not prim.IsValid():
                continue
            if _is_under_looks(prim):
                it.PruneChildren()
                continue

            if _is_root_xform_under_category(prim, cat_root):
                yield prim
                # Don't traverse into this object once we've chosen its root
                it.PruneChildren()


def _make_root_xforms_kinematic(scene_root_path: str, categories: list[str]) -> int:
    """
    Apply kinematic rigid body only to the selected root Xform prims under given categories.
    """
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return 0

    count = 0
    for prim in _iter_category_root_xforms(scene_root_path, categories):
        # Apply/ensure RigidBodyAPI on root Xform only
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(prim)
        rb = UsdPhysics.RigidBodyAPI(prim)

        rb_enabled = rb.GetRigidBodyEnabledAttr()
        if not rb_enabled:
            rb_enabled = rb.CreateRigidBodyEnabledAttr()
        rb_enabled.Set(True)

        # Portable way: author USD physics kinematic attribute
        kin = rb.GetKinematicEnabledAttr() if hasattr(rb, "GetKinematicEnabledAttr") else None
        if not kin:
            kin = prim.GetAttribute("physics:kinematicEnabled")
        if not kin:
            kin = prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool)
        kin.Set(True)

        count += 1

    return count


def update_prims_positions(root_prim_path, variations_cfg):
    pass


def load_interiorAgent_env(cfg: DictConfig, namespace: str) -> None:
    ground_plane = rep.get.prims("/World/GroundPlane")
    with ground_plane:
        rep.modify.semantics([("class", "floor")])

    asset_folder = os.path.expandvars(cfg.usd_folder)
    asset_path = os.path.join(asset_folder, cfg.env_name, cfg.env_name + ".usda")
    kinematic_categories = list(getattr(cfg, "kinematic_categories", ["chair"]))
    for i in range(cfg.num_envs):
        SCENE_PRIM = f"{namespace}/env_{i}/{cfg.env_name}"
        prim = get_prim_at_path(SCENE_PRIM)
        if prim is None or not prim.IsValid():
            prim = define_prim(SCENE_PRIM, "Xform")
        prim.GetReferences().AddReference(asset_path)
        x = UsdGeom.XformCommonAPI(get_prim_at_path(SCENE_PRIM))
        x.SetTranslate((0.0, 0.0, 0.05))

        num_disabled = _disable_rigid_bodies(SCENE_PRIM)
        if num_disabled > 0:
            print(
                f"[physics-fix] Disabled {num_disabled} rigid bodies under {SCENE_PRIM} (treating env as static)"
            )

        # num_kin = _make_root_xforms_kinematic(SCENE_PRIM, categories=kinematic_categories)
        # if num_kin > 0:
        #     print(
        #         f"[physics] Marked {num_kin} root Xforms as kinematic under {SCENE_PRIM} for categories={kinematic_categories}"
        #     )



def load_infinigen_env(cfg: DictConfig, namespace: str) -> None:
    asset_path = os.path.expandvars(cfg.infinigen_env)
    for i in range(cfg.num_envs):
        ENV_ROOT = f"{namespace}/env_{i}"
        prim_path = f"{ENV_ROOT}/infinigen_scene"
        prim = get_prim_at_path(prim_path)
        if prim is None or not prim.IsValid():
            prim = define_prim(prim_path, "Xform")
        refs = prim.GetReferences()
        try:
            existing = refs.GetAddedOrExplicitItems()
            if any(getattr(r, "assetPath", None) == asset_path for r in existing):
                continue
        except Exception:
            pass
        refs.AddReference(asset_path)

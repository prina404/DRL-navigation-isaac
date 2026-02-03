import os
import argparse

# -------------------------------------------------
# Launch Isaac Sim headless
# -------------------------------------------------

from isaaclab.app import AppLauncher

simulation_app = AppLauncher({"headless": True,}).app

from pxr import Usd, UsdPhysics, Sdf, UsdGeom
import re


def is_under_looks(prim: Usd.Prim) -> bool:
    p = prim.GetPath().pathString
    return "/Looks/" in p or p.endswith("/Looks")


def is_invisible(prim: Usd.Prim) -> bool:
    """Returns True if prim is invisible (including inherited)."""
    if not prim or not prim.IsValid():
        return False
    if not prim.IsA(UsdGeom.Imageable):
        return False
    img = UsdGeom.Imageable(prim)
    # ComputeVisibility accounts for ancestors too
    vis = img.ComputeVisibility(Usd.TimeCode.Default())
    return vis == UsdGeom.Tokens.invisible


def set_kinematic(prim: Usd.Prim):  # TODO: rework
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)

    rb = UsdPhysics.RigidBodyAPI(prim)

    rb_enabled = rb.GetRigidBodyEnabledAttr()
    if not rb_enabled:
        rb_enabled = rb.CreateRigidBodyEnabledAttr()
    rb_enabled.Set(True)

    kin = prim.GetAttribute("physics:kinematicEnabled")
    if not kin:
        kin = prim.CreateAttribute(
            "physics:kinematicEnabled",
            Sdf.ValueTypeNames.Bool,
        )
    kin.Set(True)


def group_has_any_invisible_mesh(group_prim: Usd.Prim) -> bool:
    """True if *any* Mesh under group_prim is invisible (incl. inherited) or inactive."""
    # If the group itself is invisible, treat as having invisible meshes.
    if group_prim.IsA(UsdGeom.Imageable) and is_invisible(group_prim):
        return True

    for p in Usd.PrimRange(group_prim):
        if not p.IsValid() or p.GetTypeName() != "Mesh":
            continue

        # treat inactive Mesh as "invisible present" so we skip this group.
        if is_invisible(p) or not p.IsActive():
            return True

    return False


def apply_mesh_merge_collision_to_door_groups(
    door_root_prim: Usd.Prim,
    group_name_re: re.Pattern = re.compile(r"^group_\d+$"),
) -> int:
    """
    For each direct child under door_root_prim matching group_000x:
      - if NO invisible/inactive Mesh exists in its subtree:
          - apply PhysX MeshMergeCollision to the group
          - set convexHull approximation
    Returns number of groups modified.
    """
    if not door_root_prim or not door_root_prim.IsValid():
        return 0

    modified = 0

    for child in door_root_prim.GetChildren():
        name = child.GetName()
        if group_name_re.fullmatch(name) is None:
            continue

        # Skip groups that contain any invisible/inactive meshes.
        if group_has_any_invisible_mesh(child):
            continue
        print(f"Applying MeshMergeCollision to door group: {child.GetPath().pathString}")
        # Optional: ensure collision API is present on the group container.
        UsdPhysics.CollisionAPI.Apply(child)

        mc = UsdPhysics.MeshCollisionAPI.Apply(child)  
        mc.GetApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)

        modified += 1

    return modified


def disable_rigid_body(prim: Usd.Prim):
    rb = UsdPhysics.RigidBodyAPI(prim)
    attr = rb.GetRigidBodyEnabledAttr()
    if not attr:
        attr = rb.CreateRigidBodyEnabledAttr()
    if attr.Get() is not False:
        attr.Set(False)


def preprocess_usd(
    src_path: str,
    kinematic_categories: list[str],
):
    stage = Usd.Stage.Open(src_path)
    if not stage:
        raise RuntimeError(f"Failed to open USD: {src_path}")

    root = stage.GetPseudoRoot()

    # -- Disable all rigid bodies and invisible meshes
    disabled_rb = 0
    deactivated_invisible = 0


    for prim in Usd.PrimRange(root):
        if not prim.IsValid() or is_under_looks(prim):
            continue

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):  # Disable rigid bodyAPI, making every mesh static
            disable_rigid_body(prim)
            disabled_rb += 1

        if prim.GetTypeName() == "Mesh":  # Disable invisible meshes entirely
            if is_invisible(prim):
                prim.SetActive(False)
                deactivated_invisible += 1
    
    
    door_re = re.compile(r"\/Meshes\/door_\d+$") # root Xform for mesh groups of door prims 
    is_door_mesh = lambda p: door_re.search(p.GetPath().pathString) is not None
    for prim in Usd.PrimRange(root):
        if is_door_mesh(prim):
            apply_mesh_merge_collision_to_door_groups(prim)

    # -- Mark selected objects as kinematic

    kinematic_count = 0

    for prim in Usd.PrimRange(root):
        for cat in kinematic_categories:
            pass
        # TODO: set root xform as kinematic, then disable rigid bodies in its subtree

    # -- Save baked USD
    src_dir, src_name = os.path.split(src_path)
    base, ext = os.path.splitext(src_name)
    dst_path = os.path.join(src_dir, f"{base}_baked{ext}")

    stage.GetRootLayer().Export(dst_path)

    print("USD preprocessing complete")
    print(f"  Disabled rigid bodies : {disabled_rb}")
    print(f"  Kinematic objects     : {kinematic_count}")
    print(f"  Deactivated invisible meshes : {deactivated_invisible}")
    print(f"  Saved baked USD       : {dst_path}")

    return dst_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("usd", help="Path to source USD file")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=[],
        help="Mesh categories to mark as kinematic (e.g. 'door', 'chair')",
    )

    args = parser.parse_args()

    preprocess_usd(
        src_path=os.path.abspath(args.usd),
        kinematic_categories=args.categories,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during USD preprocessing: {e}")
    finally:
        simulation_app.close()

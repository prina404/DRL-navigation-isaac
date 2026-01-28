import os
import argparse

# -------------------------------------------------
# Launch Isaac Sim headless
# -------------------------------------------------

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(
    {
        "headless": True,
    }
).app

# -------------------------------------------------
# Imports AFTER SimulationApp
# -------------------------------------------------
from pxr import Usd, UsdPhysics, Sdf, UsdGeom, PhysxSchema


# ----------------------------
# helpers
# ----------------------------


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


def set_kinematic(prim: Usd.Prim):
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


# ----------------------------
# preprocessing logic
# ----------------------------


def preprocess_usd(
    src_path: str,
    kinematic_categories: list[str],
    scene_root: str = "/World",
):
    stage = Usd.Stage.Open(src_path)
    if not stage:
        raise RuntimeError(f"Failed to open USD: {src_path}")

    root = stage.GetPseudoRoot()

    # -------------------------------------------------
    # 1) Disable all rigid bodies + convex hull collisions
    # -------------------------------------------------
    disabled_rb = 0
    deactivated_invisible = 0


    for prim in Usd.PrimRange(root):
        if not prim.IsValid() or is_under_looks(prim):
            continue

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = UsdPhysics.RigidBodyAPI(prim)
            attr = rb.GetRigidBodyEnabledAttr()
            if not attr:
                attr = rb.CreateRigidBodyEnabledAttr()
            if attr.Get() is not False:
                attr.Set(False)
                disabled_rb += 1

        if prim.GetTypeName() == "Mesh":
            if is_invisible(prim):
                prim.SetActive(False)
                deactivated_invisible += 1
                continue

            mc = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mc.GetApproximationAttr().Set(UsdPhysics.Tokens.convexHull)

    # -------------------------------------------------
    # 2) Mark selected objects as kinematic
    # -------------------------------------------------
    kinematic_count = 0
    disabled_nested_rb = 0

    for prim in Usd.PrimRange(root):
        for cat in kinematic_categories:
            pass
        # TODO: set root xform as kinematic, then disable rigid bodies in its subtree

    # -------------------------------------------------
    # 3) Save baked USD
    # -------------------------------------------------
    src_dir, src_name = os.path.split(src_path)
    base, ext = os.path.splitext(src_name)
    dst_path = os.path.join(src_dir, f"{base}_baked{ext}")

    stage.GetRootLayer().Export(dst_path)

    print("USD preprocessing complete")
    print(f"  Disabled rigid bodies : {disabled_rb}")
    print(f"  Disabled nested RBs   : {disabled_nested_rb}")
    print(f"  Kinematic objects     : {kinematic_count}")
    print(f"  Deactivated invisible meshes : {deactivated_invisible}")
    print(f"  Saved baked USD       : {dst_path}")

    return dst_path


# ----------------------------
# CLI
# ----------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("usd", help="Path to source USD file")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=[],
        help="Mesh categories to mark as kinematic",
    )
    parser.add_argument(
        "--scene-root",
        default="/World",
        help="Scene root path",
    )

    args = parser.parse_args()

    preprocess_usd(
        src_path=os.path.abspath(args.usd),
        kinematic_categories=args.categories,
        scene_root=args.scene_root,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during USD preprocessing: {e}")
    finally:
        simulation_app.close()

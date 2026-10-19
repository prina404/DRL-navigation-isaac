from pathlib import Path

from pxr import Usd, UsdGeom

#: Subtree of the baked scene exposed by each split layer
SPLIT_LAYERS = {
    "static": "/Root/Meshes/static_objects",
    "dynamic": "/Root/Meshes/dynamic_objects",
}


def write_split_layers(baked_usd: str | Path) -> list[Path]:
    """Split a baked scene into `<name>_static.usda` and `<name>_dynamic.usda`, written next to it.

    The static geometry of a map is loaded once and shared by every robot navigating it, while each robot gets its own
    copy of the doors and chairs, so the two halves are spawned separately. Each layer only references a subtree of
    the baked usd, which stays the single source of truth: the layers are tiny and never go stale.
    """
    baked_usd = Path(baked_usd)
    map_name = baked_usd.stem.removesuffix("_baked")

    layer_paths = []
    for part, subtree in SPLIT_LAYERS.items():
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        root = stage.DefinePrim(f"/{part}_objects")
        root.GetReferences().AddReference(f"./{baked_usd.name}", subtree)
        stage.SetDefaultPrim(root)

        layer_path = baked_usd.with_name(f"{map_name}_{part}.usda")
        stage.GetRootLayer().Export(str(layer_path))
        layer_paths.append(layer_path)
    return layer_paths

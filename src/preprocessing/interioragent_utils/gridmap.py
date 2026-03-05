import os
from pathlib import Path

import numpy as np
import omni
import omni.usd
import yaml
from carb import Float2, Float3
from isaacsim.asset.gen.omap.bindings import _omap
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from loguru import logger
from PIL import Image
from pxr import Usd, UsdGeom


def compute_scene_bbox(
    root_prim: Usd.Prim,
    padding_size=0.25,
) -> tuple[Float2, Float2]:

    bbox = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.render,
        ],
        useExtentsHint=False,
    )

    wmin = Float2(float("inf"), float("inf"))
    wmax = Float2(float("-inf"), float("-inf"))

    for prim in Usd.PrimRange(root_prim):
        if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
            continue

        img = UsdGeom.Imageable(prim)
        try:  # skip invisible meshes
            if img.ComputeVisibility(Usd.TimeCode.Default()) == UsdGeom.Tokens.invisible:
                continue
        except Exception:
            pass

        try:
            bound = bbox.ComputeWorldBound(prim)
            r = bound.ComputeAlignedRange()
            mn, mx = r.GetMin(), r.GetMax()

            wmin.x = min(wmin.x, float(mn[0]))
            wmax.x = max(wmax.x, float(mx[0]))
            wmin.y = min(wmin.y, float(mn[1]))
            wmax.y = max(wmax.y, float(mx[1]))
        except Exception:
            continue

    min_local = Float2(wmin.x - padding_size, wmin.y - padding_size)
    max_local = Float2(wmax.x + padding_size, wmax.y + padding_size)

    return min_local, max_local


def _get_omap_generator(
    env_bb: tuple[Float2, Float2],
    center_coord: Float2 = Float2(0, 0),
    scan_height_range: tuple = (
        0.1,
        0.5,
    ),  # Should be set to the height range occupied by the robot
    resolution: float = 0.05,
) -> _omap.Generator:

    if scan_height_range[0] < 0.01:
        raise ValueError("Lower bound of scan_height_range must be > 0 to ensure correct map generation.")

    physx = omni.physx.get_physx_interface()
    stage_id = omni.usd.get_context().get_stage_id()

    gen = _omap.Generator(physx, stage_id)

    occ_val, free_val, unknown_val = 0, 255, 205  # ROS occupancy values
    gen.update_settings(resolution, occ_val, free_val, unknown_val)

    z_origin = scan_height_range[0]
    origin_coord = Float3(center_coord.x, center_coord.y, z_origin)
    min_bound = Float3(env_bb[0].x, env_bb[0].y, z_origin)
    max_bound = Float3(env_bb[1].x, env_bb[1].y, scan_height_range[1] - z_origin)

    # ensure map bounds correspond to the bbox even if origin is not (0.0, 0.0)
    max_bound.x -= origin_coord.x
    min_bound.x -= origin_coord.x
    max_bound.y -= origin_coord.y
    min_bound.y -= origin_coord.y

    gen.set_transform(origin_coord, min_bound, max_bound)  # set mapping volume
    return gen


def compute_and_save_map(scene_prim: Usd.Prim, map_name: str, save_folder: str | Path, env_cfg: dict) -> None:
    bbox = compute_scene_bbox(scene_prim, padding_size=0.25)

    if "map_origin" in env_cfg:
        map_origin = Float2(env_cfg["map_origin"]["x"], env_cfg["map_origin"]["y"])
    else:
        map_origin = Float2((bbox[0].x + bbox[1].x) / 2, (bbox[0].y + bbox[1].y) / 2)

    gen = _get_omap_generator(bbox, center_coord=map_origin, resolution=0.05)

    gen.generate2d()
    dims = tuple(gen.get_dimensions())  # (W, H, C)
    buf = gen.get_buffer()

    origin = gen.get_min_bound()
    map_yaml = {
        "image": f"{map_name}.png",
        "resolution": 0.05,
        "origin": [origin.x, origin.y, 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }

    # plot binary map
    img = Image.fromarray(np.array(buf, dtype=np.uint8).reshape(dims[1], dims[0]))
    img.save(Path(save_folder) / f"{map_name}.png")
    logger.info(f"Saved map image to {Path(save_folder) / f'{map_name}.png'}")

    with open(Path(save_folder) / f"{map_name}.yaml", "w") as f:
        yaml.dump(map_yaml, f)


def create_map_from_file(filepath: str | Path, env_cfg: dict):
    usd_path = os.path.abspath(str(filepath))

    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()

    add_reference_to_stage(usd_path=usd_path, prim_path="/World/environment")

    my_world.reset()
    my_world.step(render=True)  # step once to ensure all prims are loaded
    for _ in range(100):
        my_world.step(render=False)  # step a few more times to ensure doors reach their final positions

    compute_and_save_map(
        scene_prim=my_world.stage.GetPrimAtPath("/World/environment"),
        map_name=Path(usd_path).stem + "_map",
        save_folder=Path(usd_path).parent,
        env_cfg=env_cfg,
    )

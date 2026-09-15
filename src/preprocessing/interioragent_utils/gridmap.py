import os
from pathlib import Path

import isaaclab.sim as sim_utils
import numpy as np
import omni
import omni.physx
import omni.usd
import warp as wp
import yaml
from carb import Float2, Float3
from isaaclab.sim import SimulationCfg, SimulationContext
from isaacsim.asset.gen.omap.bindings import _omap
from loguru import logger
from PIL import Image
from pxr import Gf, Usd, UsdGeom, UsdPhysics


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
    stage_id = sim_utils.get_current_stage_id()

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


def _sync_simulated_poses_to_usd(sim: SimulationContext, root_path: str) -> int:
    """Author the poses reached by the simulation back onto their prims.

    IsaacLab steps physics through the tensor API, which keeps the resulting poses in the physics
    views and never writes them to USD. The occupancy generator reads the stage, so without this the
    map would be built with every door still closed at its authored pose.
    """
    view = sim.physics_manager.get_physics_sim_view()
    xform_cache = UsdGeom.XformCache()

    # read every pose before touching the stage, so no cached parent transform goes stale mid-way
    poses = []
    for prim in Usd.PrimRange(sim.stage.GetPrimAtPath(root_path)):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI) or UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get():
            continue  # static and kinematic bodies stay where they were authored

        body = view.create_rigid_body_view(prim.GetPath().pathString)
        if body is None or body.count == 0:
            continue

        transform = wp.to_torch(body.get_transforms()).detach().cpu().numpy()[0]  # (pos, quat as xyzw)
        # the hierarchy bakes a unit conversion into the parent scale, which physics poses do not carry
        scale = Gf.Transform(xform_cache.GetLocalToWorldTransform(prim)).GetScale()
        parent_to_world = xform_cache.GetLocalToWorldTransform(prim.GetParent())

        world = Gf.Transform()
        world.SetScale(scale)
        world.SetRotation(Gf.Rotation(Gf.Quatd(float(transform[6]), Gf.Vec3d(*transform[3:6].tolist()))))
        world.SetTranslation(Gf.Vec3d(*transform[:3].tolist()))

        poses.append((prim, world.GetMatrix() * parent_to_world.GetInverse()))

    for prim, local in poses:
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp().Set(local)

    return len(poses)


def create_map_from_file(filepath: str | Path, env_cfg: dict):
    usd_path = os.path.abspath(str(filepath))

    # start from a fresh stage: preprocessing swaps the stage of the usd context for every scene, so a
    # simulation context left over from a previous scene would still point at the wrong one
    SimulationContext.clear_instance()
    sim_utils.create_new_stage()
    # the generator probes the scene with physics queries, and fabric would only mirror the physics
    # state for rendering, which this pass never does
    sim = SimulationContext(SimulationCfg(use_fabric=False, enable_scene_query_support=True))

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/groundPlane", ground_cfg)

    scene_cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
    scene_cfg.func("/World/environment", scene_cfg)

    sim.reset()
    sim.step(render=True)  # step once to ensure all prims are loaded
    for _ in range(100):
        sim.step(render=False)  # step a few more times to ensure doors reach their final positions

    synced = _sync_simulated_poses_to_usd(sim, "/World/environment")
    logger.debug(f"Wrote back the simulated poses of {synced} dynamic bodies before mapping.")

    # the generator maps the collision scene physx loaded from usd, which only picks up the poses
    # written above once it re-reads the stage
    omni.physx.get_physx_interface().force_load_physics_from_usd()

    compute_and_save_map(
        scene_prim=sim.stage.GetPrimAtPath("/World/environment"),
        map_name=Path(usd_path).stem + "_map",
        save_folder=Path(usd_path).parent,
        env_cfg=env_cfg,
    )

    # leave no simulation attached to the stage, the next scene reopens the usd context from scratch
    SimulationContext.clear_instance()

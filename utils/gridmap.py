import time
from loguru import logger
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
import omni
from isaacsim.asset.gen.omap.bindings import _omap
from pxr import UsdGeom, Usd
import torch
import scipy.ndimage as sp
import pyastar2d

def _compute_env_bbox(
    env: ManagerBasedRLEnv,
    env_id=0,
    padding_size=0.25,
    z_offset=0.5,
) -> tuple:
    """Store once the scene BBOX in local env coordinates."""
    if hasattr(env.scene, "bbox"):
        return env.scene.bbox
    
    if not hasattr(env.scene, "environment_prim_name"):
        raise ValueError("environment_prim_name not set in env.scene")

    stage = omni.usd.get_context().get_stage()
    env_origin = env.scene.env_origins[env_id]
    ox, oy = float(env_origin[0]), float(env_origin[1])
    env_prim_path = f"/World/envs/env_{env_id}"
    env_prim = stage.GetPrimAtPath(env_prim_path)
    bbox = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy, UsdGeom.Tokens.render],
        useExtentsHint=False,
    )
    wmin = [float("inf"), float("inf"), float("inf")]
    wmax = [float("-inf"), float("-inf"), float("-inf")]

    for prim in Usd.PrimRange(env_prim):
        if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
            continue

        img = UsdGeom.Imageable(prim)
        try:
            if img.ComputeVisibility(Usd.TimeCode.Default()) == UsdGeom.Tokens.invisible:
                continue
        except Exception:
            pass

        try:
            bound = bbox.ComputeWorldBound(prim)
            r = bound.ComputeAlignedRange()
            mn, mx = r.GetMin(), r.GetMax()
            for i in range(3):
                wmin[i] = min(wmin[i], float(mn[i]))
                wmax[i] = max(wmax[i], float(mx[i]))
        except Exception:
            continue

    min_local = (float(wmin[0] - ox) - padding_size, float(wmin[1] - oy) - padding_size, z_offset)
    max_local = (float(wmax[0] - ox) + padding_size, float(wmax[1] - oy) + padding_size, z_offset)
    setattr(env.scene, "bbox", (min_local, max_local))
    logger.info(f"Computed env_{env_id} bbox min: {min_local}, max: {max_local} and cached in env.scene.bbox")
    return min_local, max_local


def _get_omap_generator(env: ManagerBasedRLEnv, resolution = 0.05) -> _omap.Generator:
    """Create or get existing omap Generator instance."""
    if hasattr(env.scene, "_omap_generator"):
        return getattr(env.scene, "_omap_generator")

    physx = omni.physx.get_physx_interface()
    stage_id = omni.usd.get_context().get_stage_id()

    gen = _omap.Generator(physx, stage_id)

    occ_val, free_val, unknown_val = 0, 255, 205
    gen.update_settings(resolution, occ_val, free_val, unknown_val)

    setattr(env.scene, "_omap_generator", gen)
    return gen


def generate_ogm_on_reset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    map_resolution: float = 0.05,
) -> None:
    """callback function to generate occupancy grid maps for specified env ids and store them in env.scene.occupancy_maps"""
    gen = _get_omap_generator(env, map_resolution)
    if not hasattr(env.scene, "occupancy_maps"):
        setattr(env.scene, "occupancy_maps", [None] * env.num_envs)
    occupancy_maps = getattr(env.scene, "occupancy_maps")
    origin_coords = env.scene.env_origins  # [N, 3] tensor

    min_point, max_point = _compute_env_bbox(env, env_id=0)
    logger.info(f"Computed env bbox min: {min_point}, max: {max_point}")
    for i in env_ids.cpu().tolist():
        env_origin = origin_coords[i]
        ox, oy = float(env_origin[0]), float(env_origin[1])

        min_world = (min_point[0] + ox, min_point[1] + oy, min_point[2])
        max_world = (max_point[0] + ox, max_point[1] + oy, max_point[2])

        gen.set_transform(
            (ox, oy, 0.0),  # origin of the map in world coordinates
            min_world,  # min-max corners in world coordinates for i-th env
            max_world,
        )
        s = time.time()
        gen.generate2d()
        logger.info(f"Occupancy map generation for env {i} took {time.time() - s:.5f} seconds")
        dims = tuple(gen.get_dimensions())  # (W, H, C)
        buf_t = torch.tensor(gen.get_buffer(), dtype=torch.uint8, device="cpu")

        occupancy_maps[i] = {
            "dims": dims,
            "buffer": buf_t,
            "origin": (ox, oy),
            "bounds_world": (min_world, max_world),
            "resolution": map_resolution,
        }
        ## debug: save OGM as image
        from PIL import Image

        img = Image.fromarray(buf_t.cpu().numpy().reshape(dims[1], dims[0]))
        img.save(f"env_{i}_omap.png")
        logger.info(f"Occupancy map for env {i} saved as env_{i}_omap.png")

        # save also dummy a* path for testing
        start_coords = np.array([30, 60], dtype=np.int32)
        goal_coords = np.array([57, 145], dtype=np.int32)
        a_star_path = _compute_global_plan(occupancy_maps[i], start_coords, goal_coords)


    setattr(env.scene, "occupancy_maps", occupancy_maps)

# Inspired by Peter Corke's Robotics Toolbox
# https://github.com/petercorke/robotics-toolbox-python/blob/master/roboticstoolbox/mobile/OccGrid.py
def _inflate_img(img: np.ndarray, radius: float, resolution: float) -> np.ndarray:
    binary_img = (img == 0)  # occupied cells
    r = round(radius / resolution)
    Y, X = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    SE = X**2 + Y**2 <= r**2
    SE = SE.astype(int)
    bin = sp.binary_dilation(binary_img, SE).astype(np.bool_)
    distances = sp.distance_transform_edt(~binary_img)
    scale = 5.0
    costmap = np.clip(np.max(distances) - (distances * scale), 0, 255).astype(np.float32)
    return costmap 

def _compute_global_plan(
        occupancy_dict: dict,
        start_coords: np.ndarray,   # coordinates in gridmap reference frame
        goal_coords: np.ndarray,
    ) -> np.ndarray:
    dims = occupancy_dict['dims']  
    img = occupancy_dict['buffer'].cpu().numpy().reshape(dims[1], dims[0])
    if img[start_coords[0], start_coords[1]] == 0 or img[goal_coords[0], goal_coords[1]] == 0:
        raise ValueError("Start or goal is in an occupied cell.")

    costmap = _inflate_img(img, radius=0.2, resolution=occupancy_dict['resolution']) + 1.0  # avoid zero-cost cells
    costmap[costmap >= 255] = float("inf")
    a_star_path = pyastar2d.astar_path(costmap, start_coords, goal_coords, allow_diagonal=False) 
    # debug: visualize costmap and path (in color)
    from PIL import Image
    costmap_img = Image.fromarray(costmap.astype(np.uint8)).convert("RGB")
    for coord in a_star_path:
        costmap_img.putpixel((coord[1], coord[0]), (255, 0, 0))
    costmap_img.save("costmap_with_path.png")
    return a_star_path
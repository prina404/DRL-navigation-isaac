import os
import hydra
from omegaconf import DictConfig
import rclpy
import torch
import time
import math
import argparse
from isaaclab.app import AppLauncher
# # add argparse arguments
# parser = argparse.ArgumentParser(description="Tutorial on basic RL environment.")

# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
#args_cli = parser.parse_args()
app_launcher = AppLauncher(enable_cameras=True)#args_cli)
simulation_app = app_launcher.app

import torch

from go2.go2_env import Go2RSLEnvCfg, camera_follow
import go2.go2_sensors as go2_sensors
import omni
import carb
import go2.go2_ctrl as go2_ctrl
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from pxr import Usd, UsdGeom, UsdPhysics, Sdf
import omni.replicator.core as rep



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
    asset_path =  os.path.join(asset_folder, cfg.env_name, cfg.env_name + ".usda")
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
            print(f"[physics-fix] Disabled {num_disabled} rigid bodies under {SCENE_PRIM} (treating env as static)")
        
        num_kin = _make_root_xforms_kinematic(SCENE_PRIM, categories=kinematic_categories)
        if num_kin > 0:
            print(f"[physics] Marked {num_kin} root Xforms as kinematic under {SCENE_PRIM} for categories={kinematic_categories}")
          
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


FILE_PATH = os.path.join(os.path.dirname(__file__), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
def run_simulator(cfg: DictConfig):

    # Go2 Environment setup
    go2_env_cfg = Go2RSLEnvCfg()
    go2_env_cfg.scene.num_envs = cfg.num_envs
    go2_env_cfg.decimation = math.ceil(1./go2_env_cfg.sim.dt/cfg.freq)
    go2_env_cfg.sim.render_interval = go2_env_cfg.decimation
    go2_ctrl.init_base_vel_cmd(cfg.num_envs)
    env, policy = go2_ctrl.get_rsl_flat_policy(go2_env_cfg)

    load_interiorAgent_env(cfg, env.unwrapped.scene.env_ns)
    #load_infinigen_env(cfg, env.unwrapped.scene.env_ns)

    
    # Sensor setup
    sm = go2_sensors.SensorManager(cfg.num_envs)
    lidar_annotators = sm.add_rtx_lidar()
    cameras = sm.add_camera(cfg.freq)

    
    # Run simulation
    sim_step_dt = float(go2_env_cfg.sim.dt * go2_env_cfg.decimation)
    obs, _ = env.reset()
    paused = False
    while simulation_app.is_running():
        start_time = time.time()

        if paused:
            time.sleep(0.01)
            continue
        with torch.no_grad():            
            # control joints
            actions = policy(obs)
            # step the environment
            obs, _, _, _ = env.step(actions)

            if cfg.camera_follow:
                camera_follow(env)

        elapsed_time = time.time() - start_time
        if elapsed_time < sim_step_dt:
            sleep_duration = sim_step_dt - elapsed_time
            time.sleep(sleep_duration)

if __name__ == "__main__":
    run_simulator()
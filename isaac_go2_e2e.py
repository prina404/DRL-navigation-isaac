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

app_launcher = AppLauncher()#args_cli)
simulation_app = app_launcher.app

import torch

from go2.go2_env import Go2RSLEnvCfg, camera_follow
import go2.go2_sensors as go2_sensors
import omni
import carb
import go2.go2_ctrl as go2_ctrl
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from pxr import Usd, UsdGeom, UsdPhysics
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


def load_interiorAgent_env(cfg: DictConfig, namespace: str) -> None:
    ground_plane = rep.get.prims("/World/GroundPlane")
    with ground_plane:
        rep.modify.semantics([("class", "floor")])

    asset_folder = os.path.expandvars(cfg.usd_folder)
    asset_path =  os.path.join(asset_folder, cfg.env_name, cfg.env_name + ".usda")
    for i in range(cfg.num_envs):
        SCENE_PRIM = f"{namespace}/env_{i}/{cfg.env_name}" 
        prim = get_prim_at_path(SCENE_PRIM)        
        if prim is None or not prim.IsValid():
            prim = define_prim(SCENE_PRIM, "Xform")
        prim.GetReferences().AddReference(asset_path)
        x = UsdGeom.XformCommonAPI(get_prim_at_path(SCENE_PRIM))
        x.SetTranslate((0.0, 0.0, 1.0))  

        num_disabled = _disable_rigid_bodies(SCENE_PRIM)
        if num_disabled > 0:
            print(f"[physics-fix] Disabled {num_disabled} rigid bodies under {SCENE_PRIM} (treating env as static)")


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

    # Keyboard control
    system_input = carb.input.acquire_input_interface()
    system_input.subscribe_to_keyboard_events(
        omni.appwindow.get_default_app_window().get_keyboard(), go2_ctrl.sub_keyboard_event)
    
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
import argparse
import re

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(
    {
        "headless": True,
        "kit_args": "--enable isaacsim.asset.gen.omap --/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error",
    }
).app

import os
import traceback

import isaaclab.sim as sim_utils
import omni.usd
import yaml
from loguru import logger
from omni.usd.commands import MovePrimCommand
from pxr import Usd
from tqdm import tqdm

import cfg.CFG as CFG
import preprocessing.interioragent_utils.doors as doors
import preprocessing.interioragent_utils.gridmap as gridmap
import preprocessing.interioragent_utils.physics as physics
import preprocessing.interioragent_utils.chairs as chairs

parser = argparse.ArgumentParser(description="Preprocess USD for InteriorAgent.")
parser.add_argument(
    "--file",
    type=str,
    default=None,
    help="Path to the USD file to preprocess. If not provided, will preprocess all USDs in the INTERIOR_AGENT_DIR.",
)


def validate_yaml_cfg(cfg: dict, env_name: str = None):
    assert "entrance" in cfg, "Door config must specify entrance group index."
    assert len(cfg["entrance"]) > 0, "Door config must specify at least one entrance."
    all_doors = []
    for key, door_lst in cfg.items():
        if key not in ["normal_doors", "disabled_doors", "entrance"]:
            continue
        if door_lst:
            all_doors.extend(door_lst)

    assert len(set(all_doors)) == len(all_doors), f"Duplicate door names found for environment {env_name}"


def load_interioragent_cfg(env_name: str = None) -> dict:
    with open(CFG.CFG_DIR / "interioragent_preprocessing.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)

    if env_name not in cfg_dict:
        raise RuntimeError(f"No door configuration found for {env_name}.")

    cfg = cfg_dict[env_name]

    if "normal_doors" not in cfg or cfg["normal_doors"] is None:
        logger.warning(f"No normal door configuration found for {env_name}, no dynamic door will be set up.")
        cfg["normal_doors"] = []
    if "disabled_doors" not in cfg or cfg["disabled_doors"] is None:
        logger.warning(f"No disabled door configuration found for {env_name}, no doors will be disabled.")
        cfg["disabled_doors"] = []

    validate_yaml_cfg(cfg, env_name)
    return cfg

def is_dynamic_door(prim: Usd.Prim, door_cfg: dict) -> bool:
    return prim.IsValid() and doors.is_door_root_xform(prim) and prim.GetName() in door_cfg["normal_doors"]

def is_dynamic_chair(prim: Usd.Prim) -> bool:
    return prim.IsValid() and chairs.is_chair_xform(prim)

def init_static_dynamic_folders(
    usd_path: str,
    door_cfg: dict,
):
    # I need to init an omni.usd context + stage in order to use the MovePrimsCommand.
    context = omni.usd.get_context()
    context.open_stage(usd_path)
    stage = context.get_stage()
    if not stage:
        raise RuntimeError(f"Failed to open USD: {usd_path}")

    # Create two subfolders under /Root/Meshes. This is needed to differentiate lidar collisions in an efficient manner
    root_path: str = "/Root/Meshes"
    mesh_prim = stage.GetPrimAtPath(root_path)

    static_path = f"{root_path}/static_objects"
    dynamic_path = f"{root_path}/dynamic_objects"

    static_folder: Usd.Prim = stage.DefinePrim(static_path, "Xform")
    dynamic_folder = stage.DefinePrim(dynamic_path, "Xform")

    sim_utils.standardize_xform_ops(static_folder)  # needed for isaaclab xform validation
    sim_utils.standardize_xform_ops(dynamic_folder)

    # Move all prims into static folder by default
    for child in mesh_prim.GetChildren():
        if child.GetName() in ["static_objects", "dynamic_objects"]:
            continue
        src = child.GetPath()
        dst = f"{static_path}/{child.GetName()}"
        MovePrimCommand(src, dst, destructive=False, stage_or_context=stage).do()

    dynamic_other_path = f"{dynamic_path}/other"
    stage.DefinePrim(dynamic_other_path, "Xform")
    # Move only enabled doors into dynamic folder
    prims = list(Usd.PrimRange(static_folder))
    for prim in prims:
        if is_dynamic_door(prim, door_cfg) or is_dynamic_chair(prim):
            src = prim.GetPath()
            dst = f"{dynamic_other_path}/{prim.GetName()}"
            MovePrimCommand(src, dst, destructive=False, stage_or_context=stage).do()

    # Save a temp .usda file
    src_dir, src_name = os.path.split(usd_path)
    base, ext = os.path.splitext(src_name)
    dst_path = os.path.join(src_dir, f"{base}_temp{ext}")
    stage.GetRootLayer().Export(dst_path)
    context.close_stage()

    return dst_path


def preprocess_interioragent_scene(scene_dir: str, usd_path: str):
    env_name = os.path.basename(scene_dir)
    env_cfg = load_interioragent_cfg(env_name)

    temp_usd_file = init_static_dynamic_folders(usd_path, env_cfg)

    stage = Usd.Stage.Open(temp_usd_file)  # reopen stage to ensure moves are registered
    root = stage.GetPseudoRoot()

    ## Set all geometry to static by disabling rigid body API
    physics.set_stage_static(root)

    ## Enable door physics by setting up rigid body and hinge constraint APIs on enabled door prims
    dynamic_doors = doors.set_door_physics(root, env_cfg)
    dynamic_chairs = chairs.set_chair_physics(root)

    ## Save preprocessed USD
    src_dir, src_name = os.path.split(usd_path)
    base, ext = os.path.splitext(src_name)
    dst_path = os.path.join(src_dir, f"{base}_baked{ext}")
    stage.GetRootLayer().Export(dst_path)

    os.remove(temp_usd_file)  # clean up temp file
    del stage  # this should call the destructor

    gridmap.create_map_from_file(dst_path, env_cfg)  # create occupancy map for the preprocessed USD
    logger.info(f"Preprocessed USD saved to {dst_path} with {len(dynamic_doors)} dynamic doors and {len(dynamic_chairs)} dynamic chairs.")


def main(args_cli: argparse.Namespace):
    if args_cli.file:
        usd_path = args_cli.file
        scene_dir = os.path.dirname(usd_path)
        preprocess_interioragent_scene(scene_dir, usd_path)
    else:
        # logger sink setup to work well with tqdm progress bars
        logger.remove()
        logger.add(lambda msg: None, level="ERROR")
        logger.add(
            lambda msg: tqdm.write(msg, end="\n"),
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            level="INFO",
            colorize=True,
        )

        directories = sorted(os.listdir(CFG.get_dataset_dir()))
        env_dir = [d for d in directories if d.startswith("kujiale_") and os.path.isdir(os.path.join(CFG.get_dataset_dir(), d))]
        for directory in tqdm(env_dir, desc="Preprocessing InteriorAgent scenes", leave=False):
            scene_dir = os.path.join(CFG.get_dataset_dir(), directory)
            usd_path = os.path.join(scene_dir, f"{directory}.usda")
            if not os.path.exists(usd_path):
                logger.error(f"Could not find expected USD file: {usd_path}")
                continue

            preprocess_interioragent_scene(scene_dir, usd_path)


if __name__ == "__main__":
    try:
        main(parser.parse_known_args()[0])
    except Exception:
        traceback.print_exc()

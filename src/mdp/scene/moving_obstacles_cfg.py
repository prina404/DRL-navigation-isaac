import isaaclab.sim as sim
from isaaclab.assets import RigidObjectCfg

from cfg.CFG import ASSET_DIR


def get_obstacles_cfg() -> dict[str, RigidObjectCfg]:
    return {
        "toolbox": RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/toolbox",
            spawn=sim.UsdFileCfg(usd_path=str(ASSET_DIR / "obstacles/toolbox_01.usd")),
        ),
        "plant": RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/plant",
            spawn=sim.UsdFileCfg(usd_path=str(ASSET_DIR / "obstacles/plant_01.usd")),
        ),
        "trashcan": RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/trashcan",
            spawn=sim.UsdFileCfg(usd_path=str(ASSET_DIR / "obstacles/trashcan_01.usd")),
        ),
        # TODO: add more obstacles classes
    }

from pathlib import Path

import torch
import yaml

ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent
ASSET_DIR = ROOT_DIR / "asset"
CHECKPOINT_DIR = ROOT_DIR / "ckpts"
CFG_DIR = ROOT_DIR / "src/cfg"
NAVPOINTS_FILE = ROOT_DIR/"src/tasks/nav_points.yaml"

VINT_MODEL_WEIGHTS = CHECKPOINT_DIR / "vint_model_weights/vint.pth"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_cfg() -> dict:
    cfg_file = ROOT_DIR / "dataset_cfg.yaml"
    return yaml.safe_load(cfg_file.open())


def store_cfg(cfg: dict):
    cfg_file = ROOT_DIR / "dataset_cfg.yaml"
    with open(cfg_file, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def get_dataset_dir() -> Path:
    cfg = load_cfg()
    return (ROOT_DIR / cfg["dataset_folder"]).resolve()


def get_scene_usd_path() -> Path:
    cfg = load_cfg()
    env_name = cfg["current_env"]
    return get_dataset_dir() / env_name / f"{env_name}_baked.usda"

def get_map_name() -> str:
    cfg = load_cfg()
    return cfg["current_env"]
from pathlib import Path

import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ASSET_DIR = ROOT_DIR / "asset"
CHECKPOINT_DIR = ROOT_DIR / "ckpts"
CFG_DIR = ROOT_DIR / "src/cfg"

cfg_file = ROOT_DIR / "dataset_cfg.yaml"
cfg = yaml.safe_load(cfg_file.open())

VINT_MODEL_WEIGHTS = CHECKPOINT_DIR / "vint_model_weights/vint.pth"

INTERIOR_AGENT_DIR = (ROOT_DIR / cfg["dataset_folder"]).resolve()
print(f"INTERIOR_AGENT_DIR set to: {INTERIOR_AGENT_DIR}")

SCENE_USD_PATH = INTERIOR_AGENT_DIR / cfg["env_folder"] / f'{cfg["env_name"]}.usda'

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ASSET_DIR = ROOT_DIR / "asset"
CHECKPOINT_DIR = ROOT_DIR / "ckpts"
CFG_DIR = ROOT_DIR / "src/cfg"

VINT_MODEL_WEIGHTS = CHECKPOINT_DIR / "vint_model_weights/vint.pth"

INTERIOR_AGENT_DIR = ROOT_DIR.parent / "InteriorAgent"
print(f"INTERIOR_AGENT_DIR set to: {INTERIOR_AGENT_DIR}")

# TODO: fix hardcoded scene path
SCENE_USD_PATH = INTERIOR_AGENT_DIR / "kujiale_0003/kujiale_0003_baked.usda"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
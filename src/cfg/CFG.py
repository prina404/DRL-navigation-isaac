
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ASSET_DIR = ROOT_DIR / "asset"
CHECKPOINT_DIR = ROOT_DIR / "ckpts"

INTERIOR_AGENT_DIR = ROOT_DIR.parent / "InteriorAgent"

# TODO: fix hardcoded scene path
SCENE_USD_PATH = INTERIOR_AGENT_DIR / "kujiale_0067/kujiale_0067_baked.usda"
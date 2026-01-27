
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ASSET_DIR = ROOT_DIR / "asset"
CHECKPOINT_DIR = ROOT_DIR / "ckpts"


# TODO: fix hardcoded scene path
SCENE_USD_PATH = ROOT_DIR.parent / "InteriorAgent/kujiale_0067/kujiale_0067.usda"
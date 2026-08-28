from pathlib import Path

import torch
import yaml

ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent
ASSET_DIR = ROOT_DIR / "asset"
CHECKPOINT_DIR = ROOT_DIR / "ckpts"
CFG_DIR = ROOT_DIR / "src/cfg"
NAVPOINTS_FILE = ROOT_DIR/"src/tasks/nav_points.yaml"
DISTILLATION_DIR = ROOT_DIR / "distillation_data"

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


_MAP_OVERRIDE: str | None = None


def available_maps() -> list[str]:
    """Every preprocessed scene in the dataset folder, i.e. the ones that have a baked usd."""
    return sorted(p.parent.name for p in get_dataset_dir().glob("*/*_baked.usda"))


def resolve_map_name(name: str) -> str:
    stem = str(name).strip()
    if not stem:
        raise ValueError("Empty map name")

    prefix, _, digits = stem.rpartition("_")
    if not digits.isdigit():  # a name with no trailing number, keep it as it was typed
        map_name = stem
    else:
        map_name = f"{prefix or 'kujiale'}_{digits.zfill(4)}"

    if not (get_dataset_dir() / map_name / f"{map_name}_baked.usda").is_file():
        raise FileNotFoundError(
            f"Map '{map_name}' has no baked scene under {get_dataset_dir()}. "
            f"Preprocessed maps: {', '.join(available_maps()) or '<none>'}"
        )
    return map_name


def set_map_name(name: str) -> str:
    global _MAP_OVERRIDE
    _MAP_OVERRIDE = resolve_map_name(name)
    return _MAP_OVERRIDE


def get_map_name() -> str:
    if _MAP_OVERRIDE is not None:
        return _MAP_OVERRIDE
    return load_cfg()["current_env"]  # fallback for tooling with no --map, e.g. task_annotator.ipynb


def get_scene_usd_path() -> Path:
    env_name = get_map_name()
    return get_dataset_dir() / env_name / f"{env_name}_baked.usda"

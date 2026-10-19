import math
from pathlib import Path
from typing import Literal

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


_MAPS_OVERRIDE: list[str] | None = None

#: Distance between the origins of neighbouring maps [m]. The largest map spans ~17 m, so neighbours never overlap.
MAP_SPACING = 30.0


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

    if not get_scene_usd_path(map_name).is_file():
        raise FileNotFoundError(
            f"Map '{map_name}' has no baked scene under {get_dataset_dir()}. "
            f"Preprocessed maps: {', '.join(available_maps()) or '<none>'}"
        )
    return map_name


def set_map_names(names: list[str]) -> list[str]:
    global _MAPS_OVERRIDE
    _MAPS_OVERRIDE = list(dict.fromkeys(resolve_map_name(name) for name in names))  # drop duplicates, keep order
    return _MAPS_OVERRIDE


def get_map_names() -> list[str]:
    if _MAPS_OVERRIDE is not None:
        return _MAPS_OVERRIDE
    return [load_cfg()["current_env"]]  # fallback for tooling with no --maps, e.g. task_annotator.ipynb


def set_map_name(name: str) -> str:
    """Single map version of `set_map_names`, for the tooling that only handles one map (ROS, distillation)."""
    return set_map_names([name])[0]


def get_map_name() -> str:
    """The map of a single map run, for the tooling that only handles one map (ROS, distillation)."""
    names = get_map_names()
    if len(names) != 1:
        raise ValueError(f"Expected a single map, but {len(names)} are loaded: {names}")
    return names[0]


def get_run_name() -> str:
    """Names logs and checkpoints after the map of a single map run, or "generalist" when there are several."""
    names = get_map_names()
    return names[0] if len(names) == 1 else "generalist"


def get_map_origins() -> dict[str, tuple[float, float, float]]:
    """World position of the frame of every loaded map. Maps are laid out on a square grid, MAP_SPACING apart."""
    names = get_map_names()
    num_cols = math.ceil(math.sqrt(len(names)))
    return {name: (MAP_SPACING * (i // num_cols), MAP_SPACING * (i % num_cols), 0.0) for i, name in enumerate(names)}


def get_scene_usd_path(map_name: str | None = None) -> Path:
    map_name = map_name or get_map_name()  # tooling that handles a single map, e.g. task_annotator.ipynb
    return get_dataset_dir() / map_name / f"{map_name}_baked.usda"


def get_split_usd_path(map_name: str, part: Literal["static", "dynamic"]) -> Path:
    """One half of the baked scene: its static geometry, or its doors and chairs (see `write_split_layers`)."""
    path = get_dataset_dir() / map_name / f"{map_name}_{part}.usda"
    if not path.is_file():  # maps baked before the scene was split in two
        from preprocessing.interioragent_utils.layers import write_split_layers

        write_split_layers(get_scene_usd_path(map_name))
    return path

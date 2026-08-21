from pathlib import Path

import torch
from loguru import logger

from cfg.CFG import DISTILLATION_DIR

RAW_DIR = DISTILLATION_DIR / "raw_data"
"""Where ``DistillationRecorder`` output is kept before ``split_raw_recordings`` consumes it."""

EPISODE_TABLE_KEYS = ("episode_index", "episode_collided")
"""The per-episode table. These two are aligned with each other by position, not with the rows."""


def _take(raw: dict, row_keys: list[str], rows: torch.Tensor, episodes: torch.Tensor) -> dict:
    """One side of the split: row tensors indexed, the episode table subset, anything else copied as it is."""
    out = {}
    for key, value in raw.items():
        if key in EPISODE_TABLE_KEYS:
            out[key] = value[episodes]
        elif key in row_keys:
            out[key] = value[rows]
        else:  # cloned rather than passed through: torch.save writes the whole storage behind a view
            out[key] = value.clone() if torch.is_tensor(value) else value
    return out


def _split_one(raw_path: Path, train_path: Path, val_path: Path, val_fraction: float, seed: int) -> None:
    """Split one recording into ``train_path`` and ``val_path``, deleting it once it is safely in RAM.

    The recording is read into memory rather than mmap'd so that it can be unlinked *before* its two splits are
    written. The pair weighs what the original did, so writing them while the original is still on disk needs a
    spare copy's worth of free space; going through RAM instead means the recording's own blocks are what the
    splits are written into, and the disk never has to hold both. Costs about twice the recording in RAM, and
    means a crash between the unlink and the last save loses that one recording.
    """
    raw = torch.load(raw_path, map_location="cpu")  # not mmap'd: the file has to be closed before it is unlinked
    num_rows = raw["episode_id"].numel()
    # `action_std` is why this goes by shape rather than by name: the recorder documents it as a (3,) vector, but
    # every file collected so far holds an (N, 3) copy of it instead.
    row_keys = [
        key
        for key, value in raw.items()
        if key not in EPISODE_TABLE_KEYS and torch.is_tensor(value) and value.ndim and value.shape[0] == num_rows
    ]

    # the holdout is drawn over whole episodes, not over rows: consecutive rows of one rollout are 80 ms apart and
    # near-duplicates of each other, so a row-wise holdout would leak the validation set into training
    index = raw["episode_index"]
    perm = torch.randperm(index.numel(), generator=torch.Generator().manual_seed(seed))
    num_val = round(val_fraction * index.numel())
    val_pos, train_pos = perm[:num_val].sort().values, perm[num_val:].sort().values  # sorted: the table stays ascending

    val_mask = torch.isin(raw["episode_id"], index[val_pos])
    val_rows, train_rows = val_mask.nonzero(as_tuple=True)[0], (~val_mask).nonzero(as_tuple=True)[0]
    logger.info(
        f"{raw_path.name}: {num_rows} rows / {index.numel()} episodes -> train {train_rows.numel()}, "
        f"val {val_rows.numel()} ({val_rows.numel() / num_rows:.1%})"
    )

    raw_path.unlink()  # the freed blocks are what the two splits are about to be written into
    for path, rows, episodes in ((train_path, train_rows, train_pos), (val_path, val_rows, val_pos)):
        torch.save(_take(raw, row_keys, rows, episodes), path)  # freed again before the other side is built


def split_raw_recordings(
    raw_dir: Path = RAW_DIR,
    out_dir: Path = DISTILLATION_DIR,
    val_fraction: float = 0.15,
    seed: int = 0,
) -> None:
    """
    Split every recording under ``raw_dir`` into a train/val pair, one at a time, deleting each as it is split.

    Peaks at roughly twice one recording in RAM and at zero extra disk, so it runs on a nearly full disk. Already
    split maps are simply gone from ``raw_dir``, which is what makes an interrupted run resumable.
    """
    for sub in ("train", "val"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    for raw_path in sorted(raw_dir.glob("*/*.pt")):
        name = f"{raw_path.parent.name}_{raw_path.stem.split('_')[-1]}.pt"  # <map>_<deterministic|stochastic>.pt
        _split_one(raw_path, out_dir / "train" / name, out_dir / "val" / name, val_fraction, seed)


if __name__ == "__main__":
    split_raw_recordings()

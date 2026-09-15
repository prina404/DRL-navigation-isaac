"""Cut the raw recordings into a train/val/test collection, splitting over whole episodes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger

from cfg.CFG import DISTILLATION_DIR

RAW_DIR = DISTILLATION_DIR / "raw_data"
"""Where ``DistillationRecorder`` output is kept before ``split_raw_recordings`` consumes it."""

SPLITS = ("train", "val", "test")

FRACTIONS = (0.75, 0.15, 0.10)

EPISODE_TABLE_KEYS = ("episode_index", "episode_collided")


def _take(raw: dict, row_keys: set[str], rows: torch.Tensor, episodes: torch.Tensor) -> dict:
    out = {}
    for key, value in raw.items():
        if key in EPISODE_TABLE_KEYS:
            out[key] = value[episodes]
        elif key in row_keys:
            out[key] = value[rows]
        else:  # cloned rather than passed through: torch.save writes the whole storage behind a view
            out[key] = value.clone() if torch.is_tensor(value) else value
    return out


def _episode_groups(num_episodes: int, fractions: tuple[float, ...], seed: int) -> list[torch.Tensor]:
    """Positions in the episode table belonging to each split, sorted so the table stays ascending."""
    perm = torch.randperm(num_episodes, generator=torch.Generator().manual_seed(seed))
    sizes = [round(fraction * num_episodes) for fraction in fractions[:-1]]
    sizes.append(num_episodes - sum(sizes))  # the last split takes the remainder, so no episode is dropped
    offsets = torch.tensor([0] + sizes).cumsum(0)
    return [perm[start:stop].sort().values for start, stop in zip(offsets[:-1], offsets[1:])]


def _split_one(raw_path: Path, out_paths: list[Path], fractions: tuple[float, ...], seed: int) -> None:
    """Split one recording into ``out_paths``, deleting it once they are all written.
    """
    raw = torch.load(raw_path, map_location="cpu")  # not mmap'd: the file has to be closed before it is unlinked
    num_rows = raw["episode_id"].numel()
    # `action_std` is why this goes by shape rather than by name: the recorder documents it as a (3,) vector, but
    # every file collected so far holds an (N, 3) copy of it instead.
    row_keys = {
        key
        for key, value in raw.items()
        if key not in EPISODE_TABLE_KEYS and torch.is_tensor(value) and value.ndim and value.shape[0] == num_rows
    }

    index = raw["episode_index"]
    groups = _episode_groups(index.numel(), fractions, seed)

    label = torch.empty(int(index.max()) + 1, dtype=torch.int8)
    for split, episodes in enumerate(groups):
        label[index[episodes]] = split
    row_split = label[raw["episode_id"]]

    parts = [(row_split == split).nonzero(as_tuple=True)[0] for split in range(len(groups))]
    logger.info(
        f"{raw_path.name}: {num_rows} rows / {index.numel()} episodes -> "
        + ", ".join(f"{path.parent.name} {rows.numel()} ({rows.numel() / num_rows:.1%})" for path, rows in zip(out_paths, parts))
    )

    partials = [path.with_suffix(".pt.part") for path in out_paths]
    for partial, rows, episodes in zip(partials, parts, groups):
        torch.save(_take(raw, row_keys, rows, episodes), partial)
    for partial, path in zip(partials, out_paths):
        partial.rename(path)
    raw_path.unlink()


def split_raw_recordings(
    raw_dir: Path = RAW_DIR,
    out_dir: Path = DISTILLATION_DIR,
    fractions: tuple[float, ...] = FRACTIONS,
    seed: int = 0,
) -> None:
    """Split every recording under ``raw_dir`` into ``out_dir/{train,val,test}``, one recording at a time.

    """
    if len(fractions) != len(SPLITS):
        raise ValueError(f"expected one fraction per split {SPLITS}, got {fractions}")
    for split in SPLITS:
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    for raw_path in sorted(raw_dir.glob("*/*.pt")):
        name = f"{raw_path.parent.name}_{raw_path.stem.split('_')[-1]}.pt"  # <map>_<deterministic|stochastic>.pt
        _split_one(raw_path, [out_dir / split / name for split in SPLITS], fractions, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split the raw distillation recordings into train/val/test.")
    parser.add_argument("--raw_dir", type=Path, default=RAW_DIR, help="Where the recordings to split are.")
    parser.add_argument("--out_dir", type=Path, default=DISTILLATION_DIR, help="Root the split dirs are created in.")
    parser.add_argument("--fractions", type=float, nargs=len(SPLITS), default=FRACTIONS, help=f"Episode share of {SPLITS}.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the episode permutation.")
    args = parser.parse_args()
    split_raw_recordings(args.raw_dir, args.out_dir, tuple(args.fractions), args.seed)

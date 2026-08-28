
from __future__ import annotations

import random
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset

from cfg.CFG import DISTILLATION_DIR


class PolicyDataset(IterableDataset):
    '''
    Each iteration yields ``rows_per_yield`` rows at once extracted at random from the map's data file
    '''

    def __init__(
        self,
        data_path: Path,
        drop_collided: bool = True,
        rows_per_yield: int = 64,
    ):
        # one map's half of one split; picking which map and which policy mode is `_split_files`' job
        self.data_path = data_path
        self.max_chunks = 4
        self.chunk_size = 8192
        self.max_len = self.chunk_size * self.max_chunks  # rows per queue
        self.drop_collided = drop_collided
        self.rows_per_yield = rows_per_yield

        self.obs_keys = ('lidar', 'global_plan', 'velocity_buffer', 'action_buffer')
        self.data_keys = self.obs_keys + ('action_mean',) + (('episode_id',) if drop_collided else ())

        # with mmap load only metadata for init
        data = torch.load(self.data_path, map_location='cpu', mmap=True)
        self.num_rows = data['lidar'].size(0)

        std = data['action_std']
        self.action_std = (std[0] if std.ndim == 2 else std).clone()

        # `episode_collided` is aligned with `episode_index` by position, and the ids are sparse, so it cannot be
        # indexed by id directly.
        index = data['episode_index']
        self.collided = torch.zeros(int(index.max()) + 1, dtype=torch.bool)
        self.collided[index] = data['episode_collided']

    def _load_chunk(self, data, start: int) -> dict[str, torch.Tensor]:
        queue = {key: data[key][start : start + self.max_len].clone() for key in self.data_keys}

        if self.drop_collided:  # order is a subset of rows that have not collided
            # `self.collided` is keyed by episode id, so the ids index it directly
            order = (~self.collided[queue.pop('episode_id')]).nonzero(as_tuple=True)[0]
            order = order[torch.randperm(order.numel())]
        else:
            order = torch.randperm(queue['action_mean'].size(0))
        return {key: value[order] for key, value in queue.items()}

    def __iter__(self):
        # mmap must be true to avoid OOM
        data_mmap = torch.load(self.data_path, map_location='cpu', mmap=True)

        # no sharding support = no multi-workers
        worker = torch.utils.data.get_worker_info()
        if worker is not None and worker.num_workers > 1:
            raise NotImplementedError("Multi-worker support is not supported by PolicyDataset ")

        # breaking out of the loop closes the generator, which exits this block and shuts the pool down
        with ThreadPoolExecutor(1, thread_name_prefix='refill') as pool:
            cursor = 0
            pending = pool.submit(self._load_chunk, data_mmap, cursor)  # the only read the consumer ever waits for
            while pending is not None:
                data = pending.result()
                cursor += self.max_len
                pending = pool.submit(self._load_chunk, data_mmap, cursor) if cursor < self.num_rows else None

                # the tail shorter than a full block is dropped: at most 63 rows of the ~25.6k a chunk keeps
                n_rows = self.rows_per_yield
                for i in range(0, data['action_mean'].size(0) - n_rows + 1, n_rows):
                    yield ({key: data[key][i: i + n_rows] for key in self.obs_keys},
                           {'action_mean': data['action_mean'][i: i + n_rows]})


def parse_map_ids(spec: str | Sequence[str] | None) -> list[str] | None:
    if spec is None:
        return None
    items = spec.split(",") if isinstance(spec, str) else list(spec)
    ids = []
    for item in items:
        digits = str(item).strip().rsplit("_", 1)[-1]  # the trailing number of kujiale_0008, or the whole item
        if not digits:
            continue
        if not digits.isdigit():
            raise ValueError(f"'{item}' is not a map id; expected digits, e.g. 08 or 0008 or kujiale_0008")
        ids.append(digits.zfill(4))
    return ids or None


def _map_id(path: Path) -> str:
    """The zero-padded id of the map a split file holds, i.e. ``0008`` for ``kujiale_0008_deterministic.pt``."""
    return path.stem.rsplit("_", 1)[0].rsplit("_", 1)[-1]  # drop the policy mode, then keep the trailing number


def _split_files(split_dir: Path, use_deterministic: bool, maps: Sequence[str] | None = None) -> list[Path]:
    mode = "deterministic" if use_deterministic else "stochastic"
    files = sorted(split_dir.glob(f"*_{mode}.pt"))
    if not files:
        raise FileNotFoundError(f"No *_{mode}.pt recording found under {split_dir}")

    if maps is not None:
        wanted = set(maps)
        available = {_map_id(path) for path in files}
        missing = sorted(wanted - available)
        if missing:
            raise FileNotFoundError(f"{split_dir} holds no {mode} recording for map(s) {missing}, has {sorted(available)}")
        files = [path for path in files if _map_id(path) in wanted]
    return files


class MixedPolicyDataset(IterableDataset):
    """
    This is basically a round-robin of ``PolicyDataset``: pick one random PolicyDataset and yield a block of rows
    """

    def __init__(
        self,
        data_dir: Path = DISTILLATION_DIR,
        split: str = "train",
        use_deterministic: bool = True,
        drop_collided: bool = True,
        rows_per_epoch: int | None = None,
        restart_exhausted: bool = True,
        with_std: bool = False,
        seed: int | None = None,
        files: list[Path] | None = None,
        rows_per_yield: int = 64,
        maps: Sequence[str] | None = None,
    ):
        # `data_dir` is the root of the collection, `split` the split of it to read and `maps` which of its maps;
        # `files` lets a caller hand over an explicit subset instead
        self.files = files if files is not None else _split_files(data_dir / split, use_deterministic, maps)
        self.rows_per_yield = rows_per_yield
        self.child_kwargs = dict(drop_collided=drop_collided, rows_per_yield=rows_per_yield)
        self.rows_per_epoch = rows_per_epoch
        self.restart_exhausted = restart_exhausted
        self.with_std = with_std
        self.seed = seed

    @property
    def map_names(self) -> list[str]:
        return [path.stem.rsplit("_", 1)[0] for path in self.files]  # drop the trailing policy mode

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        if worker is not None and worker.num_workers > 1:
            raise NotImplementedError(
                "MixedPolicyDataset inherits PolicyDataset's single-worker limit. "
            )

        randrange = random.Random(self.seed).randrange
        datasets = [PolicyDataset(path, **self.child_kwargs) for path in self.files]
        streams = [iter(dataset) for dataset in datasets]
        live = list(range(len(datasets)))  # maps still producing rows
        limit, emitted, n_rows = self.rows_per_epoch, 0, self.rows_per_yield
        # the std is per map and constant, so one broadcast view per map covers every block it ever yields
        stds = [dataset.action_std.expand(n_rows, 3) for dataset in datasets]

        while live and (limit is None or emitted < limit):
            pick = randrange(len(live))
            index = live[pick]
            try:
                obs, target = next(streams[index])
            except StopIteration:
                if self.restart_exhausted:
                    streams[index] = iter(datasets[index])
                else:
                    live.pop(pick)
                continue
            emitted += n_rows
            yield obs, ({**target, "action_std": stds[index]} if self.with_std else target)


def _concat_blocks(blocks):
    """Collate blocks of rows into one flat batch.
    Each iteration over MixedPolicyDataset yields a block of rows from one map. This func ensures that
    all blocks are concatenated into a single batch
    """
    observations, targets = zip(*blocks)
    return (
        {key: torch.cat([block[key] for block in observations]) for key in observations[0]},
        {key: torch.cat([block[key] for block in targets]) for key in targets[0]},
    )


def build_split_dataloader(
    split: str,
    batch_size: int,
    data_dir: Path = DISTILLATION_DIR,
    use_deterministic: bool = True,
    drop_collided: bool = True,
    steps_per_epoch: int | None = None,
    with_std: bool = False,
    seed: int | None = None,
    rows_per_yield: int = 64,
    restart_exhausted: bool = True,
    maps: Sequence[str] | None = None,
) -> DataLoader:

    if rows_per_yield < 1:
        raise ValueError(f"rows_per_yield must be at least 1, got {rows_per_yield}")
    if batch_size % rows_per_yield:
        raise ValueError(f"batch_size={batch_size} must be a multiple of rows_per_yield={rows_per_yield}")

    dataset = MixedPolicyDataset(
        data_dir=data_dir,
        split=split,
        use_deterministic=use_deterministic,
        drop_collided=drop_collided,
        rows_per_epoch=None if steps_per_epoch is None else steps_per_epoch * batch_size,
        restart_exhausted=restart_exhausted,
        with_std=with_std,
        seed=seed,
        rows_per_yield=rows_per_yield,
        maps=maps,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size // rows_per_yield,
        collate_fn=_concat_blocks,
        num_workers=0,  # keep at 0 because PolicyDataset is single-worker only
        drop_last=True,
    )


def get_distillation_dataloader(
    batch_size: int,
    data_dir: Path = DISTILLATION_DIR,
    use_deterministic: bool = True,
    drop_collided: bool = True,
    steps_per_epoch: int | None = None,
    with_std: bool = False,
    seed: int | None = None,
    rows_per_yield: int = 64,
    maps: Sequence[str] | None = None,
) -> tuple[DataLoader, DataLoader]:
    shared = dict(
        data_dir=data_dir,
        use_deterministic=use_deterministic,
        drop_collided=drop_collided,
        with_std=with_std,
        seed=seed,
        rows_per_yield=rows_per_yield,
        maps=maps,
    )
    return (
        build_split_dataloader("train", batch_size, steps_per_epoch=steps_per_epoch, **shared),
        build_split_dataloader("val", batch_size, **shared),
    )

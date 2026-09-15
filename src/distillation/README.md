# Distillation datasets

`DistillationRecorder` dumps the observations and actions of a policy rollout so a student policy can be trained
offline. Produce one with:

```bash
env_isaaclab/bin/python evaluate_policy.py --num_envs=16 --max_episodes=20 --headless \
    --checkpoint=<name>.pt --distillation
```

Recording stops early once the buffered data reaches 10 GB. The file lands in
`distillation/<map_name>/<map_name>_<MM-DD_HH-MM>.pt` and holds a plain dict of tensors:

| key | shape | dtype | meaning |
| --- | --- | --- | --- |
| `<obs group>` | `(N, dim)` | float32 | one entry per group in the policy's `obs_groups`, e.g. `lidar`, `global_plan`, `action_buffer`, `velocity_buffer` |
| `action` | `(N, 3)` | float32 | action actually applied — the distribution mean unless `--stochastic` |
| `action_mean` | `(N, 3)` | float32 | mean of the policy distribution |
| `action_std` | `(N, 3)` | float32 | std of the policy distribution |
| `episode_id` | `(N,)` | int64 | episode each row belongs to, see below |
| `episode_index` | `(E,)` | int64 | every episode id in the file, sorted |
| `episode_collided` | `(E,)` | bool | whether that episode touched anything, aligned with `episode_index` |

`N = num_steps * num_envs`. Rows are ordered step-major: the first `num_envs` rows are the first step of every env,
the next `num_envs` rows the second step, and so on. All the per-row tensors share that ordering, so they can be
masked with a single index.

## Episode ids

An episode id packs the env it ran in and how many times that env had reset before it:

```
 31                    8 7        0
+-----------------------+----------+
|      run counter      |  env id  |
+-----------------------+----------+
```

```python
env_id = episode_id & 0xFF          # low 8 bits, so at most 256 envs
run = episode_id >> 8               # 0 for the first episode of that env, +1 on every reset
```

Ids are unique within a file but *not* across files — two runs of `evaluate_policy.py` both start counting from 0,
so offset them yourself before concatenating datasets.

## Filtering out collisions

An episode is flagged as collided if the contact force on the robot ever crossed `--collision_force_thresh`
(3 N by default) while it was running. The flag is sticky: one bump anywhere in the episode marks all of its rows,
including the steps before the contact happened.

```python
import torch

data = torch.load("distillation/kujiale_0036/kujiale_0036_08-15_09-44.pt")

clean = data["episode_index"][~data["episode_collided"]]     # episode ids worth keeping
mask = torch.isin(data["episode_id"], clean)                 # (N,) row mask

obs, actions = data["lidar"][mask], data["action"][mask]
```

Swap the `~` for nothing to train on the collisions instead, or skip the mask entirely to use everything. To slice a
single rollout — for plotting a trajectory, say — index on the id directly:

```python
rows = data["episode_id"] == ((7 << 8) | 3)    # env 3, its 8th episode
```

Episodes still running when recording stopped are in the table like any other, carrying the verdict they had
accumulated so far. Their rows are ordinary state-action pairs and are fine to train on, but the rollout is cut off
mid-episode, which matters if you need whole trajectories. The last recorded step holds the newest episode of every
env, so dropping those ids drops every truncated episode — plus, occasionally, one that happened to end on exactly
that step:

```python
num_envs = int((data["episode_id"] & 0xFF).max()) + 1
mask &= ~torch.isin(data["episode_id"], data["episode_id"][-num_envs:])
```

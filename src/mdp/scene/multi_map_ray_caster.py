import re
from types import SimpleNamespace

import isaaclab.sim as sim_utils
import numpy as np
import torch
import warp as wp
from isaaclab.cloner import ClonePlan
from isaaclab.sensors import MultiMeshRayCasterCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxManager
from isaaclab_physx.sensors.ray_caster.multi_mesh_ray_caster import MultiMeshRayCaster
from pxr import UsdPhysics

_ENV_ID_RE = re.compile(r"^/World/envs/env_(\d+)/")


class MultiMapRayCaster(MultiMeshRayCaster):
    """MultiMeshRayCaster whose tracked targets can differ from one environment to the next.

    Robots navigating different maps have different doors and chairs, both in number and in shape, while the upstream
    sensor expects every environment to hold the same tracked bodies. Tracked targets are therefore resolved here
    env by env: every match is followed through its own rigid body, and its warp mesh is shared by all the environments
    cloned from the same source prim. Static targets are left to the upstream implementation.
    """

    def _build_mesh_records(self, target_cfg: MultiMeshRayCasterCfg.RaycastTargetCfg, plan: ClonePlan | None, dummy_mesh_id):
        if not target_cfg.track_mesh_transforms:
            return super()._build_mesh_records(target_cfg, plan, dummy_mesh_id)

        stage = sim_utils.get_current_stage()
        has_rigid_body = lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI)  # noqa: E731
        clone_sources = self._clone_sources(plan)

        records_per_env = [[] for _ in range(self._num_envs)]
        tracked_bodies = []  # (env id, slot in the env's mesh table, rigid body path)
        for prim in sim_utils.find_matching_prims(target_cfg.prim_expr):
            env_id = int(_ENV_ID_RE.match(str(prim.GetPath())).group(1))
            body = sim_utils.get_first_matching_ancestor_prim(prim.GetPath(), has_rigid_body)
            if body is None:
                raise RuntimeError(f"Cannot track '{prim.GetPath()}', it has no rigid body ancestor")

            # load the mesh from the prim this one was cloned from, so that all its clones share one warp mesh
            source_prim = stage.GetPrimAtPath(clone_sources(env_id, str(prim.GetPath())))
            source_body = stage.GetPrimAtPath(clone_sources(env_id, str(body.GetPath())))
            mesh_id = self._load_target_prim_warp_mesh(source_prim, target_cfg, reference_prim=source_body)
            dummy_mesh_id = mesh_id if dummy_mesh_id is None else dummy_mesh_id

            # the pose is written by `_update_mesh_transforms`, park the mesh far away until then
            tracked_bodies.append((env_id, len(records_per_env[env_id]), str(body.GetPath())))
            records_per_env[env_id].append((mesh_id, (1.0e9, 1.0e9, 1.0e9), (0.0, 0.0, 0.0, 1.0)))

        # the upstream sensor hands this list over to `_create_tracked_target_view` untouched
        return records_per_env, dummy_mesh_id, tracked_bodies

    def _clone_sources(self, plan: ClonePlan | None):
        """Returns a function that maps a prim path of an environment to the path of the prim it was cloned from."""
        if plan is None:
            return lambda env_id, path: path

        clone_mask = plan.clone_mask.cpu().numpy()

        def source_path(env_id: int, path: str) -> str:
            for row in np.flatnonzero(clone_mask[:, env_id]):
                destination = plan.destinations[row].format(env_id)
                if path == destination or path.startswith(destination + "/"):
                    return plan.sources[row] + path[len(destination) :]
            return path

        return source_path

    def _create_tracked_target_view(self, tracked_bodies: list[tuple[int, int, str]]):
        """One rigid body view over every tracked body, together with the mesh table entry each of them updates."""
        if not tracked_bodies:
            return None

        env_ids, slots, body_paths = zip(*tracked_bodies)
        view = PhysxManager.get_physics_sim_view().create_rigid_body_view(list(body_paths))
        view_rows = {path: row for row, path in enumerate(view.prim_paths)}
        return SimpleNamespace(
            view=view,
            rows=torch.tensor([view_rows[path] for path in body_paths], dtype=torch.long, device=self._device),
            env_ids=torch.tensor(env_ids, dtype=torch.long, device=self._device),
            slots=torch.tensor(slots, dtype=torch.long, device=self._device),
        )

    def _update_mesh_transforms(self) -> None:
        if not hasattr(self, "_mesh_views"):
            return

        # torch views sharing memory with the (num_envs, num_meshes) warp tables
        positions = wp.to_torch(self._mesh_positions_w)
        orientations = wp.to_torch(self._mesh_orientations_w)

        mesh_offset = 0
        for tracked, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            if tracked is not None:
                transforms = wp.to_torch(tracked.view.get_transforms()).view(-1, 7)[tracked.rows]  # (x, y, z, qx, qy, qz, qw)
                positions[tracked.env_ids, mesh_offset + tracked.slots] = transforms[:, :3]
                orientations[tracked.env_ids, mesh_offset + tracked.slots] = transforms[:, 3:]
            mesh_offset += self._num_meshes_per_env[target_cfg.prim_expr]


@configclass
class MultiMapRayCasterCfg(MultiMeshRayCasterCfg):
    """Configuration for :class:`MultiMapRayCaster`."""

    class_type: type = MultiMapRayCaster

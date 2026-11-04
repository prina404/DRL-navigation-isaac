import re
from types import SimpleNamespace

import isaaclab.sim as sim_utils
import numpy as np
import torch
import trimesh
import warp as wp
from isaaclab.cloner import ClonePlan
from isaaclab.sensors import MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster.base_multi_mesh_ray_caster import (
    BaseMultiMeshRayCaster,
)
from isaaclab.utils import configclass
from isaaclab.utils.mesh import (
    PRIMITIVE_MESH_TYPES,
    create_trimesh_from_geom_mesh,
    create_trimesh_from_geom_shape,
)
from isaaclab.utils.warp import convert_to_warp_mesh
from isaaclab_physx.physics import PhysxManager
from isaaclab_physx.sensors.ray_caster.multi_mesh_ray_caster import MultiMeshRayCaster
from loguru import logger
from pxr import Gf, Usd, UsdGeom, UsdPhysics

_ENV_ID_RE = re.compile(r"^/World/envs/env_(\d+)/")


class MultiMapRayCaster(MultiMeshRayCaster):
    """MultiMeshRayCaster whose tracked targets can differ from one environment to the next.

    Robots navigating different maps have different doors and chairs, both in number and in shape, while the upstream
    sensor expects every environment to hold the same tracked bodies. Tracked targets are therefore resolved here
    env by env: every match is followed through its own rigid body, and its warp mesh is shared by all the environments
    cloned from the same source prim. The records of static targets are left to the upstream implementation, but the
    meshes of every target are built here (see `_load_target_prim_warp_mesh`).
    """

    def _load_target_prim_warp_mesh(self, target_prim: Usd.Prim, target_cfg, reference_prim: Usd.Prim | None = None) -> int:
        """Merge every mesh under `target_prim` into one warp mesh, in the frame of `reference_prim`.

        Upstream places each mesh with a positive scale (the row norms of its world matrix) and an orthonormalized
        quaternion, which cannot represent the mirror of a prim with a negative scale. The baked InteriorAgent scenes are
        full of mirrored parts, authored in millimetres far from their local origin, so upstream dropped them up to 5 m
        away in free space, where the lidar and the costmap saw walls that are not there (181 of the 1489 static meshes
        of kujiale_0037; 20 of the 23 maps have some). Each mesh is mapped through its full world matrix instead. The
        reference frame keeps the upstream convention: the pose table holds the reference prim's pose without its scale.
        For a mirrored reference that pose is the one PhysX gives the body, see below.
        """
        reference_prim = target_prim if reference_prim is None else reference_prim
        # own key, so a mesh is never shared with one placed by the upstream loader
        key = (f"{target_prim.GetPath()}@{reference_prim.GetPath()}:full_xform", self._device)
        if key in BaseMultiMeshRayCaster.meshes:
            return BaseMultiMeshRayCaster.meshes[key].id

        mesh_prims = sim_utils.get_all_matching_child_prims(
            target_prim.GetPath(), lambda prim: prim.GetTypeName() in PRIMITIVE_MESH_TYPES + ["Mesh"]
        )
        if not mesh_prims:
            raise RuntimeError(f"No mesh prims found at path: {target_prim.GetPath()}")
        if len(mesh_prims) > 1 and not target_cfg.merge_prim_meshes:
            raise RuntimeError(f"Multiple mesh prims found at path: {target_prim.GetPath()} but merging is disabled")

        xform_cache = UsdGeom.XformCache()
        reference_tf = xform_cache.GetLocalToWorldTransform(reference_prim)
        reference_tf.Orthonormalize()
        if reference_tf.GetDeterminant3() < 0:
            # PhysX factors a mirrored body into a negative scale and the rotation -R, so that is the pose the tracked
            # target reports. Against R, the mesh came out point-reflected through the body origin: the mirrored chairs
            # of kujiale_0032 and kujiale_0043 ended up upside down under the floor, out of the lidar's sight.
            reference_tf = Gf.Matrix4d().SetDiagonal(Gf.Vec4d(-1.0, -1.0, -1.0, 1.0)) * reference_tf
        world_to_reference = np.asarray(reference_tf.GetInverse(), dtype=np.float64)

        meshes = []
        for prim in mesh_prims:
            mesh = create_trimesh_from_geom_mesh(prim) if prim.GetTypeName() == "Mesh" else create_trimesh_from_geom_shape(prim)
            # USD uses the row vector convention, so transforms are applied on the right; trimesh takes the transpose
            to_reference = np.asarray(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float64) @ world_to_reference
            mesh.apply_transform(to_reference.T)
            meshes.append(mesh)
        merged = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]

        wp_mesh = convert_to_warp_mesh(merged.vertices, merged.faces, device=self._device)
        BaseMultiMeshRayCaster.meshes[key] = wp_mesh
        logger.info(
            f"Read {len(mesh_prims)} mesh prims under '{target_prim.GetPath()}': {len(merged.vertices)} vertices, "
            f"{len(merged.faces)} faces"
        )
        return wp_mesh.id

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

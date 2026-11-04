import isaaclab.sim as sim_utils
import numpy as np
import trimesh
from isaaclab.sensors import MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster.base_multi_mesh_ray_caster import (
    _ENV_PATH_RE,
    BaseMultiMeshRayCaster,
)
from isaaclab.utils import configclass
from isaaclab.utils.mesh import (
    PRIMITIVE_MESH_TYPES,
    create_trimesh_from_geom_mesh,
    create_trimesh_from_geom_shape,
)
from isaaclab.utils.warp import convert_to_warp_mesh
from isaaclab_physx.sensors.ray_caster.multi_mesh_ray_caster import MultiMeshRayCaster
from loguru import logger
from pxr import Gf, Usd, UsdGeom


class MirrorSafeRayCaster(MultiMeshRayCaster):
    """MultiMeshRayCaster that places mirrored (negative scale) meshes where they are.

    Only the mesh loader is replaced, see `_load_target_prim_warp_mesh`; the records of static and tracked targets are
    left to the upstream implementation. Ported from `MultiMapRayCaster` of the multi-env branch.
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

        Like upstream, an `is_shared` target (the static scene, cloned in every env) is loaded once and aliased by every
        environment: its vertices are relative to the reference prim, so they are the same in each clone.
        """
        reference_prim = target_prim if reference_prim is None else reference_prim
        # own key, so a mesh is never shared with one placed by the upstream loader
        key = (f"{target_prim.GetPath()}@{reference_prim.GetPath()}:full_xform", self._device)
        if key in BaseMultiMeshRayCaster.meshes:
            return BaseMultiMeshRayCaster.meshes[key].id

        shared_key = None
        if target_cfg.is_shared and self.cfg.reference_meshes:
            shared_key = (_ENV_PATH_RE.sub(r"\1env_0\2", key[0]), key[1])
            if shared_key in BaseMultiMeshRayCaster.meshes:
                BaseMultiMeshRayCaster.meshes[key] = BaseMultiMeshRayCaster.meshes[shared_key]
                return BaseMultiMeshRayCaster.meshes[shared_key].id

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
        if shared_key is not None:
            BaseMultiMeshRayCaster.meshes[shared_key] = wp_mesh
        logger.info(
            f"Read {len(mesh_prims)} mesh prims under '{target_prim.GetPath()}': {len(merged.vertices)} vertices, "
            f"{len(merged.faces)} faces"
        )
        return wp_mesh.id


@configclass
class MirrorSafeRayCasterCfg(MultiMeshRayCasterCfg):
    """Configuration for :class:`MirrorSafeRayCaster`."""

    class_type: type = MirrorSafeRayCaster

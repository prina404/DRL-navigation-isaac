#!/usr/bin/env bash
# Apply the local MultiMeshRayCaster fixes to a fresh IsaacLab clone.
# Usage: ./apply_isaaclab_patches.sh [path/to/IsaacLab]   (default: ../IsaacLab)
# Tested against IsaacLab v3.0.0-beta2; drop this once the fixes land upstream.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
patch_file="$here/fix_multi_mesh_raycaster.patch"
isaaclab_dir="${1:-$here/../IsaacLab}"

cd "$isaaclab_dir"

if git apply --reverse --check "$patch_file" 2>/dev/null; then
    echo "MultiMeshRayCaster patches already applied in $isaaclab_dir"
    exit 0
fi

# Plain apply first (works on shallow clones too); --3way merges if context drifted
# after an IsaacLab bump, as long as the surrounding code did not change.
git apply "$patch_file" 2>/dev/null || git apply --3way "$patch_file"
echo "Applied MultiMeshRayCaster patches to $isaaclab_dir"

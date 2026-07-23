#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
repository_root="$(cd -- "$script_dir/.." && pwd)"
image="${SIMBC_IMAGE:-$repository_root/simbc-cu128.sif}"
checkpoint_dir="${SIMBC_CHECKPOINT_DIR:-$repository_root/checkpoints}"
output_dir="${SIMBC_OUTPUT_DIR:-$repository_root/results}"
use_nv=1

if [[ "${1:-}" == "--cpu" ]]; then
    use_nv=0
    shift
fi
if [[ ! -f "$image" ]]; then
    echo "Container image not found: $image" >&2
    echo "Build it with: apptainer build --fakeroot simbc-cu128.sif apptainer/simbc.def" >&2
    exit 2
fi
if [[ "$#" -eq 0 ]]; then
    set -- /bin/bash
fi

mkdir -p "$checkpoint_dir" "$output_dir"

options=(
    --cleanenv
    --env MPLBACKEND=Agg
    --env MUJOCO_GL=egl
    --pwd /workspace
    --bind "$repository_root:/workspace"
    --bind "$checkpoint_dir:/checkpoints"
    --bind "$output_dir:/results"
)
if [[ -n "${SIMBC_DATA_DIR:-}" ]]; then
    options+=(--bind "$SIMBC_DATA_DIR:/data")
fi
if [[ "$use_nv" -eq 1 ]]; then
    options=(--nv "${options[@]}")
fi

exec apptainer exec "${options[@]}" "$image" "$@"

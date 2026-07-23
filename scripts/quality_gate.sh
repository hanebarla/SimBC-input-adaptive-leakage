#!/usr/bin/env bash
set -euo pipefail

export PRE_COMMIT_HOME="${PRE_COMMIT_HOME:-/tmp/simbc-pre-commit-cache}"

test_files=(tests/*.py)

python_files=(
    robo_manip_baselines/envs/mujoco/MujocoEnvBase.py
    robo_manip_baselines/envs/mujoco/task_metrics.py
    robo_manip_baselines/envs/mujoco/ur5e/MujocoUR5eCableEnv.py
    robo_manip_baselines/envs/mujoco/ur5e/MujocoUR5eParticleEnv.py
    robo_manip_baselines/envs/mujoco/ur5e/MujocoUR5eRingEnv.py
    robo_manip_baselines/sarnn/bin/TrainSarnn.py
    robo_manip_baselines/sarnn/bin/test.py
    robo_manip_baselines/sarnn/bin/rollout/RolloutSarnnMujocoUR5eCable.py
    robo_manip_baselines/sarnn/bin/rollout/RolloutSarnnMujocoUR5eParticle.py
    robo_manip_baselines/sarnn/bin/rollout/RolloutSarnnMujocoUR5eRing.py
    robo_manip_baselines/sarnn/lib/LifController.py
    robo_manip_baselines/sarnn/lib/RolloutSarnn.py
    robo_manip_baselines/sarnn/lib/experiment_config.py
    robo_manip_baselines/sarnn/lib/model_utils.py
    robo_manip_baselines/sarnn/scripts/run_rollout_jobs.py
    robo_manip_baselines/sarnn/scripts/run_cable_rollout_jobs.py
    robo_manip_baselines/sarnn/scripts/run_particle_rollout_jobs.py
    robo_manip_baselines/sarnn/scripts/run_ring_rollout_jobs.py
    robo_manip_baselines/utils/dataset_split.py
    robo_manip_baselines/utils/make_dataset.py
    scripts/smoke_test.py
    scripts/validate_checkpoints.py
    "${test_files[@]}"
)
public_files=(
    "${python_files[@]}"
    .gitignore
    MODIFICATIONS.md
    README.md
    apptainer/simbc.def
    artifacts.yaml
    doc/reproduction.md
    pyproject.toml
    robo_manip_baselines/sarnn/README.md
    scripts/apptainer_exec.sh
    scripts/quality_gate.sh
)

python -m compileall "${python_files[@]}"
ruff format --check "${python_files[@]}"
ruff check "${python_files[@]}"
pre-commit run --files "${public_files[@]}"

if rg -n '(thabara|/home/thabara|/ldisk/habara)' \
    --glob '!third_party/**' \
    --glob '!scripts/quality_gate.sh' \
    .; then
    echo "Fork maintainer personal path found." >&2
    exit 1
fi

if rg -n 'BEGIN (RSA|OPENSSH|EC) PRIVATE KEY' \
    --glob '!third_party/**' \
    --glob '!scripts/quality_gate.sh' \
    .; then
    echo "Private-key marker found." >&2
    exit 1
fi

from pathlib import Path

import numpy as np
import pytest

from robo_manip_baselines.envs.mujoco.ur5e.MujocoUR5eClothEnv import (
    get_cloth_position_offsets,
)
from robo_manip_baselines.misc.RunSarnnExperiment import (
    CLOTH_WORLD_INDICES,
    _check_output_collisions,
    build_jobs,
    main,
)


def test_cloth_default_and_evaluation_offsets():
    default_offsets = get_cloth_position_offsets(False)
    evaluation_offsets = get_cloth_position_offsets(True)

    assert default_offsets.shape == (6, 3)
    assert evaluation_offsets.shape == (21, 3)
    np.testing.assert_allclose(default_offsets[:, 1], np.linspace(-0.12, 0.08, 6))
    np.testing.assert_allclose(evaluation_offsets[:, 1], np.linspace(-0.12, 0.08, 21))
    np.testing.assert_allclose(evaluation_offsets[:, (0, 2)], 0.0)


def test_runner_builds_canonical_three_way_matrix(tmp_path):
    checkpoint = tmp_path / "checkpoint" / "policy_best.ckpt"
    jobs = build_jobs(checkpoint, tmp_path / "results")

    assert [job.name for job in jobs] == [
        "qcfs_ann",
        "qcfs_snn_t4",
        "input_adaptive_t2",
    ]
    for job in jobs:
        command = list(job.command)
        world_start = command.index("--world_idx_list") + 1
        world_end = command.index("--use_test_offset")
        assert tuple(map(int, command[world_start:world_end])) == CLOTH_WORLD_INDICES
        assert command[0]
        assert "taskset" not in command
        assert job.result_path == (
            tmp_path / "results" / "cloth" / job.name / "result.yaml"
        )
        assert job.log_path.name == "rollout.log"

    ann, conventional, proposed = (list(job.command) for job in jobs)
    assert ann[ann.index("--qcfs_T") + 1] == "0"
    assert conventional[conventional.index("--qcfs_T") + 1] == "4"
    assert proposed[proposed.index("--qcfs_T") + 1] == "2"
    assert proposed[proposed.index("--la_rb_k") + 1] == "32"
    assert proposed[proposed.index("--la_rb_thresh") + 1] == "0.03"
    assert "--no_reset" in proposed
    assert proposed[proposed.index("--spike_dec_type") + 1] == "decay"


def test_runner_protects_existing_output(tmp_path):
    jobs = build_jobs(Path("policy.ckpt"), tmp_path)
    jobs[0].result_path.parent.mkdir(parents=True)
    jobs[0].result_path.write_text("success: []\n")

    with pytest.raises(FileExistsError, match="--overwrite"):
        _check_output_collisions(jobs, overwrite=False)
    _check_output_collisions(jobs, overwrite=True)


def test_runner_dry_run_has_no_output_side_effects(tmp_path, capsys):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "policy_best.ckpt"
    checkpoint.write_bytes(b"")
    (checkpoint_dir / "model_meta_info.pkl").write_bytes(b"")
    output_dir = tmp_path / "output"

    assert (
        main(
            [
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output_dir),
                "--gpus",
                "cpu",
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert output.count("CUDA_VISIBLE_DEVICES=cpu") == 3
    assert not output_dir.exists()

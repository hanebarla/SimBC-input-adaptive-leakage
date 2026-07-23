from argparse import Namespace
from pathlib import Path

import pytest

from robo_manip_baselines.sarnn.lib.experiment_config import (
    DEFAULT_TASK,
    TASK_CONFIGS,
)
from robo_manip_baselines.sarnn.scripts.run_rollout_jobs import (
    build_command,
    build_jobs,
    main,
)


def test_default_batch_task_is_ring():
    jobs = build_jobs()
    assert len(jobs) == 63
    assert {job.task for job in jobs} == {DEFAULT_TASK}
    assert DEFAULT_TASK == "ring"


@pytest.mark.parametrize(
    ("task", "timesteps"),
    [
        ("ring", (0, 8, 1)),
        ("particle", (0, 4, 2)),
        ("cable", (0, 4, 4)),
    ],
)
def test_batch_matrix_has_63_jobs_and_task_timesteps(task, timesteps):
    jobs = build_jobs(task)
    assert len(jobs) == 63
    assert tuple(job.variant.qcfs_t for job in jobs[::21]) == timesteps
    assert {job.world_idx for job in jobs} == set(range(21))


def test_batch_command_has_no_personal_absolute_path():
    args = Namespace(
        checkpoint=Path("checkpoints/SARNN.pth"),
        output_dir=Path("results"),
        cropped_img_size=480,
        skip=6,
        overwrite=False,
        record_video=False,
        save_diagnostics=False,
    )
    command = build_command(args, TASK_CONFIGS["particle"], build_jobs("particle")[42])
    rendered = " ".join(command)
    assert "/home/" not in rendered
    assert "--qcfs_T 2" in rendered
    assert "--LA_RB_thresh 0.03" in rendered


@pytest.mark.parametrize("task", ("cable", "particle", "ring"))
def test_task_cli_dry_run_prints_exactly_63_portable_jobs(task, capsys):
    assert (
        main(
            task,
            [
                "--checkpoint",
                "checkpoints/SARNN.pth",
                "--output-dir",
                "results",
                "--dry-run",
            ],
        )
        == 0
    )
    commands = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("CUDA_DEVICE_ORDER=")
    ]
    assert len(commands) == 63
    assert all("/home/" not in command for command in commands)

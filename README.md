# SimBC Input-Adaptive Leakage

This repository is a research fork of
[RoboManipBaselines](https://github.com/isri-aist/RoboManipBaselines) for the
SimBC input-adaptive leakage study. It preserves the upstream manipulation
policies and environments while adding a reusable SARNN/QCFS inference path.

The current `input-adaptive-leakage-v3based` branch contains:

- QCFS ANN inference (`T=0`) and conventional QCFS SNN conversion (`T>0`)
- the proposed rule-based input-adaptive leakage with centered membrane
  leakage, no-reset state, and decay firing-rate decoding
- a 21-condition Cloth evaluation and a three-variant experiment runner
- shared QCFS and input-adaptive leakage CLI options for other SARNN tasks

Cloth is the task with a experiment matrix on this branch.
Task-specific experiment configurations for the other manipulation tasks will
be added to the same `input-adaptive-leakage-v3based` branch.

## Reproduce the experiments

Apptainer is the reference environment. Build and smoke-test it from the
repository root:

```bash
git submodule update --init third_party/eipl
apptainer build --fakeroot simbc-cu128.sif apptainer/simbc.def
./scripts/apptainer_exec.sh --cpu python scripts/smoke_test.py --cpu
./scripts/apptainer_exec.sh python scripts/smoke_test.py --gpu
```

The image pins Ubuntu 22.04, Python 3.10, CUDA 12.8, PyTorch 2.11.0, and
torchvision 0.26.0. The build host must have subordinate UID/GID mappings
configured for `--fakeroot`; if they are unavailable, ask the system
administrator to build the SIF. Running an existing SIF does not require
fakeroot.

The launcher enables NVIDIA passthrough and EGL by default. Use `--cpu` to
disable `--nv`. Set `SIMBC_IMAGE` to use a differently named SIF,
`SIMBC_DATA_DIR` to mount a host dataset at `/data`,
`SIMBC_CHECKPOINT_DIR` to mount checkpoints at `/checkpoints`, and
`SIMBC_OUTPUT_DIR` to mount results at `/results`. The checkpoint and output
binds default to `./checkpoints` and `./results`.

See the
[SARNN/QCFS experiment guide](robo_manip_baselines/policy/sarnn/README.md)
for the exact training and individual rollout commands.

## Cloth experiment

The Cloth evaluation shifts the cloth and board along the Y axis from
`-0.12 m` to `0.08 m` in `0.01 m` increments, giving 21 initial conditions.
The canonical comparison is:

| Variant | Configuration |
| --- | --- |
| `qcfs_ann` | `T=0` |
| `qcfs_snn_t3` | conventional SNN, `T=3`, reset each inference |
| `input_adaptive_t4` | proposed method, `T=4`, `k=32`, threshold `0.03`, no reset, decay decoder |

Run all three variants with one checkpoint:

```bash
./scripts/apptainer_exec.sh python \
  robo_manip_baselines/misc/RunSarnnExperiment.py \
  --checkpoint /checkpoints/<checkpoint-name>/policy_best.ckpt \
  --output-dir /results \
  --gpus 0
```

Use `--dry-run` to inspect commands, `--gpus 0 1` and
`--processes-per-gpu` to schedule multiple GPUs, and `--overwrite` only when
replacing existing outputs. The runner writes
`/results/cloth/<variant>/result.yaml` and
`/results/cloth/<variant>/rollout.log`. The checkpoint directory must contain
both the selected `.ckpt` and the matching `model_meta_info.pkl`.

## Citation

### SimBC input-adaptive leakage

The bibliographic record for the SimBC input-adaptive leakage paper is not yet
final. Replace every `TODO` field below before publication; this entry must not
be treated as the final citation.

```bibtex
@article{TODO_simbc_input_adaptive_leakage,
  author  = {TODO},
  title   = {TODO: SimBC Input-Adaptive Leakage},
  journal = {TODO},
  year    = {TODO},
  volume  = {TODO},
  number  = {TODO},
  pages   = {TODO},
  doi     = {TODO},
  url     = {TODO}
}
```

### RoboManipBaselines

Please also cite the original RoboManipBaselines paper:

```bibtex
@article{RoboManipBaselines_Murooka_2025,
  title={RoboManipBaselines: A Unified Framework for Imitation Learning in Robotic Manipulation across Real and Simulated Environments},
  author={Murooka, Masaki and Motoda, Tomohiro and Nakajo, Ryoichi and Oh, Hanbit and Makihara, Koshi and Shirai, Keisuke and Domae, Yukiyasu},
  journal={arXiv preprint arXiv:2509.17057},
  year={2025}
}
```

The original software citation remains available in [CITATION.cff](CITATION.cff).
Please also retain the SARNN citation in the
[SARNN/QCFS experiment guide](robo_manip_baselines/policy/sarnn/README.md).

## Upstream project and licensing

RoboManipBaselines authorship, environments, policies, and BSD-2-Clause
licensing are retained. The EIPL submodule and input-adaptive spiking
extensions are distributed under AGPL-3.0; file-level notices and
`third_party/eipl/LICENSE` take precedence where applicable.

For the full upstream documentation, visit the
[RoboManipBaselines project page](https://isri-aist.github.io/RoboManipBaselines-ProjectPage)
or the
[upstream repository](https://github.com/isri-aist/RoboManipBaselines).

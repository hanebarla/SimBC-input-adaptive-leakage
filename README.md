# SimBC Input-Adaptive Leakage

This repository is a research fork of
[RoboManipBaselines](https://github.com/isri-aist/RoboManipBaselines) for the
Ring, Particle, and Cable simulation experiments accompanying the SimBC
input-adaptive leakage study. It preserves the upstream policies and
environments while adding a focused, reproducible SARNN/QCFS evaluation path.

The public scope contains:

- QCFS ANN inference (`T=0`)
- conventional QCFS SNN conversion
- the proposed rule-based input-adaptive leakage with centered membrane
  leakage, no-reset state, and decay firing-rate decoding
- 21 MuJoCo conditions per task, structured metrics, and lightweight results

Ring is the default task used by generic commands and documentation examples;
Particle and Cable remain fully supported through their task-specific paths.

Experiments for Cloth and additional tasks are planned on a shared branch:

- Cloth and additional tasks: `<CLOTH_AND_OTHER_TASKS_BRANCH>`

This branch name is a placeholder and will be replaced when the branch is
published.

AdaFire/SSC, IAT, full-spiking prototypes, and the incomplete open-loop
evaluator are intentionally not included.

## Reproduce the experiments

Apptainer is the reference environment. Build and smoke-test it from the
repository root:

```bash
git submodule update --init third_party/eipl
apptainer build --fakeroot simbc-cu128.sif apptainer/simbc.def
./scripts/apptainer_exec.sh --cpu python scripts/smoke_test.py --cpu
./scripts/apptainer_exec.sh python scripts/smoke_test.py --gpu
```

The build host must have Apptainer subordinate-ID mappings configured for
`--fakeroot`; otherwise an administrator-built SIF is required.

The image pins Ubuntu 22.04, Python 3.10, CUDA 12.8, PyTorch 2.11.0, and
torchvision 0.26.0. See the [reproduction guide](doc/reproduction.md) for
dataset preparation, training, the 63-job task matrices, output schemas, and
validation commands.

The Ring, Particle, and Cable demonstrations are obtained from the upstream
[RoboManipBaselines dataset list](doc/dataset_list.md). URLs, checkpoint
digests, and the fixed EIPL commit are recorded in
[artifacts.yaml](artifacts.yaml).

## Results

Rollout requires both `--checkpoint` and `--output_dir`. A per-world run
writes:

- `result.json`: configuration, checkpoint SHA-256, runtime/model metadata,
  task metric, timing, and spike statistics
- `trace.npz`: frame indices, inference timing, leakage and frame-difference
  histories, cumulative layer spikes, and attention points

Videos and diagnostic image arrays are opt-in. Batch runs additionally write
task-level `summary.json` and `summary.csv`. Existing results are protected
unless `--overwrite` is given, and no public output uses pickle.

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

Please also cite the original RoboManipBaselines software:

```bibtex
@software{RoboManipBaselines_GitHub2024,
  author  = {Murooka, Masaki and Motoda, Tomohiro and Nakajo, Ryoichi},
  title   = {{RoboManipBaselines}},
  url     = {https://github.com/isri-aist/RoboManipBaselines},
  version = {1.0.0},
  year    = {2024},
  month   = dec,
}
```

This entry is also available in [CITATION.cff](CITATION.cff). Retain the SARNN
citation in the
[SARNN documentation](robo_manip_baselines/sarnn/README.md).

## Upstream project and licensing

RoboManipBaselines authorship, citations, environments, policies, and
BSD-2-Clause licensing are retained. The fixed EIPL fork and the spiking
extensions are AGPL-3.0; file-level notices and
`third_party/eipl/LICENSE` take precedence where applicable. See
[MODIFICATIONS.md](MODIFICATIONS.md) for fork provenance.

For questions about this fork, open an issue in this repository.

# Reproduction guide

## Scope

The public experiment covers Ring, Particle, and Cable with three
checkpoint execution modes: QCFS ANN, conventional QCFS SNN, and QCFS SNN
with input-adaptive leakage. All trained models use front RGB images and
actions, a sequence mask, no BatchNorm, 128 x 128 inputs, `qcfs_L=8`,
`k_dim=20`, and `rec_dim=50`.

## Container

Build from the repository root:

```bash
git submodule update --init third_party/eipl
apptainer build --fakeroot simbc-cu128.sif apptainer/simbc.def
./scripts/apptainer_exec.sh --cpu python scripts/smoke_test.py --cpu
./scripts/apptainer_exec.sh python scripts/smoke_test.py --gpu
```

`apptainer build --fakeroot` requires subordinate UID/GID mappings on the
build host. If they are unavailable, ask the system administrator to build the
SIF; runtime use does not require that build privilege.

The definition uses Ubuntu 22.04, Python 3.10, CUDA 12.8, PyTorch 2.11.0, and
torchvision 0.26.0 from the cu128 index. CUDA 12.8 is selected for Blackwell
support according to the
[NVIDIA compatibility matrix](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html).
The wheel versions follow the
[official PyTorch table](https://pytorch.org/get-started/previous-versions/).

Set `SIMBC_IMAGE` to use a differently named SIF, `SIMBC_DATA_DIR` to bind
a host dataset directory at `/data`, `SIMBC_CHECKPOINT_DIR` for
`/checkpoints`, and `SIMBC_OUTPUT_DIR` for `/results`. The checkpoint and
output binds default to ignored directories in the repository. The launcher
enables `--nv` and EGL by default. Host pip installation is a development
convenience, not the reference environment.

## Data preparation

Download the three 30-demonstration datasets through [the upstream data
list](dataset_list.md) or the machine-readable [asset manifest](../artifacts.yaml).
The commands below use Ring as the default example. Extract it under
`/data/raw/ring`, then run:

```bash
./scripts/apptainer_exec.sh python robo_manip_baselines/utils/make_dataset.py \
  --in_dir /data/raw/ring --out_dir /data/processed/ring \
  --train_ratio 0.8 --split_seed 0 --cropped_img_size 480 \
  --resized_img_size 128 --skip 6
```

For 30 sources this creates 24 train and 6 test sequences. The command writes
`dataset_manifest.json` with the deterministic split and preprocessing
provenance. The original experiments did not store a split seed, so newly
trained runs are deterministic but cannot reconstruct that historical split
bit-for-bit.

## Training

The public front-image/action configuration, using Ring by default, is:

```bash
./scripts/apptainer_exec.sh python robo_manip_baselines/sarnn/bin/TrainSarnn.py \
  --data_dir /data/processed/ring --log_dir /results/train/ring \
  --no_side_image --no_wrench --with_mask --im_size 128 128 \
  --qcfs --qcfs_L 8 --qcfs_T 0 --wobn --k_dim 20 --rec_dim 50
```

Adjust the task dataset and optimization settings as needed. The historical
Cable final run additionally loaded an unpublished L=12 checkpoint through
`--pretrained_model_path`; therefore that final training lineage cannot yet
be reproduced from public weights. The manifest is structured so a weight URL
can be added later without changing the workflow.

## Offline checkpoint check

```bash
./scripts/apptainer_exec.sh python robo_manip_baselines/sarnn/bin/test.py \
  --checkpoint /checkpoints/ring/SARNN.pth --data_dir /data/processed/ring \
  --qcfs_T 8 --output /results/offline-ring.npz --device cuda
```

The loader fills defaults missing from legacy `args.json` files and is shared
with rollout.

## Evaluation matrix

Each task wrapper creates exactly 63 jobs: three variants by worlds 0 through
20. The default example runs Ring; use the corresponding task wrapper and
checkpoint directory for Particle or Cable.

```bash
./scripts/apptainer_exec.sh python \
  robo_manip_baselines/sarnn/scripts/run_ring_rollout_jobs.py \
  --checkpoint /checkpoints/ring/SARNN.pth --output-dir /results
```

Use `--dry-run` to inspect commands, `--cuda-visible-devices 0,1` and
`--processes-per-gpu` to schedule GPUs, and `--overwrite` only when replacing
existing results. Videos and diagnostic image arrays require
`--record-video` and `--save-diagnostics`, respectively.

The canonical timesteps and controller parameters are:

| Task | QCFS ANN | Conventional SNN | Input-adaptive leakage |
| --- | --- | --- | --- |
| Ring | T=0 | T=8 | T=1, k=32, threshold=0.03 |
| Particle | T=0 | T=4 | T=2, k=32, threshold=0.03 |
| Cable | T=0 | T=4 | T=4, k=128, threshold=0 |

Per-world results are written below
`<output>/<task>/<variant>/world_<index>/`. Default files are `result.json`
and `trace.npz`; task-level `summary.json` and `summary.csv` are generated
after a batch. No output uses pickle.

## Validation

```bash
./scripts/apptainer_exec.sh --cpu pytest -q third_party/eipl/tests tests
./scripts/apptainer_exec.sh --cpu ./scripts/quality_gate.sh
```

The quality gate scopes linting to this fork's public additions, leaving
unmodified upstream policies and the fixed EIPL submodule untouched. GPU/EGL
validation should additionally run world 0 for all nine task/variant
combinations before a release.

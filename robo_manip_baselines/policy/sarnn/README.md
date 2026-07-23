# SARNN/QCFS experiments

This directory contains the reusable QCFS and input-adaptive leakage extension.
Ordinary SARNN defaults and checkpoint loading remain unchanged.

## Install

See [here](../../../doc/install.md#sarnn) for installation.

## Dataset preparation

Collect demonstration data by [teleoperation](../../teleop).

## Model training

Train a model:

```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Train.py Sarnn --dataset_dir ./dataset/<dataset_name> --checkpoint_dir ./checkpoint/Sarnn/<checkpoint_name>
```
The `--image_crop_size_list` option should be specified appropriately for each task.

To train the QCFS model used by the Cloth experiments without changing the
defaults of ordinary SARNN training, specify all reproduction settings explicitly:

```console
$ python ./bin/Train.py Sarnn \
    --dataset_dir ./dataset/<cloth_dataset> \
    --checkpoint_dir ./checkpoint/Sarnn/<checkpoint_name> \
    --use_qcfs \
    --image_crop_size_list 480 480 \
    --image_size_list 128 128 \
    --num_attentions 20 \
    --qcfs_L 8 \
    --qcfs_T 0
```

Training with `--qcfs_T 0` produces the QCFS ANN checkpoint that is converted
to ANN or SNN mode at rollout time.

## Policy rollout

Run a trained policy:

```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Rollout.py Sarnn MujocoUR5eCable --checkpoint ./checkpoint/Sarnn/<checkpoint_name>/policy_last.ckpt
```

### Cloth QCFS comparison

The Cloth evaluation uses 21 initial positions, with the cloth and board shifted
from `-0.12` m to `0.08` m along the Y axis in `0.01` m increments. The three
canonical variants are:

| Variant | Configuration |
| --- | --- |
| QCFS ANN | `T=0` |
| Conventional SNN | `T=3`, reset each inference |
| Input-adaptive SNN | `T=4`, no reset, `k=32`, difference threshold `0.03` |

Run an individual variant with all 21 conditions as follows. Replace
`<variant-options>` with one of the option sets below.

```console
$ python ./bin/Rollout.py Sarnn MujocoUR5eCloth \
    --checkpoint ./checkpoint/Sarnn/<checkpoint_name>/policy_best.ckpt \
    --world_idx_list 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
    --use_test_offset \
    --no_render \
    --no_plot \
    --auto_exit \
    --result_filename ./result.yaml \
    <variant-options>
```

```console
# QCFS ANN
--qcfs_T 0

# Conventional SNN
--qcfs_T 3 --spike_dec_type default

# Input-adaptive SNN
--qcfs_T 4 --no_reset --lif_cnt rule --la_rb_k 32 \
    --la_rb_thresh 0.03 --spike_dec_type decay
```

The rule-based controller computes leakage from the normalized first-camera
frame difference. It requires an SNN checkpoint configuration, `--no_reset`,
and the decay spike decoder; invalid combinations are rejected at startup.

### Cloth experiment runner

The runner executes all three variants and writes one result YAML and one log
per variant. The checkpoint directory must also contain the matching
`model_meta_info.pkl`.

```console
$ python ./misc/RunSarnnExperiment.py \
    --checkpoint ./checkpoint/Sarnn/<checkpoint_name>/policy_best.ckpt \
    --output-dir ./results \
    --gpus 0
```

Outputs are stored under:

```text
<output-dir>/cloth/<variant>/result.yaml
<output-dir>/cloth/<variant>/rollout.log
```

Use `--dry-run` to inspect commands, `--gpus 0 1` to distribute variants,
`--processes-per-gpu` to set concurrency, and `--overwrite` to replace existing
results. Passing `--gpus cpu` disables CUDA.

The QCFS policy and rollout options are shared by other SARNN manipulation
tasks. Cloth-specific evaluation settings are implemented here first; the
experiment configurations for Cloth and the other tasks will continue to be
developed together on the `input-adaptive-leakage-v3based` branch.

## Citation

For the technical details of SARNN, please cite:

```bibtex
@INPROCEEDINGS{SARNN_ICRA2022,
  author = {Ichiwara, Hideyuki and Ito, Hiroshi and Yamamoto, Kenjiro and Mori, Hiroki and Ogata, Tetsuya},
  title = {Contact-Rich Manipulation of a Flexible Object based on Deep Predictive Learning using Vision and Tactility},
  booktitle = {International Conference on Robotics and Automation},
  year = {2022},
  pages = {5375-5381},
  doi = {10.1109/ICRA46639.2022.9811940}
}
```

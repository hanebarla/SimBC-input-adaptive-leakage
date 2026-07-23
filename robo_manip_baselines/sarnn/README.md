# SARNN/QCFS experiments

This fork retains the upstream Spatial Attention Recurrent Neural Network
(SARNN) implementation and adds the public QCFS and input-adaptive leakage
experiments for Ring, Particle, and Cable.

## Public configuration

The released training path uses front images and actions with masks, no side
image, no wrench, no BatchNorm, 128 x 128 images, `qcfs_L=8`, `k_dim=20`,
and `rec_dim=50`. Dataset preparation uses a centered 480-pixel crop, resize
to 128, temporal skip 6, and a seed-fixed 24/6 split for the 30 source
demonstrations. The commands below use Ring as the default task example.

```bash
python ../utils/make_dataset.py \
  --in_dir /data/raw/ring --out_dir /data/processed/ring \
  --train_ratio 0.8 --split_seed 0

python bin/TrainSarnn.py \
  --data_dir /data/processed/ring --log_dir /results/train/ring \
  --no_side_image --no_wrench --with_mask --im_size 128 128 \
  --qcfs --qcfs_L 8 --qcfs_T 0 --wobn --k_dim 20 --rec_dim 50
```

Use the task-specific scripts in `scripts/run_*_rollout_jobs.py` for the
canonical 63-job matrices. The full procedure and output schema are in the
[reproduction guide](../../doc/reproduction.md).

## SARNN citation

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

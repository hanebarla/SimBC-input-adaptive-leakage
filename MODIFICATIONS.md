# Fork provenance and modifications

This repository is a research fork of
[RoboManipBaselines](https://github.com/isri-aist/RoboManipBaselines). The
upstream environments, policies, documentation, authorship, BSD-2-Clause
license, and citation metadata are retained.

The fork adds the experiment code for QCFS ANN-to-SNN conversion and
input-adaptive leakage in the Ring, Particle, and Cable MuJoCo tasks. It also
adds deterministic preprocessing, 21-condition evaluation, task metrics,
structured result files, batch execution, tests, and an Apptainer environment.

The EIPL dependency is a fixed submodule at commit
`5b09691a9b43e8b1bf61aee3ac718c49c23e2c2e`. EIPL-derived and newly added
spiking files retain their AGPL-3.0 notices. Refer to each file and
`third_party/eipl/LICENSE` when redistributing.

This public snapshot intentionally excludes experimental AdaFire/SSC, IAT,
full-spiking prototypes, and the incomplete open-loop evaluator. Large
datasets, logs, videos, and learned weights are not versioned.

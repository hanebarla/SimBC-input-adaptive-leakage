"""Deterministic source-level dataset splitting utilities."""

from __future__ import annotations

import random


def validate_split(train_indices, test_indices):
    """Validate non-empty, disjoint source index collections."""

    train_indices = list(train_indices)
    test_indices = list(test_indices)
    if not train_indices or not test_indices:
        raise ValueError("The dataset split must contain both train and test samples")
    overlap = set(train_indices) & set(test_indices)
    if overlap:
        raise ValueError(
            "Train and test splits overlap at source indices "
            + ", ".join(str(index) for index in sorted(overlap))
        )
    return train_indices, test_indices


def deterministic_split(num_sources, train_ratio=0.8, test_ratio=None, seed=0):
    """Split source indices with a local RNG and no train/test overlap."""

    if num_sources < 2:
        raise ValueError("At least two source demonstrations are required")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be strictly between zero and one")
    if test_ratio is not None and not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be strictly between zero and one")

    shuffled = list(range(num_sources))
    random.Random(seed).shuffle(shuffled)
    train_length = max(int(train_ratio * num_sources), 1)
    if test_ratio is None:
        test_length = num_sources - train_length
    else:
        test_length = max(int(test_ratio * num_sources), 1)
        if train_length + test_length > num_sources:
            raise ValueError("train_ratio and test_ratio create overlapping splits")

    train_indices = shuffled[:train_length]
    test_indices = shuffled[train_length : train_length + test_length]
    return validate_split(train_indices, test_indices)

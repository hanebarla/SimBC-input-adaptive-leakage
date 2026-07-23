import pytest

from robo_manip_baselines.utils.dataset_split import deterministic_split


def test_thirty_sources_create_a_deterministic_24_6_split():
    first_train, first_test = deterministic_split(30, train_ratio=0.8, seed=7)
    second_train, second_test = deterministic_split(30, train_ratio=0.8, seed=7)
    assert (first_train, first_test) == (second_train, second_test)
    assert len(first_train) == 24
    assert len(first_test) == 6
    assert set(first_train).isdisjoint(first_test)
    assert set(first_train + first_test) == set(range(30))


def test_invalid_or_overlapping_ratios_are_rejected():
    with pytest.raises(ValueError):
        deterministic_split(30, train_ratio=1.0)
    with pytest.raises(ValueError):
        deterministic_split(30, train_ratio=0.8, test_ratio=0.3)

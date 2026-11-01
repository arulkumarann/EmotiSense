import numpy as np
import pytest

from emotisense.data.sampler import BucketBatchSampler


def test_every_index_once_per_epoch_and_batch_sizes():
    lengths = np.random.default_rng(0).uniform(1, 10, 103)
    sampler = BucketBatchSampler(lengths, 8, drop_last=False, seed=1)
    batches = list(sampler)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(103)) and len(batches) == len(sampler) == 13
    assert all(len(b) == 8 for b in batches[:-1]) or sum(len(b) < 8 for b in batches) == 1


def test_drop_last_drops_only_the_remainder():
    sampler = BucketBatchSampler(np.arange(50.0), 8, drop_last=True)
    batches = list(sampler)
    assert len(batches) == len(sampler) == 6 and all(len(b) == 8 for b in batches)
    assert len({i for b in batches for i in b}) == 48


def test_batches_have_similar_lengths_and_less_padding():
    lengths = np.random.default_rng(1).uniform(1, 12, 2000)
    bucketed = list(BucketBatchSampler(lengths, 16, pool_batches=50, seed=0))
    rng = np.random.default_rng(2)
    shuffled = np.array_split(rng.permutation(2000)[:1984], 124)
    pad = lambda groups: sum(lengths[list(g)].max() * len(g) - lengths[list(g)].sum() for g in groups)
    assert pad(bucketed) < 0.35 * pad(shuffled)          # far less padding than random batches


def test_order_changes_each_epoch_but_is_reproducible():
    lengths = np.random.default_rng(2).uniform(1, 10, 200)
    a, b = BucketBatchSampler(lengths, 8, seed=7), BucketBatchSampler(lengths, 8, seed=7)
    first_a, second_a, first_b = list(a), list(a), list(b)
    assert first_a == first_b and first_a != second_a


def test_unshuffled_is_deterministic_and_covers_everything():
    s = BucketBatchSampler(np.arange(20.0), 5, shuffle=False, drop_last=False)
    assert list(s) == list(s.__class__(np.arange(20.0), 5, shuffle=False, drop_last=False))
    assert sorted(i for b in s for i in b) == list(range(20))

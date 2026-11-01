"""Length-bucketed batching: batches of similar-length clips waste far less compute on padding."""
from __future__ import annotations

import math

import numpy as np


class BucketBatchSampler:
    """Yields lists of dataset indices.

    Every epoch the indices are shuffled, cut into pools of ``batch_size * pool_batches`` items, sorted by
    length inside each pool and split into batches, then the batches are shuffled. Batches therefore contain
    clips of similar length (little padding) while the order stays random from epoch to epoch.
    """

    def __init__(self, lengths, batch_size: int, drop_last: bool = True, shuffle: bool = True,
                 pool_batches: int = 50, seed: int = 0):
        self.lengths = np.asarray(lengths, dtype=np.float64)
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.pool = self.batch_size * max(int(pool_batches), 1)
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        n = len(self.lengths)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = rng.permutation(len(self.lengths)) if self.shuffle else np.arange(len(self.lengths))
        batches: list[list[int]] = []
        for lo in range(0, len(order), self.pool):
            chunk = order[lo:lo + self.pool]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            for start in range(0, len(chunk), self.batch_size):
                batch = chunk[start:start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch.tolist())
        if self.shuffle:
            batches = [batches[i] for i in rng.permutation(len(batches))]
        yield from batches

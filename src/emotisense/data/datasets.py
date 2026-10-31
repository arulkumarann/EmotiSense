"""Utterance dataset and collator (padding, energy-adaptive mixup, normalisation)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .audio import crop, load_audio, normalize_wave
from .augment import energy_adaptive_mixup
from .labels import get_scheme, scheme_targets


class SERDataset(Dataset):
    """Utterance-level speech emotion dataset for one split.

    Rows outside the label scheme are dropped. ``soft_label_alpha`` blends the one-hot
    consensus label with the annotator vote distribution (0 = hard labels only).
    """

    def __init__(self, df: pd.DataFrame, scheme: str, sample_rate: int = 16000,
                 max_seconds: float = 8.0, train: bool = False, soft_label_alpha: float = 0.0):
        keep, hard, votes = scheme_targets(df, scheme)
        self.df = df[keep].reset_index(drop=True)
        self.hard = hard[keep]
        onehot = np.eye(len(get_scheme(scheme)["classes"]), dtype=np.float32)[self.hard]
        self.soft = ((1.0 - soft_label_alpha) * onehot + soft_label_alpha * votes[keep]).astype(np.float32)
        self.sample_rate = sample_rate
        self.max_len = int(max_seconds * sample_rate)
        self.train = train
        self.num_classes = onehot.shape[1]
        if {"valence", "arousal", "dominance"} <= set(self.df.columns):
            vad = self.df[["valence", "arousal", "dominance"]].to_numpy(dtype=np.float32)
            self.vad = np.clip((vad - 1.0) / 4.0, 0.0, 1.0)         # IEMOCAP scale 1-5 -> 0-1
            self.vad_mask = np.ones_like(self.vad)
        else:
            self.vad = np.zeros((len(self.df), 3), dtype=np.float32)
            self.vad_mask = np.zeros((len(self.df), 3), dtype=np.float32)
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.df)

    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng((torch.initial_seed() + os.getpid()) % (2 ** 32))
        return self._rng

    def __getitem__(self, i: int) -> dict:
        wave = load_audio(self.df.at[i, "path"], self.sample_rate)
        wave = crop(wave, self.max_len, random=self.train, rng=self._get_rng() if self.train else None)
        return {
            "idx": i, "wave": wave, "soft": self.soft[i], "hard": int(self.hard[i]),
            "vad": self.vad[i], "vad_mask": self.vad_mask[i],
        }


class Collator:
    """Pads a batch and (in training) applies energy-adaptive mixup."""

    def __init__(self, train: bool = False, normalize: bool = True, mixup_prob: float = 0.0,
                 snr_range: tuple[float, float] = (-5.0, 10.0), min_frac: float = 0.1,
                 max_frac: float = 0.5):
        self.train = train
        self.normalize = normalize
        self.mixup_prob = mixup_prob
        self.snr_range = tuple(snr_range)
        self.min_frac = min_frac
        self.max_frac = max_frac
        self._rng: np.random.Generator | None = None

    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng((torch.initial_seed() + os.getpid()) % (2 ** 32))
        return self._rng

    def __call__(self, items: list[dict]) -> dict:
        b = len(items)
        waves = [it["wave"] for it in items]
        soft = np.stack([it["soft"] for it in items]).copy()
        hard = np.array([it["hard"] for it in items], dtype=np.int64)
        vad = np.stack([it["vad"] for it in items]).copy()
        vad_mask = np.stack([it["vad_mask"] for it in items])
        span = np.full((b, 2), -1, dtype=np.int64)
        span_label = np.full(b, -1, dtype=np.int64)

        if self.train and self.mixup_prob > 0 and b > 1:
            rng = self._get_rng()
            partners = rng.permutation(b)
            for i in range(b):
                j = int(partners[i])
                if j == i or rng.random() >= self.mixup_prob:
                    continue
                res = energy_adaptive_mixup(items[i]["wave"], items[j]["wave"], rng, self.snr_range,
                                            self.min_frac, self.max_frac)
                if res is None:
                    continue
                waves[i] = res.wave
                soft[i] = (1.0 - res.lam) * items[i]["soft"] + res.lam * items[j]["soft"]
                vad[i] = (1.0 - res.lam) * items[i]["vad"] + res.lam * items[j]["vad"]
                vad_mask[i] = vad_mask[i] * items[j]["vad_mask"]
                hard[i] = items[i]["hard"] if res.lam < 0.5 else items[j]["hard"]
                span[i] = (res.start, res.end)
                if res.interferer_dominates:
                    span_label[i] = items[j]["hard"]

        if self.normalize:
            waves = [normalize_wave(w) for w in waves]
        lengths = torch.tensor([len(w) for w in waves], dtype=torch.long)
        max_len = int(lengths.max())
        padded = torch.zeros(b, max_len)
        attn = torch.zeros(b, max_len, dtype=torch.long)
        for i, w in enumerate(waves):
            padded[i, :len(w)] = torch.from_numpy(w)
            attn[i, :len(w)] = 1
        return {
            "idx": torch.tensor([it["idx"] for it in items], dtype=torch.long),
            "wave": padded, "attn": attn, "lengths": lengths,
            "soft": torch.from_numpy(soft), "hard": torch.from_numpy(hard),
            "vad": torch.from_numpy(vad), "vad_mask": torch.from_numpy(vad_mask),
            "span": torch.from_numpy(span), "span_label": torch.from_numpy(span_label),
        }

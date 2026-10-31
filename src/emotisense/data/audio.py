"""Audio I/O: load as mono float32, resample with a polyphase filter, normalise."""
from __future__ import annotations

from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def resample(wave: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wave
    g = gcd(orig_sr, target_sr)
    return resample_poly(wave, target_sr // g, orig_sr // g).astype(np.float32)


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """Read a file to a mono float32 array at ``sr`` Hz."""
    wave, file_sr = sf.read(path, dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    return resample(wave, file_sr, sr)


def normalize_wave(wave: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Zero-mean / unit-variance, as expected by WavLM-Large and HuBERT-Large."""
    wave = wave - wave.mean()
    return (wave / np.sqrt(wave.var() + eps)).astype(np.float32)


def crop(wave: np.ndarray, max_len: int, random: bool, rng: np.random.Generator | None = None) -> np.ndarray:
    if len(wave) <= max_len:
        return wave
    if random:
        rng = rng or np.random.default_rng()
        start = int(rng.integers(0, len(wave) - max_len + 1))
    else:
        start = (len(wave) - max_len) // 2
    return wave[start:start + max_len]

"""Energy-adaptive mixup (EAM), after Wang et al., arXiv:2512.04551.

A random segment of a second utterance ``x_j`` is scaled to a target SNR (drawn from
``snr_range`` dB) relative to the segment of the dominant utterance ``x_i`` it overlays,
then added. The label is mixed with weight

    lambda_mix = P''_j / (P'_i + P''_j) * (l_mix / l_i)

where ``P'_i`` is the power of the overlaid segment of ``x_i``, ``P''_j`` the power of the
scaled interfering segment and ``l_mix / l_i`` the fraction of ``x_i`` that is covered.
The paper leaves the exact power definitions open; this is the most direct reading.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MixResult:
    wave: np.ndarray
    lam: float               # weight of the interfering utterance's label
    start: int               # overlay span in samples of the mixed wave
    end: int
    interferer_dominates: bool  # True when the overlay is louder than the host segment


def energy_adaptive_mixup(x_i: np.ndarray, x_j: np.ndarray, rng: np.random.Generator,
                          snr_range: tuple[float, float] = (-5.0, 10.0),
                          min_frac: float = 0.1, max_frac: float = 0.5,
                          eps: float = 1e-8) -> MixResult | None:
    """Overlay part of ``x_j`` on ``x_i``. Returns None if the inputs are too short."""
    l_i = len(x_i)
    max_len = min(int(l_i * max_frac), len(x_j))
    min_len = max(int(l_i * min_frac), 1)
    if max_len < min_len or l_i < 2:
        return None
    l_mix = int(rng.integers(min_len, max_len + 1))
    s_i = int(rng.integers(0, l_i - l_mix + 1))
    s_j = int(rng.integers(0, len(x_j) - l_mix + 1))

    host = x_i[s_i:s_i + l_mix]
    inter = x_j[s_j:s_j + l_mix]
    p_host = float(np.mean(host ** 2)) + eps
    p_inter = float(np.mean(inter ** 2)) + eps

    snr_db = float(rng.uniform(*snr_range))
    p_target = p_host / (10.0 ** (snr_db / 10.0))     # power of the scaled interferer
    scale = np.sqrt(p_target / p_inter)

    mixed = x_i.copy()
    mixed[s_i:s_i + l_mix] += (scale * inter).astype(x_i.dtype)
    lam = (p_target / (p_host + p_target)) * (l_mix / l_i)
    return MixResult(mixed, float(lam), s_i, s_i + l_mix, p_target > p_host)

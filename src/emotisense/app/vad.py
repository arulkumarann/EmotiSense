"""Dependency-free energy voice-activity detection and chunking for long calls."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VADConfig:
    frame_ms: float = 30.0
    hop_ms: float = 10.0
    margin_db: float = 12.0       # speech must exceed the noise floor by this much
    min_abs_db: float = -50.0     # ... and this absolute level (dBFS)
    min_speech_s: float = 0.3
    merge_gap_s: float = 0.4
    pad_s: float = 0.1
    max_chunk_s: float = 6.0      # long speech is cut into chunks no longer than this
    min_chunk_s: float = 0.4

    @classmethod
    def from_dict(cls, data: dict | None) -> "VADConfig":
        known = {k: v for k, v in (data or {}).items() if k in cls.__dataclass_fields__}
        return cls(**known)


def detect_speech(wave: np.ndarray, sr: int, cfg: VADConfig | None = None) -> list[tuple[float, float]]:
    """Return ``(start_s, end_s)`` speech regions found by frame energy against an adaptive noise floor."""
    cfg = cfg or VADConfig()
    frame, hop = int(sr * cfg.frame_ms / 1000), int(sr * cfg.hop_ms / 1000)
    if len(wave) < frame:
        return []
    windows = np.lib.stride_tricks.sliding_window_view(wave.astype(np.float64), frame)[::hop]
    db = 20.0 * np.log10(np.sqrt((windows ** 2).mean(axis=1)) + 1e-9)
    floor = np.percentile(db, 10)
    voiced = db > max(floor + cfg.margin_db, cfg.min_abs_db)

    runs: list[list[float]] = []
    start = None
    for i, v in enumerate(np.append(voiced, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start * hop / sr, (i - 1) * hop / sr + cfg.frame_ms / 1000])
            start = None
    merged: list[list[float]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= cfg.merge_gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    duration = len(wave) / sr
    return [(max(0.0, s - cfg.pad_s), min(duration, e + cfg.pad_s))
            for s, e in merged if e - s >= cfg.min_speech_s]


def chunk_segments(segments: list[tuple[float, float]], cfg: VADConfig | None = None) -> list[tuple[float, float]]:
    """Split segments longer than ``max_chunk_s`` into equal pieces and drop very short ones."""
    cfg = cfg or VADConfig()
    chunks = []
    for s, e in segments:
        n = max(int(np.ceil((e - s) / cfg.max_chunk_s)), 1)
        edges = np.linspace(s, e, n + 1)
        chunks += [(float(a), float(b)) for a, b in zip(edges[:-1], edges[1:]) if b - a >= cfg.min_chunk_s]
    return chunks

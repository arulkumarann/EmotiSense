"""Tiny synthetic IEMOCAP-shaped corpus for smoke tests and CI (no real data needed).

Each emotion gets a distinct fundamental frequency / amplitude-modulation signature so a
model can in principle separate them; this checks the *plumbing*, not accuracy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .labels import IEMOCAP_CODES
from .manifest import save_manifest

_CODES = ["neu", "hap", "ang", "sad", "fru"]
_F0 = {"neu": 150.0, "hap": 260.0, "ang": 210.0, "sad": 100.0, "fru": 180.0}


def _tone(code: str, seconds: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    f0 = _F0[code] * (1.0 + 0.05 * rng.standard_normal())
    am = 1.0 + 0.5 * np.sin(2 * np.pi * (3.0 + 4.0 * (code == "ang")) * t)
    wave = am * (np.sin(2 * np.pi * f0 * t) + 0.4 * np.sin(2 * np.pi * 2 * f0 * t))
    wave += 0.05 * rng.standard_normal(len(t))
    return (0.3 * wave / np.abs(wave).max()).astype(np.float32)


def build_synthetic_corpus(root: str | Path, out_manifest: str | Path | None = None,
                           dialogs_per_session: int = 2, turns: int = 6, sr: int = 16000,
                           seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    root = Path(root)
    rows = []
    for session in range(1, 6):
        for d in range(dialogs_per_session):
            dialog = f"Ses{session:02d}F_syn{d:02d}"
            t0 = 0.0
            for turn in range(turns):
                gender = "F" if turn % 2 == 0 else "M"
                code = _CODES[int(rng.integers(0, len(_CODES)))]
                seconds = float(rng.uniform(1.0, 2.0))
                utt = f"{dialog}_{gender}{turn:03d}"
                rel = Path(f"Session{session}") / dialog / f"{utt}.wav"
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(root / rel), _tone(code, seconds, sr, rng), sr)
                votes = {c: 0.0 for c in IEMOCAP_CODES}
                votes[code] = 2.0
                votes[_CODES[int(rng.integers(0, len(_CODES)))]] += 1.0   # a dissenting evaluator
                rows.append({
                    "utt_id": utt, "path": rel.as_posix(), "dataset": "synthetic",
                    "session": session, "speaker": f"{session}{gender}", "dialog": dialog,
                    "start": round(t0, 3), "end": round(t0 + seconds, 3), "duration": round(seconds, 3),
                    "label": code, "n_eval": 3, **{f"v_{c}": v for c, v in votes.items()},
                    "valence": 3.0, "arousal": 1.0 + 4.0 * (code in ("hap", "ang", "fru")), "dominance": 3.0,
                })
                t0 += seconds + 0.3
    df = pd.DataFrame(rows)
    if out_manifest is not None:
        save_manifest(df, out_manifest)
    return df

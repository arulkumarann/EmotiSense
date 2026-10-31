"""Label schemes and target construction.

A *scheme* fixes the class list and how raw dataset labels map onto it.

* ``iemocap4``  - neutral / happy(+excited) / angry / sad. This is the protocol of the
  base paper (5,531 utterances) and the one used for the benchmark comparison.
* ``iemocap5``  - adds *frustrated*, the emotion that matters most for call centres.
* ``ravdess8``  - the eight RAVDESS emotions.

IEMOCAP is annotated by several evaluators, so beyond the hard consensus label we can
build *soft* targets from the vote distribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Raw IEMOCAP category codes in a fixed order; vote columns are named ``v_<code>``.
IEMOCAP_CODES = ["neu", "hap", "exc", "ang", "sad", "fru", "fea", "sur", "dis", "oth"]

IEMOCAP_WORD_TO_CODE = {
    "neutral": "neu", "happiness": "hap", "excited": "exc", "anger": "ang",
    "sadness": "sad", "frustration": "fru", "fear": "fea", "surprise": "sur",
    "disgust": "dis", "other": "oth",
}

LABEL_SCHEMES: dict[str, dict] = {
    "iemocap4": {
        "classes": ["neutral", "happy", "angry", "sad"],
        "map": {"neu": "neutral", "hap": "happy", "exc": "happy", "ang": "angry", "sad": "sad"},
    },
    "iemocap5": {
        "classes": ["neutral", "happy", "angry", "sad", "frustrated"],
        "map": {"neu": "neutral", "hap": "happy", "exc": "happy", "ang": "angry",
                "sad": "sad", "fru": "frustrated"},
    },
    "ravdess8": {
        "classes": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"],
        "map": {c: c for c in ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]},
    },
}

# The EmotiSense timeline outputs: name -> raw IEMOCAP codes whose votes make up its intensity.
INTENSITY_EMOTIONS: dict[str, list[str]] = {
    "anger": ["ang"],
    "frustration": ["fru"],
    "happiness": ["hap", "exc"],      # shown as "satisfaction" in the dashboard
    "sadness": ["sad"],
    "neutral": ["neu"],               # shown as "calm" in the dashboard
}
DISPLAY_NAMES = {"happiness": "satisfaction", "neutral": "calm"}


def get_scheme(name: str) -> dict:
    if name not in LABEL_SCHEMES:
        raise KeyError(f"Unknown label scheme {name!r}. Available: {sorted(LABEL_SCHEMES)}")
    return LABEL_SCHEMES[name]


def scheme_targets(df: pd.DataFrame, scheme_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build targets for a scheme.

    Returns ``(keep, hard, soft)`` where ``keep`` is a boolean mask of rows that belong
    to the scheme, ``hard`` the class index per kept row (-1 elsewhere) and ``soft`` an
    ``[N, K]`` vote distribution restricted to the scheme classes (one-hot when a dataset
    carries no votes or no vote falls inside the scheme).
    """
    scheme = get_scheme(scheme_name)
    classes = scheme["classes"]
    index = {c: i for i, c in enumerate(classes)}
    n, k = len(df), len(classes)

    mapped = df["label"].map(scheme["map"])
    keep = mapped.notna().to_numpy()
    hard = np.full(n, -1, dtype=np.int64)
    hard[keep] = mapped[keep].map(index).to_numpy(dtype=np.int64)

    soft = np.zeros((n, k), dtype=np.float32)
    soft[np.where(keep)[0], hard[keep]] = 1.0

    vote_cols = [f"v_{c}" for c in IEMOCAP_CODES]
    if all(c in df.columns for c in vote_cols):
        votes = np.zeros((n, k), dtype=np.float32)
        for code, cls in scheme["map"].items():
            votes[:, index[cls]] += df[f"v_{code}"].to_numpy(dtype=np.float32)
        totals = votes.sum(axis=1, keepdims=True)
        has_votes = (totals[:, 0] > 0) & keep
        soft_votes = np.divide(votes, totals, out=np.zeros_like(votes), where=totals > 0)
        soft[has_votes] = soft_votes[has_votes]
    return keep, hard, soft


def intensity_targets(df: pd.DataFrame) -> np.ndarray:
    """``[N, 5]`` annotator-agreement intensities in [0, 1] for the timeline emotions.

    Each intensity is the fraction of evaluators that assigned the emotion, counted over
    *all* categories (so "other"/"fear" votes lower every intensity instead of being dropped).
    """
    vote_cols = [f"v_{c}" for c in IEMOCAP_CODES]
    missing = [c for c in vote_cols if c not in df.columns]
    if missing:
        raise ValueError("Timeline targets need IEMOCAP vote columns; re-run `prepare iemocap`.")
    votes = df[vote_cols].to_numpy(dtype=np.float32)
    total = votes.sum(axis=1, keepdims=True)
    frac = np.divide(votes, total, out=np.zeros_like(votes), where=total > 0)
    col = {c: i for i, c in enumerate(IEMOCAP_CODES)}
    out = np.zeros((len(df), len(INTENSITY_EMOTIONS)), dtype=np.float32)
    for j, codes in enumerate(INTENSITY_EMOTIONS.values()):
        out[:, j] = sum(frac[:, col[c]] for c in codes)
    return out

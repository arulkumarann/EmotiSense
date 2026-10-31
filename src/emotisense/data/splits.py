"""Cross-validation protocols (speaker/session independent).

IEMOCAP follows the base paper: 5-fold *session-independent* CV, fold ``k`` tests on
session ``k+1``. The paper does not describe validation, so a held-out *speaker* from
the training sessions is used for model selection (never the test set).
RAVDESS uses 6-fold actor-independent CV (4 test actors per fold).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_FOLDS = {"iemocap": 5, "ravdess": 6, "synthetic": 5}


def n_folds(dataset: str) -> int:
    return N_FOLDS[dataset]


def split_fold(df: pd.DataFrame, dataset: str, fold: int, level: str = "speaker") -> dict[str, np.ndarray]:
    """Return boolean masks ``{"train", "val", "test"}`` over ``df`` for ``fold``.

    ``level="speaker"``: validation is a single held-out speaker (utterance-level SER).
    ``level="session"``: validation is a whole held-out session/group (dialog-level
    timeline models, where both speakers of a dialog must stay together).
    """
    k = n_folds(dataset)
    if not 0 <= fold < k:
        raise ValueError(f"fold must be in [0, {k - 1}] for {dataset}, got {fold}")
    groups = sorted(df["session"].unique())
    if len(groups) != k:
        raise ValueError(f"{dataset} manifest has {len(groups)} session groups, expected {k}")
    test_group = groups[fold]
    next_group = groups[(fold + 1) % k]
    is_test = (df["session"] == test_group).to_numpy()

    if level == "session":
        is_val = (df["session"] == next_group).to_numpy()
    elif level == "speaker":
        speakers = sorted(df.loc[df["session"] == next_group, "speaker"].unique())
        val_speaker = speakers[fold % len(speakers)]
        is_val = (df["speaker"] == val_speaker).to_numpy()
    else:
        raise ValueError(f"Unknown validation level {level!r}")

    is_train = ~(is_test | is_val)
    return {"train": is_train, "val": is_val, "test": is_test}

"""Manifest = one CSV row per audio clip. Every dataset is reduced to this schema."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_COLUMNS = ["utt_id", "path", "dataset", "session", "speaker", "dialog", "start", "end",
                "duration", "label"]


def save_manifest(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}\n"
            "Create it first, e.g. `emotisense prepare iemocap --root <IEMOCAP_full_release>` "
            "or `emotisense download ravdess && emotisense prepare ravdess`."
        )
    df = pd.read_csv(path, dtype={"speaker": str, "session": int, "dialog": str})
    missing = [c for c in BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest {path} is missing columns: {missing}")
    return df


def resolve_paths(df: pd.DataFrame, root: str | Path | None) -> pd.DataFrame:
    """Return a copy whose ``path`` column is absolute (relative paths join ``root``)."""
    out = df.copy()
    if root is not None:
        root = Path(root)
        out["path"] = [p if Path(p).is_absolute() else str(root / p) for p in out["path"]]
    return out

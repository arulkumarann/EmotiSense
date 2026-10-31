"""Manifest = one CSV row per audio clip. Every dataset is reduced to this schema."""
from __future__ import annotations

import json
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
            "Create it first with `emotisense download iemocap` (or `download ravdess`), "
            "or `emotisense prepare <dataset> --root <folder>` for your own copy."
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


def resolve_audio_root(audio_root: str | Path | None, manifest_path: str | Path) -> str | None:
    """Where the audio lives: the configured folder if it exists, else the one `prepare` recorded.

    ``prepare`` / ``download`` write ``meta.json`` next to the manifest with the absolute dataset
    root, so a dataset fetched to a cache folder (e.g. by kagglehub) needs no extra configuration.
    """
    if audio_root and Path(audio_root).exists():
        return str(audio_root)
    meta = Path(manifest_path).with_name("meta.json")
    if meta.exists():
        recorded = json.loads(meta.read_text(encoding="utf-8")).get("audio_root")
        if recorded and Path(recorded).exists():
            return recorded
    return str(audio_root) if audio_root else None

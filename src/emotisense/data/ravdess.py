"""RAVDESS (speech): 1,440 clips from 24 actors, eight emotions. Freely downloadable."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .manifest import save_manifest

RAVDESS_EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
RAVDESS_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
RAVDESS_MD5 = "bc696df654c87fed845eb13823edef8a"
RAVDESS_ZIP = "Audio_Speech_Actors_01-24.zip"


def parse_ravdess_name(name: str) -> dict | None:
    """``03-01-06-01-02-01-12.wav`` -> modality-channel-emotion-intensity-statement-repetition-actor."""
    parts = Path(name).stem.split("-")
    if len(parts) != 7 or parts[2] not in RAVDESS_EMOTIONS:
        return None
    return {
        "modality": parts[0], "channel": parts[1], "emotion": RAVDESS_EMOTIONS[parts[2]],
        "intensity": int(parts[3]), "statement": int(parts[4]),
        "repetition": int(parts[5]), "actor": int(parts[6]),
    }


def build_ravdess_manifest(root: str | Path, out_path: str | Path | None = None) -> pd.DataFrame:
    import soundfile as sf

    root = Path(root)
    wavs = sorted(root.rglob("*.wav"))
    rows = []
    for wav in wavs:
        info = parse_ravdess_name(wav.name)
        if info is None or info["channel"] != "01":  # keep speech, drop song
            continue
        duration = sf.info(str(wav)).duration
        actor = info["actor"]
        rows.append({
            "utt_id": wav.stem,
            "path": wav.relative_to(root).as_posix(),
            "dataset": "ravdess",
            "session": (actor - 1) // 4 + 1,   # 6 groups of 4 actors = the 6 CV folds
            "speaker": f"{actor:02d}",
            "dialog": f"actor{actor:02d}",
            "start": 0.0,
            "end": duration,
            "duration": round(duration, 4),
            "label": info["emotion"],
            "intensity": info["intensity"],
        })
    if not rows:
        raise RuntimeError(f"No RAVDESS speech wav files found under {root}")
    df = pd.DataFrame(rows).sort_values(["session", "speaker", "utt_id"]).reset_index(drop=True)
    if out_path is not None:
        save_manifest(df, out_path)
    return df

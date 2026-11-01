"""Build local sample sessions from IEMOCAP dialogues, analysed with the trained model.

    python scripts/make_samples.py [--timeline models/timeline_bilstm_deploy.pt] [--out data/samples]

Each sample is `<name>.<wav|mp3>` plus `<name>.json`. They are for local demos only: IEMOCAP's licence does not allow
redistributing its audio, so `data/samples/` is git-ignored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from emotisense.app.analyzer import CallAnalyzer
from emotisense.data.audio import load_audio
from emotisense.data.manifest import load_manifest, resolve_audio_root, resolve_paths

SR = 16000
GAP = 0.35
TARGET_SECONDS = 140

PICKS = [
    # name, title, tag, how to rank dialogues, audio format
    ("rising-tension", "Acted dialogue: tension builds", "Sample", lambda d: d["neg_late"] - d["neg_early"] + d["neg"], "wav"),
    ("light-and-warm", "Acted dialogue: light and warm", "Sample", lambda d: d["joy"], "mp3"),
    ("quiet-and-low", "Acted dialogue: quiet and low", "Sample", lambda d: d["sad"] + 0.3 * d["calm"], "wav"),
]


def dialog_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dialog, g in df.groupby("dialog"):
        g = g.sort_values("start")
        if len(g) < 24 or g["duration"].sum() < TARGET_SECONDS * 0.8:
            continue
        lab = g["label"].tolist()
        half = len(lab) // 2
        neg = lambda xs: float(np.mean([x in ("ang", "fru") for x in xs]))  # noqa: E731
        rows.append({
            "dialog": dialog, "neg": neg(lab), "neg_early": neg(lab[:half]), "neg_late": neg(lab[half:]),
            "joy": float(np.mean([x in ("hap", "exc") for x in lab])), "sad": float(np.mean([x == "sad" for x in lab])),
            "calm": float(np.mean([x == "neu" for x in lab])),
        })
    return pd.DataFrame(rows)


def render_dialog(turns: pd.DataFrame, seconds: float) -> np.ndarray:
    parts, total = [], 0.0
    for path in turns["path"]:
        wave = load_audio(path, SR)
        parts += [wave, np.zeros(int(GAP * SR), dtype=np.float32)]
        total += len(wave) / SR + GAP
        if total >= seconds:
            break
    return np.concatenate(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", default="models/timeline_bilstm_deploy.pt")
    ap.add_argument("--manifest", default="data/processed/iemocap/manifest.csv")
    ap.add_argument("--out", default="data/samples")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = resolve_paths(load_manifest(args.manifest), resolve_audio_root(None, args.manifest))
    stats = dialog_stats(df)
    analyzer = CallAnalyzer(args.timeline)
    used: set[str] = set()
    for name, title, tag, score, fmt in PICKS:
        ranked = stats[~stats["dialog"].isin(used)].assign(rank=lambda d: d.apply(score, axis=1)).sort_values("rank", ascending=False)
        dialog = ranked.iloc[0]["dialog"]
        used.add(dialog)
        wave = render_dialog(df[df["dialog"] == dialog].sort_values("start"), TARGET_SECONDS)
        audio_path = out / f"{name}.{fmt}"
        sf.write(str(audio_path), wave, SR, **({"format": "MP3"} if fmt == "mp3" else {}))
        print(f"{name}: dialog {dialog}, {len(wave) / SR:.0f}s -> analysing ...", flush=True)
        result = analyzer.analyze(str(audio_path), name=audio_path.name)
        (out / f"{name}.json").write_text(json.dumps({"title": title, "tag": tag, "result": result}), encoding="utf-8")
        print(f"   saved {audio_path.name} ({len(result['chunks'])} passages, {len(result['alerts'])} alerts)", flush=True)


if __name__ == "__main__":
    main()

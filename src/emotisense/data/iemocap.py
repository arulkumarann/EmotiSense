"""IEMOCAP: parse ``EmoEvaluation`` files into a manifest with votes and VAD scores.

IEMOCAP is distributed under a research licence by USC SAIL
(https://sail.usc.edu/iemocap/), so it cannot be downloaded automatically. Place the
extracted ``IEMOCAP_full_release`` folder under ``data/raw/`` and run
``emotisense prepare iemocap``.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .labels import IEMOCAP_CODES, IEMOCAP_WORD_TO_CODE
from .manifest import save_manifest

_HEADER = re.compile(
    r"^\[(?P<start>[\d.]+)\s*-\s*(?P<end>[\d.]+)\]\s+(?P<utt>\S+)\s+(?P<label>\S+)\s+"
    r"\[(?P<val>[\d.]+),\s*(?P<act>[\d.]+),\s*(?P<dom>[\d.]+)\]"
)
_CATEGORICAL = re.compile(r"^C-\w+:\s*(?P<body>.*)$")


def parse_evaluation_file(path: str | Path) -> list[dict]:
    """Parse one ``EmoEvaluation/*.txt`` file into per-utterance records."""
    records: list[dict] = []
    current: dict | None = None
    with open(path, "r", encoding="latin-1") as f:
        for raw in f:
            line = raw.strip()
            header = _HEADER.match(line)
            if header:
                if current is not None:
                    records.append(current)
                current = {
                    "utt_id": header["utt"],
                    "start": float(header["start"]),
                    "end": float(header["end"]),
                    "label": header["label"],
                    "val": float(header["val"]),
                    "act": float(header["act"]),
                    "dom": float(header["dom"]),
                    "votes": {c: 0.0 for c in IEMOCAP_CODES},
                    "n_eval": 0,
                }
                continue
            cat = _CATEGORICAL.match(line)
            if cat and current is not None:
                body = cat["body"].split("(")[0]
                words = [w.strip().lower() for w in body.split(";") if w.strip()]
                codes = [IEMOCAP_WORD_TO_CODE[w] for w in words if w in IEMOCAP_WORD_TO_CODE]
                if codes:
                    # Each evaluator contributes one vote, split across the labels they gave.
                    for code in codes:
                        current["votes"][code] += 1.0 / len(codes)
                    current["n_eval"] += 1
    if current is not None:
        records.append(current)
    return records


def build_iemocap_manifest(root: str | Path, out_path: str | Path | None = None,
                           check_audio: bool = True) -> pd.DataFrame:
    root = Path(root)
    if not (root / "Session1").exists():
        # Allow pointing at the parent directory holding IEMOCAP_full_release.
        candidates = list(root.glob("IEMOCAP*"))
        if candidates and (candidates[0] / "Session1").exists():
            root = candidates[0]
        else:
            raise FileNotFoundError(
                f"No Session1 folder under {root}. Point --root at the extracted "
                "IEMOCAP_full_release directory."
            )
    rows: list[dict] = []
    missing_audio = 0
    for session in range(1, 6):
        eval_dir = root / f"Session{session}" / "dialog" / "EmoEvaluation"
        files = sorted(p for p in eval_dir.glob("*.txt") if not p.name.startswith("._"))
        if not files:
            raise FileNotFoundError(f"No evaluation files in {eval_dir}")
        for fpath in files:
            for rec in parse_evaluation_file(fpath):
                utt = rec["utt_id"]
                dialog, turn = utt.rsplit("_", 1)
                wav = Path(f"Session{session}") / "sentences" / "wav" / dialog / f"{utt}.wav"
                if check_audio and not (root / wav).exists():
                    missing_audio += 1
                    continue
                rows.append({
                    "utt_id": utt,
                    "path": wav.as_posix(),
                    "dataset": "iemocap",
                    "session": session,
                    "speaker": f"{session}{turn[0]}",
                    "dialog": dialog,
                    "start": rec["start"],
                    "end": rec["end"],
                    "duration": round(rec["end"] - rec["start"], 4),
                    "label": rec["label"],
                    "n_eval": rec["n_eval"],
                    **{f"v_{c}": round(v, 4) for c, v in rec["votes"].items()},
                    "valence": rec["val"], "arousal": rec["act"], "dominance": rec["dom"],
                })
    if not rows:
        raise RuntimeError("No IEMOCAP utterances found; check the dataset path.")
    df = pd.DataFrame(rows).sort_values(["session", "dialog", "start"]).reset_index(drop=True)
    df.attrs["root"] = str(root)
    if missing_audio:
        print(f"[warn] {missing_audio} utterances skipped because their wav file is missing")
    if out_path is not None:
        save_manifest(df, out_path)
    return df

"""IEMOCAP: parse ``EmoEvaluation`` files into a manifest with votes and VAD scores.

The corpus is fetched from the public Kaggle copy ``ourbucket/iemocap-audio-complete``
(``emotisense download iemocap``). The builder does not depend on the exact folder layout:
it looks for ``EmoEvaluation/*.txt`` label files and for audio files anywhere under the
given root, and matches them by utterance id (e.g. ``Ses01F_impro01_F000``). The original
USC SAIL licence for IEMOCAP still applies to any use of the data.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .labels import IEMOCAP_CODES, IEMOCAP_WORD_TO_CODE
from .manifest import save_manifest

AUDIO_SUFFIXES = {".wav", ".flac"}

_HEADER = re.compile(
    r"^\[(?P<start>[\d.]+)\s*-\s*(?P<end>[\d.]+)\]\s+(?P<utt>\S+)\s+(?P<label>\S+)\s+"
    r"\[(?P<val>[\d.]+),\s*(?P<act>[\d.]+),\s*(?P<dom>[\d.]+)\]"
)
_CATEGORICAL = re.compile(r"^C-\w+:\s*(?P<body>.*)$")
_UTTERANCE = re.compile(r"^Ses(?P<session>\d{2})[FM]_")


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


def find_label_files(root: Path) -> list[Path]:
    """All top-level ``*.txt`` files of every ``EmoEvaluation`` directory under ``root``."""
    found: dict[str, Path] = {}
    for directory in sorted(root.rglob("EmoEvaluation")):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.txt")):
            if not path.name.startswith("._"):
                found.setdefault(path.name, path)
    return list(found.values())


def index_audio(root: Path) -> dict[str, Path]:
    """Map utterance id (file stem) -> audio path for every audio file under ``root``."""
    return {p.stem: p for p in root.rglob("*")
            if p.suffix.lower() in AUDIO_SUFFIXES and not p.name.startswith("._")}


def build_iemocap_manifest(root: str | Path, out_path: str | Path | None = None) -> pd.DataFrame:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"IEMOCAP folder not found: {root}. Run `emotisense download iemocap` (public Kaggle copy), "
            "or pass --root pointing at an extracted IEMOCAP_full_release folder."
        )
    label_files = find_label_files(root)
    if not label_files:
        raise FileNotFoundError(
            f"No IEMOCAP label files (*/EmoEvaluation/*.txt) found under {root}. The emotion labels are "
            "required to build the benchmark; this copy of the dataset may not include them."
        )
    audio = index_audio(root)
    if not audio:
        raise FileNotFoundError(f"Label files were found under {root} but no audio files (.wav/.flac).")

    rows: list[dict] = []
    missing_audio = skipped = 0
    for fpath in label_files:
        for rec in parse_evaluation_file(fpath):
            utt = rec["utt_id"]
            match = _UTTERANCE.match(utt)
            if match is None or "_" not in utt:
                skipped += 1
                continue
            wav = audio.get(utt)
            if wav is None:
                missing_audio += 1
                continue
            dialog, turn = utt.rsplit("_", 1)
            session = int(match["session"])
            rows.append({
                "utt_id": utt,
                "path": wav.relative_to(root).as_posix(),
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
        raise RuntimeError(
            f"Found {len(label_files)} label files and {len(audio)} audio files under {root}, but no "
            "utterance id matched between them. Expected ids such as Ses01F_impro01_F000."
        )
    if missing_audio:
        print(f"[warn] {missing_audio} labelled utterances have no audio file and were skipped")
    if skipped:
        print(f"[warn] {skipped} label entries with unrecognised utterance ids were skipped")
    df = pd.DataFrame(rows).sort_values(["session", "dialog", "start"]).reset_index(drop=True)
    df.attrs["root"] = str(root)
    if out_path is not None:
        save_manifest(df, out_path)
    return df

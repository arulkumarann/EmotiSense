"""IEMOCAP: build the manifest from the public Kaggle copy or from the official release.

Two label sources are understood, found automatically anywhere under the dataset root:

* ``EmoEvaluation/*.txt`` (official release): consensus label, every evaluator's vote and the
  valence / arousal / dominance scores. Gives soft labels, a VAD auxiliary task, timeline
  intensities and real utterance timestamps. Matches the 5,531-utterance protocol of the base paper.
* ``training_data.csv`` (``file_id, emotion, audio_path``; the Kaggle copy
  ``ourbucket/iemocap-audio-complete``): one hard label per utterance. Votes become one-hot, VAD is
  absent and utterance times are reconstructed from durations in turn order. Its label set differs from
  the official consensus (all 10,039 utterances are labelled), so scores are *not* directly comparable
  to the paper.

Audio is matched to labels by utterance id (e.g. ``Ses01F_impro01_F000``), so the folder layout
does not matter. The original USC SAIL licence for IEMOCAP still applies to any use of the data.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import soundfile as sf

from .labels import IEMOCAP_CODES, IEMOCAP_WORD_TO_CODE
from .manifest import save_manifest

AUDIO_SUFFIXES = {".wav", ".flac"}

_HEADER = re.compile(
    r"^\[(?P<start>[\d.]+)\s*-\s*(?P<end>[\d.]+)\]\s+(?P<utt>\S+)\s+(?P<label>\S+)\s+"
    r"\[(?P<val>[\d.]+),\s*(?P<act>[\d.]+),\s*(?P<dom>[\d.]+)\]"
)
_CATEGORICAL = re.compile(r"^C-\w+:\s*(?P<body>.*)$")
_UTTERANCE = re.compile(r"^Ses(?P<session>\d{2})[FM]_")
TURN_GAP_S = 0.3        # pause assumed between turns when timestamps must be reconstructed


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


def find_label_csv(root: Path) -> Path | None:
    """A CSV with ``file_id`` and ``emotion`` columns (the Kaggle copy's ``training_data.csv``)."""
    for path in sorted(root.rglob("*.csv")):
        try:
            columns = set(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue
        if {"file_id", "emotion"} <= columns:
            return path
    return None


def _build_from_evaluations(root: Path, label_files: list[Path], audio: dict[str, Path]) -> pd.DataFrame:
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
    if missing_audio:
        print(f"[warn] {missing_audio} labelled utterances have no audio file and were skipped")
    if skipped:
        print(f"[warn] {skipped} label entries with unrecognised utterance ids were skipped")
    if not rows:
        raise RuntimeError(
            f"Found {len(label_files)} label files and {len(audio)} audio files under {root}, but no "
            "utterance id matched between them. Expected ids such as Ses01F_impro01_F000."
        )
    return pd.DataFrame(rows)


def _build_from_csv(root: Path, csv_path: Path, audio: dict[str, Path]) -> pd.DataFrame:
    table = pd.read_csv(csv_path)
    rows: list[dict] = []
    missing_audio = skipped = 0
    for utt, emotion in zip(table["file_id"].astype(str), table["emotion"].astype(str)):
        code = IEMOCAP_WORD_TO_CODE.get(emotion.strip().lower())
        match = _UTTERANCE.match(utt)
        if code is None or match is None or "_" not in utt:
            skipped += 1
            continue
        wav = audio.get(utt)
        if wav is None:
            missing_audio += 1
            continue
        dialog, turn = utt.rsplit("_", 1)
        session = int(match["session"])
        digits = re.sub(r"\D", "", turn)
        rows.append({
            "utt_id": utt,
            "path": wav.relative_to(root).as_posix(),
            "dataset": "iemocap",
            "session": session,
            "speaker": f"{session}{turn[0]}",
            "dialog": dialog,
            "turn": int(digits) if digits else 0,
            "duration": round(sf.info(str(wav)).duration, 4),
            "label": code,
            "n_eval": 1,
            **{f"v_{c}": float(c == code) for c in IEMOCAP_CODES},     # one hard vote
        })
    if missing_audio:
        print(f"[warn] {missing_audio} labelled utterances have no audio file and were skipped")
    if skipped:
        print(f"[warn] {skipped} CSV rows with unknown emotion or utterance id were skipped")
    if not rows:
        raise RuntimeError(f"{csv_path} is a label table, but none of its file_ids matched an audio file.")
    df = pd.DataFrame(rows).sort_values(["session", "dialog", "turn", "speaker"]).reset_index(drop=True)
    # No timestamps in this source: place turns back to back in dialogue order.
    step = df["duration"] + TURN_GAP_S
    df["start"] = (step.groupby(df["dialog"]).cumsum() - step).round(4)
    df["end"] = (df["start"] + df["duration"]).round(4)
    return df.drop(columns="turn")


def build_iemocap_manifest(root: str | Path, out_path: str | Path | None = None) -> pd.DataFrame:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"IEMOCAP folder not found: {root}. Run `emotisense download iemocap` (public Kaggle copy), "
            "or pass --root pointing at an extracted IEMOCAP_full_release folder."
        )
    label_files = find_label_files(root)
    csv_path = None if label_files else find_label_csv(root)
    if not label_files and csv_path is None:
        raise FileNotFoundError(
            f"No IEMOCAP labels found under {root}: expected */EmoEvaluation/*.txt (official release) or a "
            "CSV with file_id and emotion columns (Kaggle copy)."
        )
    audio = index_audio(root)
    if not audio:
        raise FileNotFoundError(f"Labels were found under {root} but no audio files (.wav/.flac).")
    if label_files:
        df = _build_from_evaluations(root, label_files, audio)
        source = "official"
    else:
        df = _build_from_csv(root, csv_path, audio)
        source = "csv"
    df = df.sort_values(["session", "dialog", "start"]).reset_index(drop=True)
    df.attrs["root"] = str(root)
    df.attrs["label_source"] = source
    if out_path is not None:
        save_manifest(df, out_path)
    return df

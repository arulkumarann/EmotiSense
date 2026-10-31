"""Dataset download helpers (resumable, checksummed)."""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from .ravdess import RAVDESS_MD5, RAVDESS_URL, RAVDESS_ZIP

IEMOCAP_INSTRUCTIONS = """\
IEMOCAP is licence-gated and cannot be downloaded automatically.

  1. Request access (free for research): https://sail.usc.edu/iemocap/iemocap_release.htm
  2. Extract the archive so that this folder exists:
         data/raw/IEMOCAP_full_release/Session1 ... Session5
  3. Build the manifest:
         emotisense prepare iemocap --root data/raw/IEMOCAP_full_release
"""


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download_file(url: str, dest: Path, expected_md5: str | None = None) -> Path:
    """Stream ``url`` to ``dest`` with resume support and optional MD5 verification."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_md5 and md5sum(dest) == expected_md5:
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    resume_at = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}
    with requests.get(url, stream=True, headers=headers, timeout=60) as resp:
        if resp.status_code == 416:  # already fully downloaded
            resume_at = 0
        resp.raise_for_status()
        if resume_at and resp.status_code != 206:
            resume_at = 0  # server ignored the Range header; start over
        total = int(resp.headers.get("content-length", 0)) + resume_at
        mode = "ab" if resume_at else "wb"
        with open(part, mode) as f, tqdm(total=total, initial=resume_at, unit="B",
                                         unit_scale=True, desc=dest.name) as bar:
            for block in resp.iter_content(chunk_size=1 << 20):
                f.write(block)
                bar.update(len(block))
    part.replace(dest)
    if expected_md5 and md5sum(dest) != expected_md5:
        dest.unlink()
        raise RuntimeError(f"Checksum mismatch for {dest.name}; the download was removed, retry.")
    return dest


def download_ravdess(raw_dir: str | Path = "data/raw") -> Path:
    raw_dir = Path(raw_dir)
    target = raw_dir / "RAVDESS"
    if target.exists() and any(target.rglob("*.wav")):
        print(f"RAVDESS already present at {target}")
        return target
    archive = download_file(RAVDESS_URL, raw_dir / RAVDESS_ZIP, RAVDESS_MD5)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)
    archive.unlink()
    print(f"RAVDESS extracted to {target}")
    return target

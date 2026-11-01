"""Audio I/O: load as mono float32, resample with a polyphase filter, normalise."""
from __future__ import annotations

import io
import shutil
import subprocess
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


class UnsupportedAudio(RuntimeError):
    """The file is not audio we can decode."""


def _ffmpeg_decode(path: str, sr: int) -> np.ndarray:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise UnsupportedAudio("This format needs ffmpeg. Use wav, mp3, ogg or flac, or install ffmpeg.")
    proc = subprocess.run([exe, "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", "-"],
                          capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise UnsupportedAudio("Could not decode this file as audio: " + proc.stderr.decode(errors="ignore")[:200])
    wave, _ = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    return wave


def probe_duration(path: str) -> float | None:
    """Duration in seconds, or None when it cannot be determined cheaply."""
    try:
        return float(sf.info(path).duration)
    except (sf.LibsndfileError, RuntimeError):
        pass
    exe = shutil.which("ffprobe")
    if exe:
        out = subprocess.run([exe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path],
                             capture_output=True, text=True)
        try:
            return float(out.stdout.strip())
        except ValueError:
            return None
    return None


def resample(wave: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wave
    g = gcd(orig_sr, target_sr)
    return resample_poly(wave, target_sr // g, orig_sr // g).astype(np.float32)


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """Read a file to a mono float32 array at ``sr`` Hz (wav/mp3/ogg/flac natively, anything else via ffmpeg)."""
    try:
        wave, file_sr = sf.read(path, dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError):
        return _ffmpeg_decode(path, sr)
    wave = wave.mean(axis=1)
    return resample(wave, file_sr, sr)


def normalize_wave(wave: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Zero-mean / unit-variance, as expected by WavLM-Large and HuBERT-Large."""
    wave = wave - wave.mean()
    return (wave / np.sqrt(wave.var() + eps)).astype(np.float32)


def crop(wave: np.ndarray, max_len: int, random: bool, rng: np.random.Generator | None = None) -> np.ndarray:
    if len(wave) <= max_len:
        return wave
    if random:
        rng = rng or np.random.default_rng()
        start = int(rng.integers(0, len(wave) - max_len + 1))
    else:
        start = (len(wave) - max_len) // 2
    return wave[start:start + max_len]

"""CallAnalyzer: a recording -> emotion timeline, tension alerts and summary."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..data.audio import load_audio, resample
from ..data.labels import DISPLAY_NAMES
from ..embedding import build_embedder
from ..timeline.escalation import EscalationConfig, analyze_timeline
from ..timeline.model import TimelineModel
from .vad import VADConfig, chunk_segments, detect_speech


def waveform_peaks(wave: np.ndarray, bins: int = 720) -> list[float]:
    """Peak amplitude per bin, scaled to 0..1, for drawing a waveform without shipping the audio."""
    if len(wave) == 0:
        return [0.0] * bins
    edges = np.linspace(0, len(wave), bins + 1).astype(int)
    peaks = np.array([np.abs(wave[a:max(b, a + 1)]).max() for a, b in zip(edges[:-1], edges[1:])])
    peak = peaks.max()
    return [round(float(v), 3) for v in (peaks / peak if peak > 0 else peaks)]


class CallAnalyzer:
    """Pipeline: resample -> energy VAD -> chunks -> embeddings -> timeline model -> escalation logic."""

    def __init__(self, timeline_checkpoint: str | Path, ser_checkpoint: str | Path | None = None,
                 device: str | None = None, vad: VADConfig | dict | None = None,
                 escalation: EscalationConfig | dict | None = None, embedder=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.vad = vad if isinstance(vad, VADConfig) else VADConfig.from_dict(vad)
        self._escalation_override = escalation
        self._ser_checkpoint = ser_checkpoint
        self._injected_embedder = embedder is not None
        self.embedder = embedder
        self.embedder_spec: dict | None = None
        self.use_timeline(timeline_checkpoint)

    def use_timeline(self, checkpoint: str | Path) -> None:
        """Switch to another timeline checkpoint.

        Timeline models that share an encoder (all of ours use the frozen WavLM-Large) reuse the already-loaded encoder,
        so switching only swaps a small head and is nearly instant. Nothing is changed if loading fails.
        """
        ckpt = torch.load(checkpoint, map_location="cpu")
        mc = ckpt["model_cfg"]
        model = TimelineModel(ckpt["in_dim"], len(ckpt["emotions"]), mc["kind"], mc["hidden"],
                              mc["layers"], mc["heads"], mc["dropout"]).to(self.device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        spec = dict(ckpt["embedder"])
        if self._ser_checkpoint is not None and spec["kind"] == "finetuned":
            spec["checkpoint"] = str(self._ser_checkpoint)
        embedder = self.embedder
        if not self._injected_embedder and (embedder is None or spec != self.embedder_spec):
            embedder = build_embedder(spec, str(self.device))
        override = self._escalation_override
        escalation = (override if isinstance(override, EscalationConfig)
                      else EscalationConfig.from_dict(override or ckpt.get("escalation")))
        # commit only after everything loaded
        self.model, self.embedder, self.embedder_spec = model, embedder, spec
        self.emotions = list(ckpt["emotions"])
        self.escalation = escalation
        self.timeline_name = str(checkpoint)
        self.timeline_id = Path(checkpoint).stem
        self.timeline_kind = mc["kind"]

    def _load(self, audio, sr: int | None) -> np.ndarray:
        target = self.embedder.sample_rate
        if isinstance(audio, (str, Path)):
            return load_audio(str(audio), target)
        wave = np.asarray(audio, dtype=np.float32)
        if wave.ndim > 1:
            wave = wave.mean(axis=-1)
        return resample(wave, sr or target, target)

    def analyze(self, audio, sr: int | None = None, name: str | None = None, progress=None) -> dict:
        """``progress(stage, fraction, detail)`` is called as the analysis advances (for UIs)."""
        report = progress or (lambda stage, fraction, detail="": None)
        report("decode", 0.02, "Reading audio")
        wave = self._load(audio, sr)
        rate = self.embedder.sample_rate
        duration = len(wave) / rate
        result: dict = {
            "name": name or (Path(audio).name if isinstance(audio, (str, Path)) else "call"),
            "duration": duration, "emotions": self.emotions,
            "display_names": {e: DISPLAY_NAMES.get(e, e) for e in self.emotions},
            "escalation_config": self.escalation.to_dict(),
            "model": {"id": self.timeline_id, "kind": self.timeline_kind},
            "waveform": waveform_peaks(wave),
        }
        report("speech", 0.08, "Finding speech")
        chunks = chunk_segments(detect_speech(wave, rate, self.vad), self.vad)
        if not chunks:
            result.update({"chunks": [], "scores": {e: [] for e in self.emotions}, "class_probs": None,
                           "summary": analyze_timeline(np.array([]), {}, self.escalation),
                           "alerts": [], "warning": "No speech detected."})
            return result

        pieces = [wave[int(s * rate):int(e * rate)] for s, e in chunks]
        parts, prob_parts = [], []
        for lo in range(0, len(pieces), 8):
            e, p = self.embedder.embed(pieces[lo:lo + 8])
            parts.append(e)
            prob_parts.append(p)
            done = min(lo + 8, len(pieces))
            report("listen", 0.1 + 0.8 * done / len(pieces), f"Listening, {done} of {len(pieces)} passages")
        emb = torch.cat(parts)
        probs = torch.cat(prob_parts) if prob_parts[0] is not None else None
        report("timeline", 0.93, "Tracing the timeline")
        with torch.no_grad():
            x = emb.unsqueeze(0).to(self.device)
            scores = self.model(x, torch.ones(1, len(chunks), dtype=torch.bool, device=self.device))[0].cpu().numpy()
        times = np.array([s for s, _ in chunks])
        ends = np.array([e for _, e in chunks])
        by_emotion = {e: scores[:, j] for j, e in enumerate(self.emotions)}
        summary = analyze_timeline(times, by_emotion, self.escalation, ends)
        alerts = summary.pop("alerts")
        result.update({
            "chunks": [{"start": s, "end": e} for s, e in chunks],
            "scores": {e: v.tolist() for e, v in by_emotion.items()},
            "class_probs": None if probs is None else {
                "classes": self.embedder.classes, "values": probs.numpy().tolist()},
            "summary": summary, "alerts": alerts,
        })
        return result

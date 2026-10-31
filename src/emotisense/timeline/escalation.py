"""Escalation detection on a predicted emotion timeline.

A call *escalates* when the negative-emotion score (anger / frustration) either crosses an
absolute threshold or rises sharply within a short window. Everything here works on plain
numpy arrays so it can be unit-tested and reused for live, incremental scoring.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class EscalationConfig:
    emotions: tuple = ("anger", "frustration")
    combine: str = "max"             # max | mean over `emotions`
    smooth_window: int = 3           # moving-average length in timeline points
    causal: bool = False             # trailing (live) vs centred (post-call) smoothing
    alert_threshold: float = 0.6     # absolute level that raises an alert
    rise_delta: float = 0.3          # rise within `rise_window_s` that raises an alert ...
    rise_window_s: float = 20.0
    rise_floor: float = 0.35         # ... provided the score has also reached this level
    merge_gap_s: float = 5.0         # alerts closer than this become one event
    trend_delta: float = 0.2         # fitted change over the call needed to call it a trend

    @classmethod
    def from_dict(cls, data: dict | None) -> "EscalationConfig":
        data = dict(data or {})
        if "emotions" in data:
            data["emotions"] = tuple(data["emotions"])
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["emotions"] = list(self.emotions)
        return d


@dataclass
class Alert:
    start: float
    end: float
    peak: float
    peak_time: float
    kind: str                        # "threshold" | "rise"
    indices: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "peak": self.peak,
                "peak_time": self.peak_time, "kind": self.kind}


def negative_score(scores: dict[str, np.ndarray], cfg: EscalationConfig) -> np.ndarray:
    stacked = np.stack([np.asarray(scores[e], dtype=np.float64) for e in cfg.emotions])
    return stacked.max(axis=0) if cfg.combine == "max" else stacked.mean(axis=0)


def smooth(x: np.ndarray, window: int, causal: bool = False) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if window <= 1 or len(x) == 0:
        return x.copy()
    out = np.empty_like(x)
    for i in range(len(x)):
        if causal:
            lo, hi = max(0, i - window + 1), i + 1
        else:
            half = window // 2
            lo, hi = max(0, i - half), min(len(x), i + half + 1)
        out[i] = x[lo:hi].mean()
    return out


def detect_alerts(times: np.ndarray, score: np.ndarray, cfg: EscalationConfig,
                  ends: np.ndarray | None = None) -> list[Alert]:
    """Find alert events in ``score`` (already smoothed) sampled at ``times`` seconds."""
    times = np.asarray(times, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    ends = times if ends is None else np.asarray(ends, dtype=np.float64)
    n = len(score)
    flag = np.zeros(n, dtype=bool)
    kind = [""] * n
    for k in range(n):
        if score[k] >= cfg.alert_threshold:
            flag[k], kind[k] = True, "threshold"
            continue
        recent = (times < times[k]) & (times >= times[k] - cfg.rise_window_s)
        if recent.any() and score[k] >= cfg.rise_floor and score[k] - score[recent].min() >= cfg.rise_delta:
            flag[k], kind[k] = True, "rise"

    events: list[list[int]] = []
    for k in np.where(flag)[0]:
        if events and (times[k] - ends[events[-1][-1]]) <= cfg.merge_gap_s:
            events[-1].append(int(k))
        else:
            events.append([int(k)])
    alerts = []
    for idx in events:
        peak_i = idx[int(np.argmax(score[idx]))]
        kinds = {kind[i] for i in idx}
        alerts.append(Alert(start=float(times[idx[0]]), end=float(ends[idx[-1]]), peak=float(score[peak_i]),
                            peak_time=float(times[peak_i]),
                            kind="threshold" if "threshold" in kinds else "rise", indices=idx))
    return alerts


def classify_trend(times: np.ndarray, score: np.ndarray, cfg: EscalationConfig) -> str:
    """``escalating`` / ``de-escalating`` / ``stable`` from the least-squares change over the call."""
    times, score = np.asarray(times, dtype=np.float64), np.asarray(score, dtype=np.float64)
    if len(score) < 3 or np.ptp(times) <= 0:
        return "stable"
    slope = np.polyfit(times, score, 1)[0]
    change = slope * np.ptp(times)
    if change >= cfg.trend_delta:
        return "escalating"
    if change <= -cfg.trend_delta:
        return "de-escalating"
    return "stable"


def analyze_timeline(times: np.ndarray, scores: dict[str, np.ndarray], cfg: EscalationConfig,
                     ends: np.ndarray | None = None) -> dict:
    """Score a call: smoothed negative-emotion curve, alerts, trend and headline numbers."""
    times = np.asarray(times, dtype=np.float64)
    if len(times) == 0:
        return {"negative": [], "alerts": [], "trend": "stable", "mean": 0.0, "peak": 0.0,
                "peak_time": 0.0, "n_alerts": 0, "fraction_above": 0.0}
    neg = smooth(negative_score(scores, cfg), cfg.smooth_window, cfg.causal)
    alerts = detect_alerts(times, neg, cfg, ends)
    peak_i = int(np.argmax(neg))
    return {
        "negative": neg.tolist(),
        "alerts": [a.to_dict() for a in alerts],
        "trend": classify_trend(times, neg, cfg),
        "mean": float(neg.mean()),
        "peak": float(neg[peak_i]),
        "peak_time": float(times[peak_i]),
        "n_alerts": len(alerts),
        "fraction_above": float((neg >= cfg.alert_threshold).mean()),
    }

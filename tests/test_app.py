import json

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from emotisense.app.analyzer import CallAnalyzer
from emotisense.app.api import create_app
from emotisense.app.plots import agent_figure, format_time, moments_to_review, timeline_figure
from emotisense.app.store import CallStore
from emotisense.app.vad import VADConfig, chunk_segments, detect_speech
from emotisense.config import load_config
from emotisense.timeline.train import run_timeline_cv

SR = 16000


def make_call(path, seconds=(2.0, 1.0, 3.0), gap=1.0):
    """Tone bursts separated by silence: a crude stand-in for turns of speech."""
    parts = [np.zeros(int(0.5 * SR), dtype=np.float32)]
    for i, dur in enumerate(seconds):
        t = np.arange(int(dur * SR)) / SR
        parts += [(0.3 * np.sin(2 * np.pi * (180 + 40 * i) * t)).astype(np.float32),
                  np.zeros(int(gap * SR), dtype=np.float32)]
    wave = np.concatenate(parts)
    sf.write(str(path), wave, SR)
    return wave


# ------------------------------------------------------------------------ VAD
def test_vad_finds_bursts_and_ignores_silence(tmp_path):
    wave = make_call(tmp_path / "c.wav")
    segs = detect_speech(wave, SR)
    assert len(segs) == 3
    assert segs[0][0] == pytest.approx(0.4, abs=0.15) and segs[0][1] == pytest.approx(2.6, abs=0.2)
    assert detect_speech(np.zeros(SR * 3, dtype=np.float32), SR) == []
    assert detect_speech(np.zeros(10, dtype=np.float32), SR) == []


def test_vad_rejects_steady_noise_floor():
    rng = np.random.default_rng(0)
    quiet_noise = (0.001 * rng.standard_normal(SR * 3)).astype(np.float32)
    assert detect_speech(quiet_noise, SR) == []


def test_chunking_splits_long_speech():
    cfg = VADConfig(max_chunk_s=6.0, min_chunk_s=0.4)
    chunks = chunk_segments([(0.0, 14.0), (20.0, 20.2)], cfg)
    assert len(chunks) == 3 and all(b - a <= 6.0 + 1e-9 for a, b in chunks)   # short fragment dropped
    assert chunks[0][0] == 0.0 and chunks[-1][1] == pytest.approx(14.0)


# -------------------------------------------------------------- analyzer + API
@pytest.fixture(scope="module")
def deploy_checkpoint(synthetic_corpus, tmp_path_factory):
    root, df = synthetic_corpus
    tmp = tmp_path_factory.mktemp("deploy")
    df.to_csv(tmp / "m.csv", index=False)
    cfg = load_config("configs/timeline/smoke.yaml", [
        f"data.manifest={(tmp / 'm.csv').as_posix()}", f"data.audio_root={root.as_posix()}",
        f"embeddings.cache_dir={tmp.as_posix()}/cache", "train.epochs=2",
    ])
    return run_timeline_cv(cfg, run_dir=tmp / "run", folds=[0])["deploy_checkpoint"]


def test_analyzer_end_to_end(deploy_checkpoint, tmp_path):
    make_call(tmp_path / "call.wav")
    analyzer = CallAnalyzer(deploy_checkpoint, device="cpu")
    result = analyzer.analyze(tmp_path / "call.wav")
    n = len(result["chunks"])
    assert n >= 3 and result["duration"] == pytest.approx(9.5, abs=0.1)
    assert set(result["scores"]) == set(analyzer.emotions)
    assert all(len(v) == n and 0 <= min(v) and max(v) <= 1 for v in result["scores"].values())
    assert len(result["summary"]["negative"]) == n
    assert result["summary"]["trend"] in ("stable", "escalating", "de-escalating")
    assert result["display_names"]["happiness"] == "satisfaction" and result["display_names"]["neutral"] == "calm"
    json.dumps(result)                                                      # fully serialisable

    silent = analyzer.analyze(np.zeros(SR * 2, dtype=np.float32), sr=SR, name="silence")
    assert silent["chunks"] == [] and "No speech" in silent["warning"] and silent["alerts"] == []

    resampled = analyzer.analyze(np.asarray(sf.read(str(tmp_path / "call.wav"))[0])[::2], sr=SR // 2)
    assert resampled["chunks"]


class FakeAnalyzer:
    def analyze(self, path, name=None):
        n = 4
        scores = {e: [0.1 * (i + 1) for i in range(n)] for e in ("anger", "frustration", "happiness", "sadness", "neutral")}
        neg = [0.2, 0.4, 0.7, 0.9]
        return {"name": name, "duration": 12.0, "emotions": list(scores),
                "display_names": {"happiness": "satisfaction", "neutral": "calm"},
                "chunks": [{"start": 3.0 * i, "end": 3.0 * i + 2} for i in range(n)], "scores": scores,
                "summary": {"mean": 0.55, "peak": 0.9, "peak_time": 9.0, "trend": "escalating", "n_alerts": 1,
                            "negative": neg, "fraction_above": 0.5},
                "alerts": [{"start": 6.0, "end": 9.0, "peak": 0.9, "peak_time": 9.0, "kind": "threshold"}],
                "escalation_config": {"alert_threshold": 0.6}}


def test_api_flow(tmp_path):
    make_call(tmp_path / "c.wav")
    client = TestClient(create_app(FakeAnalyzer(), tmp_path / "calls.sqlite"))
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    with open(tmp_path / "c.wav", "rb") as f:
        resp = client.post("/analyze", files={"file": ("c.wav", f, "audio/wav")}, data={"agent": "asha"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == 1 and body["agent"] == "asha" and body["summary"]["trend"] == "escalating"
    client.post("/analyze", files={"file": ("d.wav", b"RIFF", "audio/wav")}, data={"agent": "ravi"})
    assert [c["agent"] for c in client.get("/calls").json()] == ["ravi", "asha"]
    assert client.get("/calls/1").json()["name"] == "c.wav"
    assert client.get("/calls/99").status_code == 404
    agents = {a["agent"]: a for a in client.get("/agents").json()}
    assert agents["asha"]["n_calls"] == 1 and agents["asha"]["escalation_rate"] == 1.0
    assert agents["asha"]["churn_risk"] == "high"


def test_api_validation_and_missing_model(tmp_path, monkeypatch):
    client = TestClient(create_app(FakeAnalyzer(), tmp_path / "a.sqlite"))
    assert client.post("/analyze", files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 415
    monkeypatch.delenv("EMOTISENSE_TIMELINE_CKPT", raising=False)
    cold = TestClient(create_app(None, tmp_path / "b.sqlite"))
    assert cold.get("/health").json()["model_loaded"] is False
    resp = cold.post("/analyze", files={"file": ("x.wav", b"RIFFxxxx", "audio/wav")})
    assert resp.status_code == 503 and "EMOTISENSE_TIMELINE_CKPT" in resp.json()["detail"]


def test_store_persists_and_ranks_agents(tmp_path):
    store = CallStore(tmp_path / "s.sqlite")
    calm = FakeAnalyzer().analyze(None, "calm")
    calm["summary"].update(mean=0.1, peak=0.2, n_alerts=0)
    store.add(calm, "good_agent")
    store.add(FakeAnalyzer().analyze(None, "angry"), "bad_agent")
    reopened = CallStore(tmp_path / "s.sqlite")
    summary = reopened.agent_summary()
    assert [a["agent"] for a in summary] == ["bad_agent", "good_agent"]
    assert summary[0]["churn_risk"] == "high" and summary[1]["churn_risk"] == "normal"
    assert reopened.get(1)["name"] == "calm" and len(reopened.list_calls("bad_agent")) == 1


def test_plot_helpers():
    result = FakeAnalyzer().analyze(None, "x")
    fig = timeline_figure(result)
    assert len(fig.data) == 6 and fig.layout.shapes            # 5 emotions + negative curve, alert band + threshold
    assert agent_figure([{"agent": "a", "avg_negative": 0.4, "escalation_rate": 0.2}]).data[0].y[0] == 0.4
    assert format_time(125) == "2:05"
    moments = moments_to_review(result, top=2)
    assert [m["time"] for m in moments] == ["0:06", "0:09"]
    assert moments_to_review({"summary": {"negative": []}, "chunks": []}) == []


def test_dashboard_renders_a_saved_result(tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    script = tmp_path / "page.py"
    script.write_text(
        "import streamlit as st\n"
        "from emotisense.app.dashboard import show_result\n"
        f"import json\nshow_result(json.load(open(r'{tmp_path / 'r.json'}')))\n")
    (tmp_path / "r.json").write_text(json.dumps(FakeAnalyzer().analyze(None, "demo.wav")))
    at = AppTest.from_file(str(script), default_timeout=60).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert labels == ["Overall negativity", "Peak score", "Peak time", "Trend"]
    assert at.metric[3].value == "Escalating"

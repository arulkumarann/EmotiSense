import json
import shutil
import time

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from emotisense.app.analyzer import waveform_peaks
from emotisense.data.audio import UnsupportedAudio, load_audio, probe_duration
from emotisense.timeline.model import TimelineModel
from emotisense.web.registry import ModelRegistry
from emotisense.web.server import create_web_app, seed_samples
from emotisense.web.store import SessionStore

SR = 16000
EMOTIONS = ["anger", "frustration", "happiness", "sadness", "neutral"]
FORMATS = {"wav": ("WAV", None), "mp3": ("MP3", None), "ogg": ("OGG", "VORBIS"), "opus": ("OGG", "OPUS"),
           "flac": ("FLAC", None)}


def write_audio(path, seconds=6.0, fmt="wav"):
    t = np.arange(int(seconds * SR)) / SR
    wave = (0.4 * np.sin(2 * np.pi * 200 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.5 * t))).astype(np.float32)
    major, sub = FORMATS[fmt]
    sf.write(str(path), wave, SR, format=major, **({"subtype": sub} if sub else {}))
    return path


class FakeAnalyzer:
    """Decodes the file for real (so format support is exercised) and fabricates a plausible timeline."""

    def __init__(self, fail=False, silent=False):
        self.fail, self.silent = fail, silent

    def analyze(self, path, name=None, progress=None):
        report = progress or (lambda *a, **k: None)
        report("decode", 0.02, "Reading audio")
        if self.fail:
            raise RuntimeError("boom")
        wave = load_audio(str(path), SR)
        duration = len(wave) / SR
        starts = np.arange(0, max(duration - 1.5, 0.5), 2.0)
        chunks = [] if self.silent else [{"start": float(s), "end": float(min(s + 1.8, duration))} for s in starts]
        n = len(chunks)
        ramp = np.linspace(0.1, 0.9, max(n, 1))[:n]
        scores = {"anger": ramp.tolist(), "frustration": (ramp * 0.8).tolist(), "happiness": (1 - ramp).tolist(),
                  "sadness": [0.1] * n, "neutral": (1 - ramp * 0.5).tolist()}
        for i in range(n):
            report("listen", 0.1 + 0.8 * (i + 1) / n, f"Listening, {i + 1} of {n} passages")
        return {"name": name, "duration": duration, "emotions": EMOTIONS,
                "display_names": {"happiness": "joy", "neutral": "calm"}, "waveform": waveform_peaks(wave),
                "chunks": chunks, "scores": scores, "class_probs": None, "summary": {}, "alerts": [],
                "escalation_config": {}}


@pytest.fixture()
def client(tmp_path):
    app = create_web_app(analyzer=FakeAnalyzer(), data_dir=tmp_path / "data", samples_dir=None)
    with TestClient(app) as c:
        c.tmp = tmp_path
        yield c


def wait_for(client, job_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def upload(client, path, **form):
    with open(path, "rb") as f:
        return client.post("/api/sessions", files={"file": (path.name, f, "application/octet-stream")}, data=form)


# --------------------------------------------------------------------- formats
@pytest.mark.parametrize("fmt", sorted(FORMATS))
def test_every_audio_format_decodes_and_analyses(client, fmt):
    path = write_audio(client.tmp / f"talk.{fmt}", 6.0, fmt)
    assert abs(load_audio(str(path), SR).shape[0] / SR - 6.0) < 0.2
    assert abs(probe_duration(str(path)) - 6.0) < 0.2
    resp = upload(client, path, title="My talk", tag="Interview")
    assert resp.status_code == 202
    job = wait_for(client, resp.json()["job_id"])
    assert job["status"] == "done" and job["progress"] == 1.0
    session = client.get(f"/api/sessions/{resp.json()['session_id']}").json()
    assert session["title"] == "My talk" and session["tag"] == "Interview" and len(session["waveform"]) == 720
    assert len(session["chunks"]) >= 2 and set(session["scores"]) == set(EMOTIONS)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_m4a_goes_through_ffmpeg_fallback(client):
    wav = write_audio(client.tmp / "src.wav", 4.0, "wav")
    import subprocess
    m4a = client.tmp / "voice.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), str(m4a)], check=True)
    assert abs(load_audio(str(m4a), SR).shape[0] / SR - 4.0) < 0.3
    job = wait_for(client, upload(client, m4a).json()["job_id"])
    assert job["status"] == "done"


def test_invalid_uploads_are_rejected_clearly(client):
    (client.tmp / "notes.txt").write_text("hello")
    assert upload(client, client.tmp / "notes.txt").status_code == 415
    (client.tmp / "empty.wav").write_bytes(b"")
    assert upload(client, client.tmp / "empty.wav").status_code == 422
    (client.tmp / "fake.mp3").write_bytes(b"this is not an mp3 at all" * 40)
    resp = upload(client, client.tmp / "fake.mp3")
    assert resp.status_code == 422 and "audio" in resp.json()["detail"].lower()
    assert client.get("/api/sessions").json() == []
    with pytest.raises(UnsupportedAudio):
        load_audio(str(client.tmp / "fake.mp3"), SR)


def test_failed_and_silent_analyses_report_errors_and_clean_up(tmp_path):
    for analyzer, message in ((FakeAnalyzer(fail=True), "boom"), (FakeAnalyzer(silent=True), "No speech")):
        data = tmp_path / message.replace(" ", "_")
        with TestClient(create_web_app(analyzer=analyzer, data_dir=data, samples_dir=None)) as c:
            c.tmp = tmp_path
            job = wait_for(c, upload(c, write_audio(tmp_path / "a.wav", 4.0)).json()["job_id"])
            assert job["status"] == "error" and message in job["error"]
            assert list((data / "audio").iterdir()) == [] and c.get("/api/sessions").json() == []


def test_progress_is_reported_while_running(tmp_path):
    seen = []

    class Slow(FakeAnalyzer):
        def analyze(self, path, name=None, progress=None):
            for i in range(3):
                progress("listen", 0.2 + 0.2 * i, f"step {i}")
                time.sleep(0.25)
            return super().analyze(path, name, progress)

    with TestClient(create_web_app(analyzer=Slow(), data_dir=tmp_path / "d", samples_dir=None)) as c:
        job_id = upload(c, write_audio(tmp_path / "a.wav", 5.0)).json()["job_id"]
        while True:
            job = c.get(f"/api/jobs/{job_id}").json()
            seen.append((job["status"], job["stage"], job["progress"]))
            if job["status"] in ("done", "error"):
                break
            time.sleep(0.05)
    assert any(s[0] == "running" and 0 < s[2] < 1 for s in seen) and seen[-1][0] == "done"
    assert [s[2] for s in seen] == sorted(s[2] for s in seen)          # progress never goes backwards
    assert c.get("/api/jobs/nope").status_code == 404


# ----------------------------------------------------------------------- audio
def test_audio_streaming_supports_ranges(client):
    path = write_audio(client.tmp / "a.mp3", 5.0, "mp3")
    resp = upload(client, path).json()
    assert wait_for(client, resp["job_id"])["status"] == "done"
    sid = resp["session_id"]
    raw = path.read_bytes()
    full = client.get(f"/api/sessions/{sid}/audio")
    assert full.status_code == 200 and full.headers["accept-ranges"] == "bytes" and full.content == raw
    assert full.headers["content-type"] == "audio/mpeg"
    part = client.get(f"/api/sessions/{sid}/audio", headers={"Range": "bytes=100-199"})
    assert part.status_code == 206 and part.content == raw[100:200]
    assert part.headers["content-range"] == f"bytes 100-199/{len(raw)}"
    tail = client.get(f"/api/sessions/{sid}/audio", headers={"Range": "bytes=-50"})
    assert tail.status_code == 206 and tail.content == raw[-50:]
    open_end = client.get(f"/api/sessions/{sid}/audio", headers={"Range": f"bytes={len(raw) - 10}-"})
    assert open_end.content == raw[-10:]
    assert client.get(f"/api/sessions/{sid}/audio", headers={"Range": "bytes=99999999-"}).status_code == 416
    assert client.get(f"/api/sessions/{sid}/audio", headers={"Range": "garbage"}).status_code == 416
    assert client.get("/api/sessions/zzz/audio").status_code == 404


# ------------------------------------------------------ sessions, notes, settings
def make_session(client, **form):
    resp = upload(client, write_audio(client.tmp / "s.wav", 8.0), **form)
    wait_for(client, resp.json()["job_id"])
    return resp.json()["session_id"]


def test_session_lifecycle_notes_and_delete(client):
    sid = make_session(client, title="Standup", tag="Meeting")
    listing = client.get("/api/sessions").json()
    assert listing[0]["id"] == sid and listing[0]["summary"]["sparkline"] and "emotion_means" in listing[0]["summary"]
    assert client.get("/api/sessions", params={"tag": "Meeting"}).json()[0]["id"] == sid
    assert client.get("/api/sessions", params={"tag": "Other"}).json() == []
    assert client.get("/api/sessions", params={"q": "stand"}).json()[0]["id"] == sid

    assert client.patch(f"/api/sessions/{sid}", json={"title": "Renamed", "tag": "Work"}).status_code == 200
    assert client.get(f"/api/sessions/{sid}").json()["title"] == "Renamed"

    note = client.post(f"/api/sessions/{sid}/notes", json={"t": 3.5, "text": "  Voice rises here  "})
    assert note.status_code == 201 and note.json()["text"] == "Voice rises here"
    client.post(f"/api/sessions/{sid}/notes", json={"t": 1.0, "text": "Earlier"})
    assert [n["t"] for n in client.get(f"/api/sessions/{sid}").json()["notes"]] == [1.0, 3.5]
    assert client.get("/api/sessions").json()[0]["notes"] == 2
    assert client.post(f"/api/sessions/{sid}/notes", json={"t": 1, "text": "  "}).status_code == 422
    assert client.delete(f"/api/notes/{note.json()['id']}").status_code == 200
    assert client.delete("/api/notes/999").status_code == 404

    audio_files = list((client.tmp / "data" / "audio").iterdir())
    assert len(audio_files) == 1
    assert client.delete(f"/api/sessions/{sid}").status_code == 200
    assert list((client.tmp / "data" / "audio").iterdir()) == [] and client.get(f"/api/sessions/{sid}").status_code == 404
    assert client.delete(f"/api/sessions/{sid}").status_code == 404


def test_settings_change_alerts_everywhere(client):
    sid = make_session(client)
    before = client.get(f"/api/sessions/{sid}").json()
    assert before["escalation_config"]["alert_threshold"] == 0.6 and len(before["alerts"]) >= 1
    saved = client.put("/api/settings", json={"alert_threshold": 0.99, "rise_delta": 0.99}).json()
    assert saved["values"]["alert_threshold"] == 0.99 and saved["defaults"]["alert_threshold"] == 0.6
    after = client.get(f"/api/sessions/{sid}").json()
    assert after["alerts"] == [] and after["escalation_config"]["alert_threshold"] == 0.99
    assert client.get("/api/sessions").json()[0]["summary"]["n_alerts"] == 0
    assert client.put("/api/settings", json={"alert_threshold": "high"}).status_code == 422
    assert client.get("/api/settings").json()["values"]["alert_threshold"] == 0.99


def test_samples_are_seeded_once_and_their_audio_is_never_deleted(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    write_audio(samples / "demo.wav", 8.0)
    result = FakeAnalyzer().analyze(str(samples / "demo.wav"), "demo.wav")
    (samples / "demo.json").write_text(json.dumps({"title": "Demo session", "tag": "Sample", "result": result}))
    (samples / "orphan.json").write_text(json.dumps({"result": result}))          # no audio file: ignored
    store = SessionStore(tmp_path / "x.sqlite")
    assert seed_samples(store, samples) == 1 and seed_samples(store, samples) == 0
    assert seed_samples(store, tmp_path / "missing") == 0
    app = create_web_app(analyzer=FakeAnalyzer(), data_dir=tmp_path / "data", samples_dir=samples)
    with TestClient(app) as c:
        item = c.get("/api/sessions").json()[0]
        assert item["is_sample"] and item["title"] == "Demo session" and item["id"] == "sample-demo"
        assert c.get("/api/sessions/sample-demo/audio").status_code == 200
        assert c.delete("/api/sessions/sample-demo").status_code == 200
    assert (samples / "demo.wav").exists()


def test_health_reports_model_state(tmp_path):
    with TestClient(create_web_app(analyzer=FakeAnalyzer(), data_dir=tmp_path / "a", samples_dir=None)) as c:
        assert c.get("/api/health").json()["model"] == "ready"
        assert "mp3" in c.get("/api/health").json()["formats"]
    with TestClient(create_web_app(data_dir=tmp_path / "b", samples_dir=None)) as c:
        assert c.get("/api/health").json()["model"] in ("error", "absent")
    def broken():
        raise RuntimeError("weights missing")
    with TestClient(create_web_app(analyzer_factory=broken, data_dir=tmp_path / "c", samples_dir=None)) as c:
        for _ in range(50):
            if c.get("/api/health").json()["model"] != "loading":
                break
            time.sleep(0.05)
        health = c.get("/api/health").json()
        assert health["model"] == "error" and "weights missing" in health["model_error"]
        job = wait_for(c, upload(c, write_audio(tmp_path / "a.wav", 4.0)).json()["job_id"])
        assert job["status"] == "error" and "weights missing" in job["error"]


def test_static_app_is_served_with_spa_fallback(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>app</html>")
    (dist / "assets" / "main.js").write_text("console.log(1)")
    (tmp_path / "secret.txt").write_text("top secret")
    with TestClient(create_web_app(analyzer=FakeAnalyzer(), data_dir=tmp_path / "d", static_dir=dist,
                                   samples_dir=None)) as c:
        assert c.get("/").text == "<html>app</html>"
        assert c.get("/app/session/abc").text == "<html>app</html>"          # client-side routes
        assert c.get("/assets/main.js").text == "console.log(1)"
        assert c.get("/api/nothing").status_code == 404
        assert "top secret" not in c.get("/..%2fsecret.txt").text and "top secret" not in c.get("/../secret.txt").text


# ------------------------------------------------------------------- models
def write_timeline_ckpt(path, kind="bilstm", val_ccc=0.5, encoder="microsoft/wavlm-large"):
    model = TimelineModel(16, 5, kind, 8, 1, 2, 0.1)
    torch.save({"model": model.state_dict(), "in_dim": 16, "emotions": EMOTIONS, "fold": "deploy",
                "model_cfg": {"kind": kind, "hidden": 8, "layers": 1, "heads": 2, "dropout": 0.1},
                "embedder": {"kind": "frozen", "encoder": encoder, "tiny": False}, "escalation": {}, "val_ccc": val_ccc}, path)
    return path


class SwitchingAnalyzer(FakeAnalyzer):
    """Records which timeline model it was asked to use, like the real analyzer does."""

    def __init__(self, timeline_id):
        super().__init__()
        self.timeline_id, self.switched = timeline_id, []

    def use_timeline(self, path):
        self.switched.append(path.stem)
        self.timeline_id = path.stem


@pytest.fixture()
def models_dir(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    write_timeline_ckpt(d / "timeline_mlp_deploy.pt", "mlp", 0.49)
    write_timeline_ckpt(d / "timeline_bilstm_deploy.pt", "bilstm", 0.59)
    write_timeline_ckpt(d / "timeline_transformer_deploy.pt", "transformer", 0.58)
    (d / "notes.pt").write_bytes(b"not a checkpoint")                      # ignored without error
    torch.save({"something": "else"}, d / "other.pt")                       # a checkpoint of some other kind
    return d


def test_registry_finds_only_timeline_models_in_a_stable_order(models_dir):
    reg = ModelRegistry(models_dir)
    assert [e.kind for e in reg.timeline_models()] == ["bilstm", "transformer", "mlp"]
    assert reg.get("timeline_mlp_deploy").params > 0 and reg.get("nope") is None and reg.get(None) is None
    assert reg.default_id() == "timeline_bilstm_deploy" and reg.default_id("timeline_mlp_deploy") == "timeline_mlp_deploy"
    assert reg.default_id("missing") == "timeline_bilstm_deploy"
    assert ModelRegistry(None).timeline_models() == [] and ModelRegistry(models_dir / "absent").default_id() is None
    extra = write_timeline_ckpt(models_dir.parent / "elsewhere.pt", "lstm")
    assert len(ModelRegistry(models_dir, extra_paths=(extra, models_dir / "timeline_mlp_deploy.pt")).timeline_models()) == 4


def test_models_api_describes_everything_for_the_ui(models_dir, tmp_path):
    ser = tmp_path / "ser"
    ser.mkdir()
    (ser / "plus_fold0_best.pt").write_bytes(b"x" * 2_000_000)
    app = create_web_app(analyzer=FakeAnalyzer(), data_dir=tmp_path / "d", samples_dir=None,
                         registry=ModelRegistry(models_dir, ser_dirs=(ser, tmp_path / "missing")),
                         default_model="timeline_bilstm_deploy")
    with TestClient(app) as c:
        body = c.get("/api/models").json()
        assert body["active"] == "timeline_bilstm_deploy"
        assert [m["kind"] for m in body["timeline"]] == ["bilstm", "transformer", "mlp"]
        bilstm = body["timeline"][0]
        assert bilstm["active"] and bilstm["selectable"] and bilstm["stats"]["ccc"]["mean"] == pytest.approx(0.5129)
        assert set(bilstm["stats"]["per_emotion_ccc"]) == set(EMOTIONS)
        assert bilstm["strengths"] and bilstm["limits"] and bilstm["params"] > 0 and bilstm["file"].endswith(".pt")
        assert not any(m["active"] for m in body["timeline"][1:])
        assert body["encoder"]["name"] == "WavLM-Large" and not body["encoder"]["selectable"]
        clf = body["classifier"]
        assert clf["available"] and clf["files"] == [{"file": "plus_fold0_best.pt", "fold": 1, "size_mb": 2}]
        assert len(clf["stats"]["folds"]) == 5 and clf["stats"]["mean"]["ua"] == pytest.approx(67.41)
        assert {g["term"] for g in body["glossary"]} >= {"CCC", "Alert F1"} and body["supported"] and body["data_note"]
    with TestClient(create_web_app(analyzer=FakeAnalyzer(), data_dir=tmp_path / "e", samples_dir=None)) as c:
        empty = c.get("/api/models").json()
        assert empty["timeline"] == [] and empty["active"] is None and not empty["classifier"]["available"]


def test_active_model_can_be_changed_and_is_remembered(models_dir, tmp_path):
    def build():
        return create_web_app(analyzer=FakeAnalyzer(), data_dir=tmp_path / "d", samples_dir=None,
                              registry=ModelRegistry(models_dir), default_model="timeline_bilstm_deploy")
    with TestClient(build()) as c:
        assert c.put("/api/models/active", json={"id": "timeline_mlp_deploy"}).json() == {"active": "timeline_mlp_deploy"}
        assert c.get("/api/models").json()["active"] == "timeline_mlp_deploy"
        assert c.put("/api/models/active", json={"id": "bogus"}).status_code == 422
        assert c.put("/api/models/active", json={}).status_code == 422
        assert c.get("/api/models").json()["active"] == "timeline_mlp_deploy"        # a bad request changes nothing
    with TestClient(build()) as c:                                                       # "restart"
        assert c.get("/api/models").json()["active"] == "timeline_mlp_deploy"
    (models_dir / "timeline_mlp_deploy.pt").unlink()
    with TestClient(build()) as c:                                                       # remembered model was removed
        assert c.get("/api/models").json()["active"] == "timeline_bilstm_deploy"


def test_analysis_uses_the_requested_model(models_dir, tmp_path):
    analyzer = SwitchingAnalyzer("timeline_bilstm_deploy")
    app = create_web_app(analyzer=analyzer, data_dir=tmp_path / "d", samples_dir=None,
                         registry=ModelRegistry(models_dir), default_model="timeline_bilstm_deploy")
    with TestClient(app) as c:
        audio = write_audio(tmp_path / "a.wav", 5.0)
        assert wait_for(c, upload(c, audio).json()["job_id"])["status"] == "done"
        assert analyzer.switched == []                                                   # default is already loaded
        job = wait_for(c, upload(c, audio, model="timeline_transformer_deploy").json()["job_id"])
        assert job["status"] == "done" and analyzer.switched == ["timeline_transformer_deploy"]
        c.put("/api/models/active", json={"id": "timeline_mlp_deploy"})
        wait_for(c, upload(c, audio).json()["job_id"])                                   # no explicit model: the saved default
        assert analyzer.switched[-1] == "timeline_mlp_deploy"
        wait_for(c, upload(c, audio, model="timeline_mlp_deploy").json()["job_id"])
        assert analyzer.switched.count("timeline_mlp_deploy") == 1                       # already loaded: no needless reload
        bad = upload(c, audio, model="nope")
        assert bad.status_code == 422 and "Unknown model" in bad.json()["detail"]
        assert c.get("/api/health").json()["timeline_model"] == "timeline_mlp_deploy"

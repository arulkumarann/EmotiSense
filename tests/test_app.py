import json

import numpy as np
import pytest
import soundfile as sf
import torch

from emotisense.app.analyzer import CallAnalyzer
from emotisense.app.vad import VADConfig, chunk_segments, detect_speech
from emotisense.config import load_config
from emotisense.timeline.model import TimelineModel
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


def test_analyzer_reports_progress_and_waveform(deploy_checkpoint, tmp_path):
    make_call(tmp_path / "c.wav")
    analyzer = CallAnalyzer(deploy_checkpoint, device="cpu")
    seen = []
    result = analyzer.analyze(tmp_path / "c.wav", progress=lambda stage, fraction, detail="": seen.append((stage, fraction)))
    stages = [s for s, _ in seen]
    assert stages[0] == "decode" and "listen" in stages and stages[-1] == "timeline"
    fractions = [f for _, f in seen]
    assert fractions == sorted(fractions) and 0 < fractions[0] < fractions[-1] < 1
    assert len(result["waveform"]) == 720 and max(result["waveform"]) == 1.0 and min(result["waveform"]) >= 0.0


def test_analyzer_switches_timeline_model_without_reloading_the_encoder(deploy_checkpoint, tmp_path):
    ckpt = torch.load(deploy_checkpoint, map_location="cpu")
    mlp = TimelineModel(ckpt["in_dim"], 5, "mlp", 16, 1, 4, 0.1)
    other = tmp_path / "timeline_mlp_x.pt"
    torch.save({**ckpt, "model": mlp.state_dict(),
                "model_cfg": {"kind": "mlp", "hidden": 16, "layers": 1, "heads": 4, "dropout": 0.1}}, other)
    analyzer = CallAnalyzer(deploy_checkpoint, device="cpu")
    encoder = analyzer.embedder
    assert analyzer.timeline_kind != "mlp"

    analyzer.use_timeline(other)
    assert (analyzer.timeline_kind, analyzer.timeline_id) == ("mlp", "timeline_mlp_x")
    assert analyzer.embedder is encoder                      # same frozen encoder: nothing heavy was reloaded
    make_call(tmp_path / "c.wav")
    result = analyzer.analyze(tmp_path / "c.wav")
    assert result["model"] == {"id": "timeline_mlp_x", "kind": "mlp"} and result["chunks"]

    broken = tmp_path / "broken.pt"
    broken.write_bytes(b"this is not a checkpoint")
    with pytest.raises(Exception):
        analyzer.use_timeline(broken)
    assert analyzer.timeline_id == "timeline_mlp_x" and analyzer.analyze(tmp_path / "c.wav")["chunks"]   # still works

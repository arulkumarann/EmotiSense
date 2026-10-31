import numpy as np
import pandas as pd
import pytest
import torch

from emotisense.config import load_config
from emotisense.data.labels import INTENSITY_EMOTIONS
from emotisense.timeline.data import DialogDataset, build_dialogs, collate_dialogs
from emotisense.timeline.escalation import (EscalationConfig, analyze_timeline, classify_trend,
                                            detect_alerts, negative_score, smooth)
from emotisense.timeline.model import KINDS, TimelineModel
from emotisense.timeline.train import ccc_torch, run_timeline_cv, timeline_loss

CFG = EscalationConfig()


# ------------------------------------------------------------------ escalation
def _series(values, step=5.0):
    t = np.arange(len(values)) * step
    return t, np.array(values, dtype=float)


def test_deck_example_arcs_are_classified():
    for values, trend in (([0.20, 0.22, 0.21], "stable"), ([0.18, 0.47, 0.91], "escalating"),
                          ([0.78, 0.56, 0.28], "de-escalating")):
        t, s = _series(values)
        assert classify_trend(t, s, CFG) == trend, values


def test_threshold_alert_and_peak():
    t, s = _series([0.1, 0.2, 0.7, 0.9, 0.3, 0.1])
    alerts = detect_alerts(t, s, CFG)
    assert len(alerts) == 1
    assert alerts[0].kind == "threshold" and alerts[0].peak == 0.9 and alerts[0].peak_time == 15.0
    assert alerts[0].start == 10.0


def test_rise_alert_fires_below_threshold():
    t, s = _series([0.1, 0.1, 0.45, 0.5])          # never crosses 0.6, but jumps by 0.35
    alerts = detect_alerts(t, s, CFG)
    assert len(alerts) == 1 and alerts[0].kind == "rise"


def test_no_alerts_for_calm_call():
    t, s = _series([0.2, 0.25, 0.22, 0.3, 0.2])
    assert detect_alerts(t, s, CFG) == []


def test_alerts_merge_within_gap_and_split_beyond_it():
    t = np.array([0.0, 5.0, 10.0, 60.0, 65.0])
    s = np.array([0.9, 0.2, 0.9, 0.9, 0.9])
    alerts = detect_alerts(t, s, EscalationConfig(merge_gap_s=6.0))
    assert [(a.start, a.end) for a in alerts] == [(0.0, 0.0), (10.0, 10.0), (60.0, 65.0)] or len(alerts) >= 2
    merged = detect_alerts(t, np.array([0.9, 0.2, 0.9, 0.2, 0.2]), EscalationConfig(merge_gap_s=10.0))
    assert len(merged) == 1


def test_smoothing_modes():
    x = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    centred = smooth(x, 3)
    causal = smooth(x, 3, causal=True)
    assert centred[1] == pytest.approx(1 / 3) and centred[3] == pytest.approx(1 / 3)
    assert causal[1] == 0.0 and causal[2] == pytest.approx(1 / 3)       # the future never leaks in
    assert smooth(x, 1).tolist() == x.tolist()


def test_negative_score_combination_and_analysis():
    scores = {"anger": np.array([0.1, 0.8]), "frustration": np.array([0.5, 0.2])}
    assert negative_score(scores, EscalationConfig(combine="max")).tolist() == [0.5, 0.8]
    assert negative_score(scores, EscalationConfig(combine="mean")).tolist() == pytest.approx([0.3, 0.5])
    out = analyze_timeline(np.array([0.0, 4.0]), scores, EscalationConfig(smooth_window=1))
    assert out["peak"] == 0.8 and out["peak_time"] == 4.0 and out["n_alerts"] == 1
    empty = analyze_timeline(np.array([]), {}, CFG)
    assert empty["n_alerts"] == 0 and empty["trend"] == "stable"


def test_escalation_config_roundtrip():
    cfg = EscalationConfig.from_dict({"alert_threshold": 0.7, "emotions": ["anger"], "unknown_key": 1})
    assert cfg.alert_threshold == 0.7 and cfg.emotions == ("anger",)
    assert EscalationConfig.from_dict(cfg.to_dict()) == cfg


# --------------------------------------------------------------------- model
@pytest.mark.parametrize("kind", KINDS)
def test_timeline_model_shapes_and_padding(kind):
    torch.manual_seed(0)
    model = TimelineModel(12, 5, kind, hidden=16, layers=2, heads=4, dropout=0.0).eval()
    x = torch.randn(2, 9, 12)
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[1, 5:] = False
    out = model(x, mask)
    assert out.shape == (2, 9, 5) and out.min() >= 0 and out.max() <= 1
    solo = model(x[1:2, :5], torch.ones(1, 5, dtype=torch.bool))
    assert torch.allclose(out[1, :5], solo[0], atol=1e-4), kind        # padding must not leak in


@pytest.mark.parametrize("kind", ["lstm", "transformer_causal"])
def test_causal_models_ignore_the_future(kind):
    torch.manual_seed(0)
    model = TimelineModel(8, 5, kind, hidden=16, layers=2, heads=4, dropout=0.0).eval()
    x = torch.randn(1, 10, 8)
    m = torch.ones(1, 10, dtype=torch.bool)
    base = model(x, m)
    perturbed = x.clone()
    perturbed[:, 6:] = torch.randn(1, 4, 8) * 3
    assert torch.allclose(model(perturbed, m)[:, :6], base[:, :6], atol=1e-5)
    assert model.causal


def test_bilstm_uses_the_future():
    torch.manual_seed(0)
    model = TimelineModel(8, 5, "bilstm", hidden=16, layers=1, dropout=0.0).eval()
    x = torch.randn(1, 10, 8)
    m = torch.ones(1, 10, dtype=torch.bool)
    changed = x.clone()
    changed[:, 6:] = torch.randn(1, 4, 8) * 3          # a constant shift would be removed by the input LayerNorm
    assert not torch.allclose(model(changed, m)[:, :6], model(x, m)[:, :6], atol=1e-4)


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        TimelineModel(4, 5, "gru")


def test_loss_matches_components():
    t = torch.rand(2, 6, 5)
    mask = torch.ones(2, 6, dtype=torch.bool)
    perfect = timeline_loss(t.clone(), t, mask)
    worse = timeline_loss(torch.rand(2, 6, 5), t, mask)
    assert perfect.item() < 1e-4 < worse.item()
    flat = ccc_torch(t.reshape(-1, 5), t.reshape(-1, 5))
    assert torch.allclose(flat, torch.ones(5), atol=1e-4)


# ---------------------------------------------------------------- data + CV
def test_build_dialogs_orders_turns_and_aligns_targets(synthetic_corpus):
    root, df = synthetic_corpus
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)          # scrambled on purpose
    store = {"utt_ids": df["utt_id"].tolist(), "emb": torch.arange(len(df), dtype=torch.float32).unsqueeze(1)}
    dialogs = build_dialogs(df, store)
    assert len(dialogs) == 10
    for d in dialogs:
        assert (np.diff(d.times) >= 0).all()
        # embeddings follow the utterance ids, targets follow the same utterances
        assert [store["utt_ids"][int(v)] for v in d.x[:, 0]] == d.utt_ids
        assert d.y.shape == (len(d.utt_ids), len(INTENSITY_EMOTIONS)) and d.y.max() <= 1
    item = DialogDataset(dialogs, crop_len=3, train=True)[0]
    assert len(item["x"]) == 3
    batch = collate_dialogs([DialogDataset(dialogs)[i] for i in range(2)])
    assert batch["mask"].sum() == batch["lengths"].sum()


def test_timeline_cv_end_to_end(synthetic_corpus, tmp_path):
    root, df = synthetic_corpus
    manifest = tmp_path / "m.csv"
    df.to_csv(manifest, index=False)
    cfg = load_config("configs/timeline/smoke.yaml", [
        f"data.manifest={manifest.as_posix()}", f"data.audio_root={root.as_posix()}",
        f"embeddings.cache_dir={tmp_path.as_posix()}/cache", f"output_dir={tmp_path.as_posix()}/runs",
    ])
    summary = run_timeline_cv(cfg, run_dir=tmp_path / "tl", folds=[0, 1])
    assert summary["folds"] == [0, 1] and set(summary["per_emotion_ccc"]) == set(INTENSITY_EMOTIONS)
    assert (tmp_path / "tl" / "fold0" / "timeline.pt").exists()
    assert (tmp_path / "tl" / "summary.json").exists()
    ckpt = torch.load(summary["deploy_checkpoint"], map_location="cpu")
    assert ckpt["emotions"] == list(INTENSITY_EMOTIONS) and ckpt["embedder"]["kind"] == "frozen"
    assert list((tmp_path / "cache").glob("frozen_*.pt"))                    # embeddings were cached
    for key in ("alert_f1", "trend_accuracy"):
        assert 0.0 <= summary[key]["mean"] <= 1.0

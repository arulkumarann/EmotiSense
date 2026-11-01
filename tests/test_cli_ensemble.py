import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from emotisense.cli import build_parser, main
from emotisense.evaluation.ensemble import ensemble_runs


def _fake_run(root, probs_by_fold, y_true_by_fold, classes=("a", "b", "c")):
    for fold, probs in probs_by_fold.items():
        d = root / f"fold{fold}"
        d.mkdir(parents=True)
        table = pd.DataFrame({"utt_id": [f"u{i}" for i in range(len(probs))], "y_true": y_true_by_fold[fold],
                              "y_pred": np.argmax(probs, 1)})
        for j, c in enumerate(classes):
            table[f"p_{c}"] = probs[:, j]
        table.to_csv(d / "preds.csv", index=False)


def test_ensemble_beats_individual_runs_when_errors_differ(tmp_path):
    y = np.array([0, 1, 2, 0, 1, 2])
    run1 = np.eye(3)[[0, 1, 2, 0, 1, 0]] * 0.9 + 0.05          # wrong on the last item, confidently
    run2 = np.eye(3)[[0, 1, 2, 1, 1, 2]] * 0.9 + 0.05          # wrong on the fourth item, confidently
    run3 = np.eye(3)[[0, 1, 2, 0, 1, 2]] * 0.6 + 0.2           # always right but less confident
    for name, p in (("r1", run1), ("r2", run2), ("r3", run3)):
        _fake_run(tmp_path / name, {0: p, 1: p}, {0: y, 1: y})
    summary = ensemble_runs([tmp_path / n for n in ("r1", "r2", "r3")], tmp_path / "ens")
    assert summary["folds"] == ["fold0", "fold1"]
    assert summary["mean_std"]["wa"]["mean"] == pytest.approx(1.0)
    assert (tmp_path / "ens" / "fold0" / "preds.csv").exists()
    single = ensemble_runs([tmp_path / "r1"], tmp_path / "solo")
    assert single["mean_std"]["wa"]["mean"] == pytest.approx(5 / 6)


def test_ensemble_rejects_mismatched_runs(tmp_path):
    y = np.array([0, 1, 2])
    p = np.eye(3)
    _fake_run(tmp_path / "a", {0: p}, {0: y})
    _fake_run(tmp_path / "b", {0: p}, {0: y}, classes=("a", "b", "z"))
    with pytest.raises(ValueError, match="disagree"):
        ensemble_runs([tmp_path / "a", tmp_path / "b"], tmp_path / "out")
    _fake_run(tmp_path / "c", {1: p}, {1: y})
    with pytest.raises(ValueError, match="no finished folds"):
        ensemble_runs([tmp_path / "a", tmp_path / "c"], tmp_path / "out2")


def test_parser_exposes_every_stage():
    parser = build_parser()
    sub = next(a for a in parser._actions if a.dest == "command")
    assert {"doctor", "download", "prepare", "train", "smoke", "timeline", "ensemble", "analyze", "web"} <= set(sub.choices)
    args = parser.parse_args(["train", "--config", "x.yaml", "--folds", "0", "2", "--set", "train.epochs=3",
                              "--set", "seed=1"])
    assert args.folds == [0, 2] and args.overrides == ["train.epochs=3", "seed=1"]


def test_prepare_synthetic_and_doctor(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["prepare", "synthetic"]) == 0
    assert (tmp_path / "data/processed/synthetic/manifest.csv").exists()
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "Dataset synthetic" in out and "ready" in out


def test_download_iemocap_uses_kaggle_copy_and_records_root(tmp_path, monkeypatch, capsys):
    from test_data import _fake_iemocap

    import emotisense.data.download as dl

    cache = tmp_path / "kaggle_cache"
    _fake_iemocap(cache)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dl, "download_iemocap", lambda raw_dir="data/raw": cache)
    assert main(["download", "iemocap"]) == 0
    manifest = tmp_path / "data/processed/iemocap/manifest.csv"
    assert manifest.exists() and len(pd.read_csv(manifest)) == 15
    meta = json.loads((manifest.parent / "meta.json").read_text())
    assert Path(meta["audio_root"]) == cache.resolve()
    assert "iemocap4 benchmark subset" in capsys.readouterr().out


def test_prepare_reports_missing_data_without_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["prepare", "iemocap"]) == 1
    assert "emotisense download iemocap" in capsys.readouterr().err


def test_limit_threads_caps_but_respects_user_settings(monkeypatch):
    from emotisense.cli import limit_threads

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("os.cpu_count", lambda: 255)          # what a container reports
    limit_threads()
    assert {os.environ[v] for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")} == {"8"}
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    limit_threads()
    assert os.environ["OMP_NUM_THREADS"] == "3"               # an explicit choice is never overridden

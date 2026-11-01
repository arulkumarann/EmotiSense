import json

import pandas as pd
import pytest
import torch

from emotisense.config import load_config
from emotisense.data.labels import get_scheme
from emotisense.training import FoldTrainer, run_cv, summarize_run
from emotisense.training import cv as cv_module
from emotisense.training import trainer as trainer_module


@pytest.fixture()
def smoke_cfg(synthetic_corpus, tmp_path):
    root, df = synthetic_corpus
    manifest = tmp_path / "manifest.csv"
    df.to_csv(manifest, index=False)
    return load_config("configs/ser/smoke.yaml", [
        f"data.manifest={manifest.as_posix()}", f"data.audio_root={root.as_posix()}",
        f"output_dir={tmp_path.as_posix()}/runs", "train.epochs=2",
    ])


def test_fold_training_writes_everything(smoke_cfg, tmp_path):
    df = trainer_module.load_dataframe(smoke_cfg)
    result = FoldTrainer(smoke_cfg, df, 0, tmp_path / "fold0").fit()
    for name in ("best.pt", "config.yaml", "metrics.csv", "preds.csv", "result.json", "train.log"):
        assert (tmp_path / "fold0" / name).exists(), name
    assert not (tmp_path / "fold0" / "last.pt").exists()       # cleaned up when a fold finishes
    assert 0.0 <= result["test"]["ua"] <= 1.0
    metrics = pd.read_csv(tmp_path / "fold0" / "metrics.csv")
    assert {"val_ua", "train_loss", "epoch_seconds"} <= set(metrics.columns)
    preds = pd.read_csv(tmp_path / "fold0" / "preds.csv")
    classes = get_scheme("iemocap5")["classes"]
    assert all(f"p_{c}" in preds.columns for c in classes)
    assert preds[[f"p_{c}" for c in classes]].sum(axis=1).round(4).eq(1.0).all()


def test_training_learns_synthetic_signal(smoke_cfg, tmp_path):
    cfg = load_config("configs/ser/smoke.yaml", [
        f"data.manifest={smoke_cfg.data.manifest}", f"data.audio_root={smoke_cfg.data.audio_root}",
        "train.epochs=12", "train.patience=12", "augment.mixup_prob=0.0", "train.ema_decay=0.0",
        "data.val_level=speaker",
    ])
    df = trainer_module.load_dataframe(cfg)
    result = FoldTrainer(cfg, df, 1, tmp_path / "fold1").fit()
    # The validation speaker has only a handful of clips, so check optimisation via the training loss.
    losses = pd.read_csv(tmp_path / "fold1" / "metrics.csv")["train_loss"].dropna().tolist()
    assert losses[-1] < 0.85 * losses[0]
    assert result["best_epoch"] >= 0


def test_checkpoint_rebuilds_without_network(smoke_cfg, tmp_path):
    from emotisense.models import SERModel, SpeechEncoder

    df = trainer_module.load_dataframe(smoke_cfg)
    FoldTrainer(smoke_cfg, df, 0, tmp_path / "f").fit()
    ckpt = torch.load(tmp_path / "f" / "best.pt", map_location="cpu")
    enc = SpeechEncoder(encoder_config=ckpt["encoder_config"])
    model = SERModel(len(ckpt["classes"]), enc, pool_heads=4, head_hidden=32, center_dim=8, supcon_dim=16)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ckpt["model"].items()})
    assert ckpt["classes"] == get_scheme("iemocap5")["classes"]


def test_resume_continues_from_last_epoch(smoke_cfg, tmp_path):
    cfg = load_config("configs/ser/smoke.yaml", [
        f"data.manifest={smoke_cfg.data.manifest}", f"data.audio_root={smoke_cfg.data.audio_root}",
        "train.epochs=3", "train.keep_last=true",
    ])
    df = trainer_module.load_dataframe(cfg)
    run = tmp_path / "fold0"
    FoldTrainer(cfg, df, 0, run).fit()
    assert (run / "last.pt").exists()
    ckpt = torch.load(run / "last.pt", map_location="cpu")
    assert ckpt["epoch"] == 2
    cfg.train.epochs = 5
    FoldTrainer(cfg, df, 0, run).fit(resume=True)
    log_epochs = pd.read_csv(run / "metrics.csv")["epoch"].dropna().astype(int).tolist()
    assert log_epochs == [0, 1, 2, 3, 4]       # epochs 3 and 4 were appended, nothing repeated


def test_cv_skips_finished_folds_and_summarises(smoke_cfg, tmp_path):
    run_dir = tmp_path / "cv"
    summary = run_cv(smoke_cfg, run_dir=run_dir, folds=[0, 1])
    assert summary["folds"] == [0, 1]
    assert (run_dir / "summary.json").exists() and (run_dir / "summary.md").exists()
    stamp = (run_dir / "fold0" / "result.json").stat().st_mtime_ns
    again = run_cv(smoke_cfg, run_dir=run_dir, folds=[0, 1])
    assert (run_dir / "fold0" / "result.json").stat().st_mtime_ns == stamp   # not retrained
    assert again["mean_std"]["ua"]["mean"] == summary["mean_std"]["ua"]["mean"]
    data = json.loads((run_dir / "summary.json").read_text())
    assert data["pooled"]["n"] == len(pd.read_csv(run_dir / "fold0" / "preds.csv")) + \
        len(pd.read_csv(run_dir / "fold1" / "preds.csv"))
    assert summarize_run(tmp_path / "empty", ["a"]) is None


def test_missing_manifest_gives_actionable_error(tmp_path):
    cfg = load_config("configs/ser/smoke.yaml", [f"data.manifest={tmp_path.as_posix()}/nope.csv"])
    with pytest.raises(FileNotFoundError, match="emotisense prepare"):
        cv_module.load_dataframe(cfg)


def test_limit_per_split_caps_every_split(smoke_cfg, tmp_path):
    smoke_cfg.data.limit_per_split = 5
    df = trainer_module.load_dataframe(smoke_cfg)
    trainer = FoldTrainer(smoke_cfg, df, 0, tmp_path / "limited")
    assert all(0 < len(v) <= 5 for v in trainer.splits.values())

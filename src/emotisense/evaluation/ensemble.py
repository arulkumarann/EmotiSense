"""Average the predicted probabilities of several finished runs (same folds, same classes)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import get_logger, save_json
from .metrics import aggregate_folds, classification_metrics


def ensemble_runs(run_dirs: list[str | Path], out_dir: str | Path) -> dict:
    run_dirs = [Path(r) for r in run_dirs]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("emotisense")
    fold_sets = [{p.parent.name for p in r.glob("fold*/preds.csv")} for r in run_dirs]
    folds = sorted(set.intersection(*fold_sets))
    if not folds:
        raise ValueError("The runs share no finished folds; nothing to ensemble.")
    per_fold, pooled_true, pooled_pred = {}, [], []
    for fold in folds:
        tables = [pd.read_csv(r / fold / "preds.csv") for r in run_dirs]
        base = tables[0]
        prob_cols = [c for c in base.columns if c.startswith("p_")]
        for t in tables[1:]:
            if t["utt_id"].tolist() != base["utt_id"].tolist() or [c for c in t.columns if c.startswith("p_")] != prob_cols:
                raise ValueError(f"{fold}: runs disagree on utterances or classes; they cannot be ensembled.")
        probs = np.mean([t[prob_cols].to_numpy() for t in tables], axis=0)
        y_true, y_pred = base["y_true"].to_numpy(), probs.argmax(axis=1)
        classes = [c[2:] for c in prob_cols]
        per_fold[fold] = classification_metrics(y_true, y_pred, classes)
        pooled_true.append(y_true)
        pooled_pred.append(y_pred)
        merged = base[["utt_id", "y_true"]].copy()
        merged["y_pred"] = y_pred
        merged[prob_cols] = probs
        (out_dir / fold).mkdir(exist_ok=True)
        merged.to_csv(out_dir / fold / "preds.csv", index=False)
    pooled = classification_metrics(np.concatenate(pooled_true), np.concatenate(pooled_pred), classes)
    summary = {
        "runs": [str(r) for r in run_dirs], "folds": folds,
        "per_fold": {f: {"wa": m["wa"], "ua": m["ua"]} for f, m in per_fold.items()},
        "mean_std": aggregate_folds(list(per_fold.values())),
        "pooled": {k: pooled[k] for k in ("wa", "ua", "macro_f1", "confusion_matrix", "per_class_recall")},
    }
    save_json(summary, out_dir / "summary.json")
    ms = summary["mean_std"]
    log.info("Ensemble of %d runs over %d folds | WA %.2f +/- %.2f | UA %.2f +/- %.2f", len(run_dirs), len(folds),
             100 * ms["wa"]["mean"], 100 * ms["wa"]["std"], 100 * ms["ua"]["mean"], 100 * ms["ua"]["std"])
    return summary

"""Cross-validation driver: runs folds, skips finished ones, writes a summary."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config, save_config
from ..data.labels import get_scheme
from ..data.splits import n_folds
from ..evaluation.metrics import aggregate_folds, classification_metrics
from ..utils import get_logger, load_json, save_json
from .trainer import FoldTrainer, load_dataframe


def new_run_dir(cfg: Config) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(cfg.get("output_dir", "runs")) / cfg.experiment / stamp


def summarize_run(run_dir: Path, classes: list[str]) -> dict | None:
    """Aggregate every finished ``fold*/result.json`` under ``run_dir`` into ``summary.json/.md``."""
    results = [load_json(p) for p in sorted(run_dir.glob("fold*/result.json"))]
    if not results:
        return None
    test = [r["test"] for r in results]
    pooled_true, pooled_pred = [], []
    for p in sorted(run_dir.glob("fold*/preds.csv")):
        table = pd.read_csv(p)
        pooled_true.append(table["y_true"].to_numpy())
        pooled_pred.append(table["y_pred"].to_numpy())
    pooled = classification_metrics(np.concatenate(pooled_true), np.concatenate(pooled_pred), classes)
    summary = {
        "folds": [r["fold"] for r in results],
        "per_fold": {f"fold{r['fold']}": {"wa": r["test"]["wa"], "ua": r["test"]["ua"],
                                          "best_epoch": r["best_epoch"]} for r in results},
        "mean_std": aggregate_folds(test),
        "pooled": {k: pooled[k] for k in ("wa", "ua", "macro_f1", "per_class_recall", "confusion_matrix", "n")},
        "classes": classes,
    }
    save_json(summary, run_dir / "summary.json")
    ms = summary["mean_std"]
    lines = [f"# Results: {run_dir.parent.name}/{run_dir.name}", "",
             f"Folds finished: {len(results)}", "",
             "| Fold | WA | UA |", "|---|---|---|"]
    for r in results:
        lines.append(f"| {r['fold']} | {100 * r['test']['wa']:.2f} | {100 * r['test']['ua']:.2f} |")
    lines += ["", f"**Mean +/- std over folds:** WA {100 * ms['wa']['mean']:.2f} +/- {100 * ms['wa']['std']:.2f}"
              f" | UA {100 * ms['ua']['mean']:.2f} +/- {100 * ms['ua']['std']:.2f}",
              f"**Pooled over all test utterances:** WA {100 * pooled['wa']:.2f} | UA {100 * pooled['ua']:.2f}",
              "", "Base paper (Wang et al., arXiv:2512.04551), IEMOCAP: WA 78.47 | UA 79.14"]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run_cv(cfg: Config, run_dir: str | Path | None = None, folds: list[int] | None = None,
           resume: bool = False) -> dict | None:
    run_dir = Path(run_dir) if run_dir else new_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    log = get_logger("emotisense", run_dir / "run.log")
    df = load_dataframe(cfg)
    folds = folds if folds is not None else list(range(n_folds(cfg.data.dataset)))
    log.info("Run directory: %s | folds: %s", run_dir, folds)
    for fold in folds:
        fold_dir = run_dir / f"fold{fold}"
        if (fold_dir / "result.json").exists():
            log.info("Fold %d already finished, skipping", fold)
            continue
        FoldTrainer(cfg, df, fold, fold_dir).fit(resume=resume)
    classes = get_scheme(cfg.data.scheme)["classes"]
    summary = summarize_run(run_dir, classes)
    if summary:
        ms = summary["mean_std"]
        log.info("CV summary (%d folds): WA %.2f +/- %.2f | UA %.2f +/- %.2f", len(summary["folds"]),
                 100 * ms["wa"]["mean"], 100 * ms["wa"]["std"], 100 * ms["ua"]["mean"], 100 * ms["ua"]["std"])
    return summary

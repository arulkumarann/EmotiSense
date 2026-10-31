"""Classification (WA / UA) and regression (CCC / RMSE / MAE) metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    """WA = overall accuracy, UA = mean per-class recall (the two metrics of the base paper)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    k = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(k)))
    support = cm.sum(axis=1)
    recall = np.divide(np.diag(cm), support, out=np.zeros(k), where=support > 0)
    present = support > 0
    return {
        "wa": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "ua": float(recall[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(k)), average="macro", zero_division=0)),
        "per_class_recall": {n: float(r) for n, r in zip(class_names, recall)},
        "confusion_matrix": cm.tolist(),
        "n": int(len(y_true)),
    }


def ccc(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    """Lin's concordance correlation coefficient."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return 0.0
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    return float(2 * cov / (vx + vy + (mx - my) ** 2 + eps))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]) -> dict:
    """Per-column CCC / RMSE / MAE plus their means. Inputs are ``[N, C]``."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out: dict = {"per_emotion": {}}
    for j, name in enumerate(names):
        t, p = y_true[:, j], y_pred[:, j]
        out["per_emotion"][name] = {
            "ccc": ccc(t, p),
            "rmse": float(np.sqrt(np.mean((t - p) ** 2))),
            "mae": float(np.mean(np.abs(t - p))),
        }
    for key in ("ccc", "rmse", "mae"):
        out[f"mean_{key}"] = float(np.mean([v[key] for v in out["per_emotion"].values()]))
    out["n"] = int(len(y_true))
    return out


def aggregate_folds(fold_metrics: list[dict], keys: tuple[str, ...] = ("wa", "ua", "macro_f1")) -> dict:
    """Mean and standard deviation of scalar metrics across folds."""
    out = {}
    for key in keys:
        vals = np.array([m[key] for m in fold_metrics], dtype=np.float64)
        out[key] = {"mean": float(vals.mean()), "std": float(vals.std()), "values": vals.tolist()}
    return out

"""Train and cross-validate the timeline model on IEMOCAP dialogues."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..config import Config, save_config
from ..data.labels import INTENSITY_EMOTIONS
from ..data.manifest import load_manifest, resolve_audio_root, resolve_paths
from ..data.splits import n_folds, split_fold
from ..embedding import build_embedder, extract_embeddings
from ..evaluation.metrics import regression_metrics
from ..utils import MetricLogger, get_logger, save_json, seed_everything
from .data import DialogDataset, build_dialogs, collate_dialogs, load_store
from .escalation import EscalationConfig, analyze_timeline
from .model import TimelineModel

EMOTIONS = list(INTENSITY_EMOTIONS)


# --------------------------------------------------------------------- losses
def ccc_torch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-column Lin's CCC for ``[N, C]`` tensors."""
    pm, tm = pred.mean(0), target.mean(0)
    pv, tv = pred.var(0, unbiased=False), target.var(0, unbiased=False)
    cov = ((pred - pm) * (target - tm)).mean(0)
    return 2 * cov / (pv + tv + (pm - tm) ** 2 + eps)


def timeline_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, ccc_weight: float = 0.5):
    p, t = pred[mask], target[mask]
    mse = ((p - t) ** 2).mean()
    if p.size(0) < 2:
        return mse
    return ccc_weight * (1.0 - ccc_torch(p, t).mean()) + (1.0 - ccc_weight) * mse


# ------------------------------------------------------------------ embeddings
def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")


def get_store(cfg: Config, df: pd.DataFrame, fold: int | None = None, checkpoint: str | None = None) -> dict:
    """Load (or compute and cache) the utterance embeddings used by the timeline model."""
    e = cfg.embeddings
    cache_dir = Path(e.get("cache_dir", "data/cache"))
    if e.source == "frozen":
        key = f"frozen_{_slug(e.encoder)}{'_tiny' if e.get('tiny') else ''}_{cfg.data.dataset}.pt"
        spec = {"kind": "frozen", "encoder": e.encoder, "tiny": e.get("tiny", False)}
    elif e.source == "finetuned":
        if checkpoint is None:
            if fold is None:
                raise ValueError("Set deploy.ser_checkpoint to the SER best.pt used for the deployable timeline model.")
            checkpoint = str(Path(e.ser_run) / f"fold{fold}" / "best.pt")
        key = f"finetuned_{_slug(str(Path(checkpoint).parent.parent.name + '-' + Path(checkpoint).parent.name))}_{cfg.data.dataset}.pt"
        spec = {"kind": "finetuned", "checkpoint": checkpoint}
    else:
        raise ValueError(f"embeddings.source must be 'frozen' or 'finetuned', got {e.source!r}")
    path = cache_dir / key
    if path.exists():
        store = load_store(path)
        if store["utt_ids"] == df["utt_id"].tolist():
            return store
    cache_dir.mkdir(parents=True, exist_ok=True)
    embedder = build_embedder(spec, cfg.get("device"))
    store = extract_embeddings(embedder, df, batch_size=e.get("batch_size", 8))
    torch.save(store, path)
    return store


# -------------------------------------------------------------------- training
def _evaluate_loss(model, loader, device, ccc_weight):
    model.eval()
    total, n, preds, targets = 0.0, 0, [], []
    with torch.no_grad():
        for b in loader:
            x, y, m = b["x"].to(device), b["y"].to(device), b["mask"].to(device)
            pred = model(x, m)
            total += float(timeline_loss(pred, y, m, ccc_weight)) * int(m.sum())
            n += int(m.sum())
            preds.append(pred[m].cpu())
            targets.append(y[m].cpu())
    p, t = torch.cat(preds), torch.cat(targets)
    return total / max(n, 1), float(ccc_torch(p, t).mean()) if len(p) > 1 else 0.0


def fit_timeline(cfg: Config, train_dialogs, val_dialogs, in_dim: int, device: torch.device,
                 logger: MetricLogger | None = None, log=None) -> tuple[TimelineModel, dict]:
    t, mc = cfg.train, cfg.model
    model = TimelineModel(in_dim, len(EMOTIONS), mc.kind, mc.hidden, mc.layers, mc.heads, mc.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t.epochs, eta_min=t.lr * 0.02)
    train_loader = DataLoader(DialogDataset(train_dialogs, t.get("crop_len"), True), batch_size=t.batch_size,
                              shuffle=True, collate_fn=collate_dialogs)
    val_loader = DataLoader(DialogDataset(val_dialogs), batch_size=t.batch_size, collate_fn=collate_dialogs)
    best = {"ccc": -2.0, "epoch": -1, "state": None}
    stale = 0
    for epoch in range(t.epochs):
        model.train()
        running = 0.0
        for b in train_loader:
            x, y, m = b["x"].to(device), b["y"].to(device), b["mask"].to(device)
            loss = timeline_loss(model(x, m), y, m, t.ccc_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t.get("grad_clip", 1.0))
            opt.step()
            running += float(loss)
        sched.step()
        val_loss, val_ccc = _evaluate_loss(model, val_loader, device, t.ccc_weight)
        if logger:
            logger.log({"train_loss": running / max(len(train_loader), 1), "val_loss": val_loss,
                        "val_ccc": val_ccc, "epoch": epoch}, step=epoch)
        if val_ccc > best["ccc"]:
            best = {"ccc": val_ccc, "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        if stale >= t.patience:
            break
    model.load_state_dict(best["state"])
    if log:
        log.info("best val CCC %.4f at epoch %d", best["ccc"], best["epoch"])
    return model.eval(), {"val_ccc": best["ccc"], "best_epoch": best["epoch"]}


@torch.no_grad()
def predict_dialogs(model: TimelineModel, dialogs, device) -> list[np.ndarray]:
    model.eval()
    out = []
    for d in dialogs:
        x = d.x.unsqueeze(0).to(device)
        m = torch.ones(1, len(d.x), dtype=torch.bool, device=device)
        out.append(model(x, m)[0].cpu().numpy())
    return out


def escalation_agreement(dialogs, preds: list[np.ndarray], esc: EscalationConfig) -> dict:
    """Does the predicted timeline raise an alert / trend exactly where the annotated one does?"""
    tp = fp = fn = tn = trend_ok = 0
    for d, p in zip(dialogs, preds):
        true_scores = {e: d.y[:, j].numpy() for j, e in enumerate(EMOTIONS)}
        pred_scores = {e: p[:, j] for j, e in enumerate(EMOTIONS)}
        a = analyze_timeline(d.times, true_scores, esc, d.ends)
        b = analyze_timeline(d.times, pred_scores, esc, d.ends)
        ta, pb = a["n_alerts"] > 0, b["n_alerts"] > 0
        tp += ta and pb
        fp += (not ta) and pb
        fn += ta and (not pb)
        tn += (not ta) and (not pb)
        trend_ok += a["trend"] == b["trend"]
    n = max(len(dialogs), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"alert_precision": precision, "alert_recall": recall,
            "alert_f1": 2 * precision * recall / max(precision + recall, 1e-9),
            "alert_accuracy": (tp + tn) / n, "trend_accuracy": trend_ok / n, "n_dialogs": len(dialogs),
            "true_alert_rate": (tp + fn) / n}


def save_timeline_checkpoint(path: Path, model: TimelineModel, cfg: Config, in_dim: int, store: dict, extra: dict):
    torch.save({
        "model": model.state_dict(), "in_dim": in_dim, "emotions": EMOTIONS,
        "model_cfg": cfg.model.to_dict(), "embedder": store["spec"], "escalation": cfg.escalation.to_dict(),
        **extra,
    }, path)


def run_timeline_cv(cfg: Config, run_dir: str | Path | None = None, folds: list[int] | None = None) -> dict:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(run_dir) if run_dir else Path(cfg.get("output_dir", "runs")) / cfg.experiment / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    log = get_logger("emotisense", run_dir / "run.log")
    df = resolve_paths(load_manifest(cfg.data.manifest),
                       resolve_audio_root(cfg.data.get("audio_root"), cfg.data.manifest))
    df = df.sort_values(["session", "dialog", "start"]).reset_index(drop=True)
    device = torch.device(cfg.train.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    esc = EscalationConfig.from_dict(cfg.get("escalation"))
    folds = folds if folds is not None else list(range(n_folds(cfg.data.dataset)))
    per_fold = []
    for fold in folds:
        seed_everything(cfg.seed + fold)
        fold_dir = run_dir / f"fold{fold}"
        fold_dir.mkdir(exist_ok=True)
        masks = split_fold(df, cfg.data.dataset, fold, level="session")
        store = get_store(cfg, df, fold)
        parts = {k: build_dialogs(df[m], store) for k, m in masks.items()}
        log.info("Fold %d | dialogs train=%d val=%d test=%d", fold, *(len(parts[k]) for k in ("train", "val", "test")))
        mlog = MetricLogger(fold_dir, tensorboard=cfg.get("logging", {}).get("tensorboard", False))
        model, info = fit_timeline(cfg, parts["train"], parts["val"], store["emb"].size(1), device, mlog, log)
        mlog.close()
        preds = predict_dialogs(model, parts["test"], device)
        y_true = np.concatenate([d.y.numpy() for d in parts["test"]])
        y_pred = np.concatenate(preds)
        reg = regression_metrics(y_true, y_pred, EMOTIONS)
        agree = escalation_agreement(parts["test"], preds, esc)
        result = {"fold": fold, **info, "test": reg, "escalation": agree}
        save_json(result, fold_dir / "result.json")
        save_timeline_checkpoint(fold_dir / "timeline.pt", model, cfg, store["emb"].size(1), store, {"fold": fold})
        log.info("Fold %d TEST | CCC %.4f RMSE %.4f MAE %.4f | alert F1 %.3f trend acc %.3f", fold,
                 reg["mean_ccc"], reg["mean_rmse"], reg["mean_mae"], agree["alert_f1"], agree["trend_accuracy"])
        per_fold.append(result)

    summary = _summarize(per_fold)
    if cfg.get("deploy", {}).get("enabled", False):
        summary["deploy_checkpoint"] = str(_train_deploy_model(cfg, df, device, run_dir, log))
    save_json(summary, run_dir / "summary.json")
    log.info("Timeline CV | CCC %.4f +/- %.4f | RMSE %.4f | MAE %.4f", summary["ccc"]["mean"], summary["ccc"]["std"],
             summary["rmse"]["mean"], summary["mae"]["mean"])
    return summary


def _summarize(per_fold: list[dict]) -> dict:
    out: dict = {"folds": [r["fold"] for r in per_fold], "per_emotion_ccc": {}}
    for key, field in (("ccc", "mean_ccc"), ("rmse", "mean_rmse"), ("mae", "mean_mae")):
        vals = np.array([r["test"][field] for r in per_fold])
        out[key] = {"mean": float(vals.mean()), "std": float(vals.std()), "values": vals.tolist()}
    for e in EMOTIONS:
        vals = np.array([r["test"]["per_emotion"][e]["ccc"] for r in per_fold])
        out["per_emotion_ccc"][e] = {"mean": float(vals.mean()), "std": float(vals.std())}
    for key in ("alert_f1", "alert_accuracy", "trend_accuracy"):
        vals = np.array([r["escalation"][key] for r in per_fold])
        out[key] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return out


def _train_deploy_model(cfg: Config, df: pd.DataFrame, device, run_dir: Path, log) -> Path:
    """Train the model shipped to the analyzer on all sessions (the last one is used for validation)."""
    seed_everything(cfg.seed)
    sessions = sorted(df["session"].unique())
    val_session = sessions[-1]
    store = get_store(cfg, df, checkpoint=cfg.deploy.get("ser_checkpoint"))
    train_d = build_dialogs(df[df["session"] != val_session], store)
    val_d = build_dialogs(df[df["session"] == val_session], store)
    model, info = fit_timeline(cfg, train_d, val_d, store["emb"].size(1), device, None, log)
    path = run_dir / "timeline_deploy.pt"
    save_timeline_checkpoint(path, model, cfg, store["emb"].size(1), store, {"fold": "deploy", **info})
    log.info("Deployment timeline model saved to %s", path)
    return path

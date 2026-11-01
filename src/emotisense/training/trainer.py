"""Single-fold training of the SER model: AMP, accumulation, EMA, resume, early stopping."""
from __future__ import annotations

import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..config import Config, save_config
from ..data.datasets import Collator, SERDataset
from ..data.labels import get_scheme
from ..data.manifest import load_manifest, resolve_audio_root, resolve_paths
from ..data.sampler import BucketBatchSampler
from ..data.splits import split_fold
from ..evaluation.metrics import classification_metrics
from ..losses import MultiLoss
from ..models import build_ser_model
from ..utils import MetricLogger, get_logger, git_hash, save_json, seed_everything
from .ema import ModelEMA
from .optim import build_param_groups, lr_factor


def pick_device(cfg_train) -> torch.device:
    name = cfg_train.get("device")
    return torch.device(name) if name else torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_context(device: torch.device, precision: str):
    """Returns ``(context_factory, use_scaler)`` for the requested precision."""
    if device.type != "cuda" or precision == "fp32":
        return (lambda: nullcontext()), False
    if precision == "bf16" and torch.cuda.is_bf16_supported():
        return (lambda: torch.autocast("cuda", dtype=torch.bfloat16)), False
    return (lambda: torch.autocast("cuda", dtype=torch.float16)), True


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def build_loaders(cfg: Config, splits: dict[str, pd.DataFrame]):
    d, t = cfg.data, cfg.train
    common = dict(scheme=d.scheme, sample_rate=d.sample_rate)
    sets = {
        "train": SERDataset(splits["train"], max_seconds=d.train_max_seconds, train=True,
                            soft_label_alpha=d.get("soft_label_alpha", 0.0), **common),
        "val": SERDataset(splits["val"], max_seconds=d.eval_max_seconds, train=False, **common),
        "test": SERDataset(splits["test"], max_seconds=d.eval_max_seconds, train=False, **common),
    }
    aug = cfg.get("augment", {})
    train_collate = Collator(train=True, normalize=d.normalize, mixup_prob=aug.get("mixup_prob", 0.0),
                             snr_range=aug.get("snr_range", (-5.0, 10.0)),
                             min_frac=aug.get("min_frac", 0.1), max_frac=aug.get("max_frac", 0.5))
    eval_collate = Collator(train=False, normalize=d.normalize)
    pin = torch.cuda.is_available()
    nw = int(t.get("num_workers", 4))
    kwargs = dict(num_workers=nw, pin_memory=pin, persistent_workers=nw > 0)

    def by_length(ds: SERDataset) -> list[int]:
        # Similar lengths per eval batch = minimal padding (matters for group-norm base models).
        return sorted(range(len(ds)), key=lambda i: ds.df.at[i, "duration"])
    drop_last = len(sets["train"]) > t.batch_size
    if t.get("bucket_by_length", True):      # similar-length batches: roughly halves the padding waste
        lengths = sets["train"].df["duration"].clip(upper=d.train_max_seconds).to_numpy()
        train_loader = DataLoader(sets["train"], collate_fn=train_collate, batch_sampler=BucketBatchSampler(
            lengths, t.batch_size, drop_last=drop_last, seed=cfg.seed), **kwargs)
    else:
        train_loader = DataLoader(sets["train"], batch_size=t.batch_size, shuffle=True, drop_last=drop_last,
                                  collate_fn=train_collate, **kwargs)
    loaders = {
        "train": train_loader,
        "val": DataLoader(sets["val"], batch_size=t.eval_batch_size, sampler=by_length(sets["val"]),
                          collate_fn=eval_collate, **kwargs),
        "test": DataLoader(sets["test"], batch_size=t.eval_batch_size, sampler=by_length(sets["test"]),
                           collate_fn=eval_collate, **kwargs),
    }
    return sets, loaders


@torch.no_grad()
def predict(model, loader: DataLoader, device: torch.device, autocast) -> dict[str, np.ndarray]:
    """Run the model over ``loader``; outputs are ordered by dataset index."""
    model.eval()
    idxs, probs, vads = [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        with autocast():
            out = model(batch["wave"], batch["attn"], batch["lengths"])
        idxs.append(batch["idx"].cpu())
        probs.append(torch.softmax(out["logits"].float(), dim=-1).cpu())
        vads.append(out["vad"].float().cpu())
    order = torch.cat(idxs).argsort()
    return {"probs": torch.cat(probs)[order].numpy(), "vad": torch.cat(vads)[order].numpy()}


def evaluate(model, loader, dataset: SERDataset, device, autocast, classes: list[str]) -> tuple[dict, dict]:
    pred = predict(model, loader, device, autocast)
    metrics = classification_metrics(dataset.hard, pred["probs"].argmax(axis=1), classes)
    return metrics, pred


class FoldTrainer:
    def __init__(self, cfg: Config, df: pd.DataFrame, fold: int, run_dir: str | Path):
        self.cfg = cfg
        self.fold = fold
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log = get_logger("emotisense", self.run_dir / "train.log")
        self.device = pick_device(cfg.train)
        self.classes = get_scheme(cfg.data.scheme)["classes"]
        masks = split_fold(df, cfg.data.dataset, fold, level=cfg.data.get("val_level", "speaker"))
        self.splits = {name: df[m].reset_index(drop=True) for name, m in masks.items()}
        limit = cfg.data.get("limit_per_split")       # dry runs: a fixed random sample of every split
        if limit:
            self.splits = {k: v.sample(n=min(int(limit), len(v)), random_state=0).reset_index(drop=True)
                           for k, v in self.splits.items()}
        self.log.info("Fold %d | train=%d val=%d test=%d utterances", fold,
                      *(len(self.splits[s]) for s in ("train", "val", "test")))

    # ------------------------------------------------------------------ setup
    def _build(self):
        cfg, t = self.cfg, self.cfg.train
        self.sets, self.loaders = build_loaders(cfg, self.splits)
        k = len(self.classes)
        self.model = build_ser_model(cfg, k, pretrained=not cfg.encoder.get("tiny", False)).to(self.device)
        lw = cfg.loss
        self.criterion = MultiLoss(k, cfg.pool.get("center_dim", 64), lw.weights, lw.get("focal_gamma", 2.0),
                                   lw.get("supcon_temperature", 0.1), lw.get("supcon_max_frames", 32)).to(self.device)
        groups = build_param_groups(self.model, self.criterion, t)
        self.group_kinds = [g.pop("kind") for g in groups]
        self._head_group = self.group_kinds.index("head")
        self.optimizer = torch.optim.AdamW(groups, betas=tuple(t.get("betas", (0.9, 0.999))))
        accum = int(t.get("accum_steps", 1))
        self.steps_per_epoch = max(math.ceil(len(self.loaders["train"]) / accum), 1)
        self.total_steps = self.steps_per_epoch * t.epochs
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            [lambda s: lr_factor(s, self.steps_per_epoch, self.total_steps, t.schedule)] * len(groups),
        )
        self.autocast, use_scaler = autocast_context(self.device, t.get("precision", "bf16"))
        scaler_cls = getattr(torch.amp, "GradScaler", None)
        self.scaler = (scaler_cls("cuda", enabled=use_scaler) if scaler_cls
                       else torch.cuda.amp.GradScaler(enabled=use_scaler))
        self.ema = ModelEMA(self.model, t.ema_decay) if t.get("ema_decay", 0) > 0 else None
        self._encoder_params = [p for n, p in self.model.named_parameters()
                                if n.startswith("encoder.model.") and p.requires_grad]
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        self.log.info("Model: %.1fM parameters | device=%s | precision=%s", n_params, self.device,
                      t.get("precision", "bf16"))

    def _set_encoder_trainable(self, trainable: bool) -> None:
        for p in self._encoder_params:
            p.requires_grad_(trainable)

    # ------------------------------------------------------------ checkpoints
    def _checkpoint_payload(self, epoch: int, best: dict) -> dict:
        return {
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(), "scaler": self.scaler.state_dict(),
            "ema": self.ema.state_dict() if self.ema else None, "criterion": self.criterion.state_dict(),
            "epoch": epoch, "best": best,
        }

    def _save_best(self, epoch: int, metrics: dict) -> None:
        state = self.model.state_dict()   # EMA weights are applied when this is called
        half = self.cfg.train.get("save_dtype", "fp16") == "fp16"
        state = {k: (v.half() if half and v.dtype == torch.float32 else v).cpu() for k, v in state.items()}
        torch.save({
            "model": state, "config": self.cfg.to_dict(), "classes": self.classes, "fold": self.fold,
            "encoder_config": self.model.encoder.config.to_dict(), "epoch": epoch,
            "val": metrics, "git": git_hash(),
        }, self.run_dir / "best.pt")

    # ------------------------------------------------------------------- fit
    def fit(self, resume: bool = False) -> dict:
        cfg, t = self.cfg, self.cfg.train
        seed_everything(cfg.seed + self.fold, cfg.get("deterministic", False))
        save_config(cfg, self.run_dir / "config.yaml")
        self._build()
        mlog = MetricLogger(self.run_dir, tensorboard=cfg.logging.get("tensorboard", False),
                            wandb_cfg=cfg.logging.get("wandb"), config=cfg.to_dict(),
                            name=f"{cfg.experiment}-fold{self.fold}")
        start_epoch, best = 0, {"score": -1.0, "epoch": -1}
        last_path = self.run_dir / "last.pt"
        if resume and last_path.exists():
            ckpt = torch.load(last_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.scaler.load_state_dict(ckpt["scaler"])
            self.criterion.load_state_dict(ckpt["criterion"])
            if self.ema and ckpt.get("ema"):
                self.ema.load_state_dict(ckpt["ema"])
            start_epoch, best = ckpt["epoch"] + 1, ckpt["best"]
            self.log.info("Resumed from epoch %d (best %s=%.4f)", start_epoch, t.monitor, best["score"])

        accum = int(t.get("accum_steps", 1))
        stale = 0
        step = start_epoch * self.steps_per_epoch
        for epoch in range(start_epoch, t.epochs):
            started = time.time()
            self._set_encoder_trainable(epoch >= int(t.get("encoder_freeze_epochs", 0)))
            self.model.train()
            running: dict[str, float] = {}
            n_batches = 0
            self.optimizer.zero_grad(set_to_none=True)
            loader = self.loaders["train"]
            for i, batch in enumerate(loader):
                batch = to_device(batch, self.device)
                with self.autocast():
                    out = self.model(batch["wave"], batch["attn"], batch["lengths"])
                    loss, logs = self.criterion(out, batch)
                if not torch.isfinite(loss):
                    self.log.warning("Non-finite loss at epoch %d batch %d; skipping batch", epoch, i)
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                self.scaler.scale(loss / accum).backward()
                if (i + 1) % accum == 0 or (i + 1) == len(loader):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), t.get("grad_clip", 5.0))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    if self.ema:
                        self.ema.update(self.model)
                    step += 1
                for key, val in logs.items():
                    running[key] = running.get(key, 0.0) + val
                n_batches += 1
                if t.get("log_every", 0) and (i + 1) % t.log_every == 0:
                    self.log.info("epoch %d [%d/%d] loss=%.4f", epoch, i + 1, len(loader), logs["loss"])

            if self.ema:
                self.ema.apply(self.model)
            val_metrics, _ = evaluate(self.model, self.loaders["val"], self.sets["val"], self.device,
                                      self.autocast, self.classes)
            score = val_metrics[t.get("monitor", "ua")]
            improved = score > best["score"]
            if improved:
                best = {"score": score, "epoch": epoch}
                self._save_best(epoch, val_metrics)
                stale = 0
            else:
                stale += 1
            if self.ema:
                self.ema.restore(self.model)

            row = {f"train_{k}": v / max(n_batches, 1) for k, v in running.items()}
            row.update({"epoch": epoch, "val_wa": val_metrics["wa"], "val_ua": val_metrics["ua"],
                        "val_macro_f1": val_metrics["macro_f1"], "lr_head": self.optimizer.param_groups[self._head_group]["lr"],
                        "epoch_seconds": time.time() - started})
            mlog.log(row, step=epoch)
            self.log.info("epoch %d | loss %.4f | val WA %.4f UA %.4f%s | %.0fs", epoch,
                          row.get("train_loss", float("nan")), val_metrics["wa"], val_metrics["ua"],
                          "  *best*" if improved else "", row["epoch_seconds"])
            if t.get("save_last", True):
                torch.save(self._checkpoint_payload(epoch, best), last_path)
            if stale >= int(t.get("patience", 1_000_000)):
                self.log.info("Early stopping: no improvement for %d epochs", stale)
                break

        result = self._finalize(best)
        mlog.log({"test_wa": result["test"]["wa"], "test_ua": result["test"]["ua"]}, step=t.epochs)
        mlog.close()
        if not t.get("keep_last", False) and last_path.exists():
            last_path.unlink()
        return result

    # ------------------------------------------------------------- final test
    def _finalize(self, best: dict) -> dict:
        ckpt = torch.load(self.run_dir / "best.pt", map_location=self.device)
        self.model.load_state_dict({k: v.float() if v.is_floating_point() else v
                                    for k, v in ckpt["model"].items()})
        val_metrics, _ = evaluate(self.model, self.loaders["val"], self.sets["val"], self.device,
                                  self.autocast, self.classes)
        test_metrics, pred = evaluate(self.model, self.loaders["test"], self.sets["test"], self.device,
                                      self.autocast, self.classes)
        ds = self.sets["test"]
        table = pd.DataFrame({"utt_id": ds.df["utt_id"], "y_true": ds.hard, "y_pred": pred["probs"].argmax(1)})
        for j, name in enumerate(self.classes):
            table[f"p_{name}"] = pred["probs"][:, j]
        table.to_csv(self.run_dir / "preds.csv", index=False)
        result = {"fold": self.fold, "best_epoch": best["epoch"], "val": val_metrics, "test": test_metrics,
                  "classes": self.classes}
        save_json(result, self.run_dir / "result.json")
        self.log.info("Fold %d TEST | WA %.4f UA %.4f | best epoch %d", self.fold, test_metrics["wa"],
                      test_metrics["ua"], best["epoch"])
        return result


def load_dataframe(cfg: Config) -> pd.DataFrame:
    df = load_manifest(cfg.data.manifest)
    return resolve_paths(df, resolve_audio_root(cfg.data.get("audio_root"), cfg.data.manifest))

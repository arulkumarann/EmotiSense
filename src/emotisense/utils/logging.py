"""Console / file logging and a multi-backend metric logger (CSV, TensorBoard, W&B)."""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"


def get_logger(name: str = "emotisense", log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(getattr(h, "_emotisense_console", False) for h in logger.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(_FORMAT, "%H:%M:%S"))
        console._emotisense_console = True  # type: ignore[attr-defined]
        logger.addHandler(console)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # Replace any previous file handler so each run writes to its own file.
        for h in list(logger.handlers):
            if getattr(h, "_emotisense_file", False):
                logger.removeHandler(h)
                h.close()
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FORMAT))
        fh._emotisense_file = True  # type: ignore[attr-defined]
        logger.addHandler(fh)
    return logger


class MetricLogger:
    """Writes scalar metrics to ``metrics.csv`` and, optionally, TensorBoard / W&B.

    The CSV is always written so a run is inspectable without any tracking service.
    """

    def __init__(self, run_dir: str | Path, tensorboard: bool = False, wandb_cfg: dict | None = None,
                 config: dict | None = None, name: str | None = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "metrics.csv"
        self._rows: list[dict] = []
        self._fieldnames: list[str] = []
        self._tb = None
        self._wandb = None
        if self.csv_path.exists():
            with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self._fieldnames = list(reader.fieldnames or [])
                self._rows = list(reader)
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(str(self.run_dir / "tb"))
            except Exception as exc:  # tensorboard is optional
                get_logger().warning("TensorBoard disabled: %s", exc)
        if wandb_cfg and wandb_cfg.get("enabled"):
            try:
                import wandb
                self._wandb = wandb.init(
                    project=wandb_cfg.get("project", "emotisense"),
                    entity=wandb_cfg.get("entity"),
                    name=name,
                    group=wandb_cfg.get("group"),
                    config=config,
                    dir=str(self.run_dir),
                    reinit=True,
                )
            except Exception as exc:  # offline machines should still train
                get_logger().warning("W&B disabled: %s", exc)

    def log(self, metrics: dict, step: int, flush_csv: bool = True) -> None:
        row = {"step": step, **{k: _scalar(v) for k, v in metrics.items()}}
        self._rows.append(row)
        for k in row:
            if k not in self._fieldnames:
                self._fieldnames.append(k)
        if self._tb is not None:
            for k, v in row.items():
                if k != "step" and isinstance(v, (int, float)):
                    self._tb.add_scalar(k, v, step)
        if self._wandb is not None:
            self._wandb.log({k: v for k, v in row.items() if k != "step"}, step=step)
        if flush_csv:
            self.flush()

    def flush(self) -> None:
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
            for row in self._rows:
                writer.writerow(row)

    def close(self) -> None:
        self.flush()
        if self._tb is not None:
            self._tb.close()
        if self._wandb is not None:
            self._wandb.finish()


def _scalar(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v

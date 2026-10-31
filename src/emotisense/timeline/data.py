"""Dialog sequences: ordered utterance embeddings with annotator-agreement intensity targets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..data.labels import intensity_targets


@dataclass
class Dialog:
    dialog_id: str
    utt_ids: list
    times: np.ndarray         # utterance start (s)
    ends: np.ndarray          # utterance end (s)
    x: torch.Tensor           # [L, D]
    y: torch.Tensor           # [L, C]


def load_store(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu")


def build_dialogs(df: pd.DataFrame, store: dict) -> list[Dialog]:
    """Group ``df`` by dialog, order by start time and attach embeddings + targets."""
    index = {u: i for i, u in enumerate(store["utt_ids"])}
    targets = intensity_targets(df)
    dialogs = []
    df = df.reset_index(drop=True)
    for dialog_id, rows in df.groupby("dialog", sort=True):
        rows = rows.sort_values("start")
        emb_idx = [index[u] for u in rows["utt_id"]]
        dialogs.append(Dialog(
            dialog_id=dialog_id, utt_ids=rows["utt_id"].tolist(),
            times=rows["start"].to_numpy(dtype=np.float64), ends=rows["end"].to_numpy(dtype=np.float64),
            x=store["emb"][emb_idx].float(), y=torch.from_numpy(targets[rows.index.to_numpy()]).float(),
        ))
    return dialogs


class DialogDataset(Dataset):
    """Training uses random windows of ``crop_len`` turns as augmentation; evaluation uses whole dialogs."""

    def __init__(self, dialogs: list[Dialog], crop_len: int | None = None, train: bool = False):
        self.dialogs, self.crop_len, self.train = dialogs, crop_len, train

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, i: int):
        d = self.dialogs[i]
        x, y = d.x, d.y
        if self.train and self.crop_len and len(x) > self.crop_len:
            start = int(torch.randint(0, len(x) - self.crop_len + 1, (1,)))
            x, y = x[start:start + self.crop_len], y[start:start + self.crop_len]
        return {"x": x, "y": y, "i": i}


def collate_dialogs(items: list[dict]) -> dict:
    lengths = torch.tensor([len(it["x"]) for it in items])
    b, length = len(items), int(lengths.max())
    x = torch.zeros(b, length, items[0]["x"].size(1))
    y = torch.zeros(b, length, items[0]["y"].size(1))
    mask = torch.zeros(b, length, dtype=torch.bool)
    for k, it in enumerate(items):
        n = len(it["x"])
        x[k, :n], y[k, :n], mask[k, :n] = it["x"], it["y"], True
    return {"x": x, "y": y, "mask": mask, "lengths": lengths, "i": torch.tensor([it["i"] for it in items])}

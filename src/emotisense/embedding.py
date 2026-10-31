"""Turn raw audio into utterance embeddings, shared by timeline training and the call analyzer.

Two embedders exist:

* ``frozen``     - a pretrained speech encoder, layers averaged and mean-pooled. Leak-free
                   for any dataset because it never sees emotion labels.
* ``finetuned``  - the pooled embedding (plus class probabilities) of a trained SER checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import Config
from .data.audio import crop, load_audio, normalize_wave
from .models import SERModel, SpeechEncoder, build_ser_model


def _pad(waves: list[np.ndarray], device: torch.device):
    lengths = torch.tensor([len(w) for w in waves], dtype=torch.long)
    batch = torch.zeros(len(waves), int(lengths.max()))
    attn = torch.zeros(len(waves), int(lengths.max()), dtype=torch.long)
    for i, w in enumerate(waves):
        batch[i, :len(w)] = torch.from_numpy(w)
        attn[i, :len(w)] = 1
    return batch.to(device), attn.to(device), lengths.to(device)


def load_ser_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[SERModel, dict]:
    """Rebuild a trained SER model from ``best.pt`` without any network access."""
    ckpt = torch.load(path, map_location="cpu")
    cfg = Config(ckpt["config"])
    model = build_ser_model(cfg, len(ckpt["classes"]), pretrained=False, encoder_config=ckpt["encoder_config"])
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ckpt["model"].items()})
    return model.to(device).eval(), ckpt


class Embedder:
    """Base class: ``embed(waves)`` returns ``(embeddings [N, D], probs [N, K] or None)``."""

    sample_rate = 16000
    normalize = True
    max_seconds = 15.0
    spec: dict = {}
    dim: int = 0
    classes: list[str] | None = None

    def embed(self, waves: list[np.ndarray], batch_size: int = 8):
        raise NotImplementedError

    def _prepare(self, wave: np.ndarray) -> np.ndarray:
        wave = crop(wave, int(self.max_seconds * self.sample_rate), random=False)
        return normalize_wave(wave) if self.normalize else wave


class FrozenEmbedder(Embedder):
    def __init__(self, encoder: str = "microsoft/wavlm-large", device: str | None = None, tiny: bool = False):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = SpeechEncoder(name=encoder, tiny=tiny, layer_weighting=False,
                                     freeze_feature_extractor=True).to(self.device).eval()
        self.dim = self.encoder.hidden_size
        self.spec = {"kind": "frozen", "encoder": encoder, "tiny": tiny}

    @torch.no_grad()
    def embed(self, waves, batch_size: int = 8):
        waves = [self._prepare(w) for w in waves]
        order = np.argsort([len(w) for w in waves])           # similar lengths per batch = less padding
        out = torch.zeros(len(waves), self.dim)
        for lo in range(0, len(waves), batch_size):
            idx = order[lo:lo + batch_size]
            wave, attn, lengths = _pad([waves[i] for i in idx], self.device)
            frames, mask = self.encoder(wave, attn, lengths, uniform_layers=True)
            m = mask.unsqueeze(-1).to(frames.dtype)
            out[torch.as_tensor(idx)] = ((frames * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu()
        return out, None


class FinetunedEmbedder(Embedder):
    def __init__(self, checkpoint: str | Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, ckpt = load_ser_checkpoint(checkpoint, self.device)
        cfg = Config(ckpt["config"])
        self.normalize = cfg.data.normalize
        self.sample_rate = cfg.data.sample_rate
        self.max_seconds = cfg.data.eval_max_seconds
        self.classes = list(ckpt["classes"])
        self.dim = self.model.encoder.hidden_size
        self.spec = {"kind": "finetuned", "checkpoint": str(checkpoint)}

    @torch.no_grad()
    def embed(self, waves, batch_size: int = 8):
        waves = [self._prepare(w) for w in waves]
        order = np.argsort([len(w) for w in waves])
        emb = torch.zeros(len(waves), self.dim)
        probs = torch.zeros(len(waves), len(self.classes))
        for lo in range(0, len(waves), batch_size):
            idx = order[lo:lo + batch_size]
            wave, attn, lengths = _pad([waves[i] for i in idx], self.device)
            out = self.model(wave, attn, lengths)
            sel = torch.as_tensor(idx)
            emb[sel] = out["emb"].float().cpu()
            probs[sel] = torch.softmax(out["logits"].float(), -1).cpu()
        return emb, probs


def build_embedder(spec: dict, device: str | None = None) -> Embedder:
    if spec["kind"] == "frozen":
        return FrozenEmbedder(spec["encoder"], device, spec.get("tiny", False))
    if spec["kind"] == "finetuned":
        return FinetunedEmbedder(spec["checkpoint"], device)
    raise ValueError(f"Unknown embedder kind {spec['kind']!r}")


def extract_embeddings(embedder: Embedder, df: pd.DataFrame, batch_size: int = 8, chunk: int = 256) -> dict:
    """Embed every row of ``df`` (absolute ``path`` column). Returns a cache dict."""
    embs, probs = [], []
    for lo in tqdm(range(0, len(df), chunk), desc="embedding", unit="chunk"):
        part = df.iloc[lo:lo + chunk]
        waves = [load_audio(p, embedder.sample_rate) for p in part["path"]]
        e, p = embedder.embed(waves, batch_size)
        embs.append(e)
        probs.append(p)
    return {
        "utt_ids": df["utt_id"].tolist(),
        "emb": torch.cat(embs),
        "probs": torch.cat(probs) if probs and probs[0] is not None else None,
        "classes": embedder.classes,
        "spec": embedder.spec,
    }

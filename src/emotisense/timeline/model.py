"""Temporal heads over a sequence of utterance / chunk embeddings -> per-step emotion intensities."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

KINDS = ("mlp", "lstm", "bilstm", "transformer", "transformer_causal")


class TimelineModel(nn.Module):
    """``mlp`` has no temporal context (the E4 ablation); ``lstm`` and ``transformer_causal`` only look
    backwards and are the variants usable for live, in-call scoring."""

    def __init__(self, in_dim: int, out_dim: int = 5, kind: str = "bilstm", hidden: int = 256,
                 layers: int = 2, heads: int = 4, dropout: float = 0.3, max_len: int = 4096):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"Unknown timeline model {kind!r}; choose from {KINDS}")
        self.kind, self.hidden = kind, hidden
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        if kind == "mlp":
            self.core = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
            width = hidden
        elif kind in ("lstm", "bilstm"):
            self.core = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True,
                                bidirectional=kind == "bilstm", dropout=dropout if layers > 1 else 0.0)
            width = hidden * (2 if kind == "bilstm" else 1)
        else:
            layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=hidden * 2, dropout=dropout,
                                               batch_first=True, norm_first=True, activation="gelu")
            self.core = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            pe = torch.zeros(max_len, hidden)
            pos = torch.arange(max_len).unsqueeze(1)
            div = torch.exp(torch.arange(0, hidden, 2) * (-math.log(10000.0) / hidden))
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
            self.register_buffer("pe", pe, persistent=False)
            width = hidden
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, out_dim))

    @property
    def causal(self) -> bool:
        return self.kind in ("lstm", "transformer_causal", "mlp")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``x`` [B, L, D], ``mask`` [B, L] bool (True = real step). Returns [B, L, C] in (0, 1)."""
        h = self.inp(x)
        length = h.size(1)
        if self.kind == "mlp":
            h = self.core(h)
        elif self.kind in ("lstm", "bilstm"):
            lengths = mask.sum(1).clamp(min=1).cpu()
            packed = pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
            h, _ = self.core(packed)
            h, _ = pad_packed_sequence(h, batch_first=True, total_length=length)
        else:
            h = h + self.pe[:length].unsqueeze(0)
            attn_mask = None
            if self.kind == "transformer_causal":
                attn_mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=h.device), 1)
            h = self.core(h, mask=attn_mask, src_key_padding_mask=~mask)
        return torch.sigmoid(self.out(h))

"""Frame-level attention pooling (base paper): residual MHSA followed by attention pooling."""
from __future__ import annotations

import torch
import torch.nn as nn


class FrameAttentionPool(nn.Module):
    """``F' = F + MSA(F)``; ``alpha_t = softmax(F'_t . w)``; ``f = sum_t alpha_t F'_t``."""

    def __init__(self, dim: int, heads: int = 16, dropout: float = 0.1):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"Encoder dim {dim} is not divisible by pool.heads={heads}")
        self.msa = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, frames: torch.Tensor, mask: torch.Tensor):
        """``frames`` [B, T, D]; ``mask`` [B, T] bool (True = valid). Returns (pooled, frames', alpha)."""
        attended, _ = self.msa(frames, frames, frames, key_padding_mask=~mask, need_weights=False)
        frames = self.norm(frames + attended)
        scores = self.score(frames).squeeze(-1).masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(scores.float(), dim=1).to(frames.dtype)
        pooled = (alpha.unsqueeze(-1) * frames).sum(dim=1)
        return pooled, frames, alpha

"""Utterance-level speech emotion model (WavLM + frame attention + multi-task heads)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import SpeechEncoder
from .pooling import FrameAttentionPool


class SERModel(nn.Module):
    """Outputs per utterance:

    * ``logits``  - class scores for the KL / focal losses
    * ``z``       - 64-d projection for the center loss
    * ``h``       - L2-normalised per-frame projection (with context broadcasting) for SupCon
    * ``vad``     - valence/arousal/dominance regression (auxiliary multi-task head)
    * ``emb``     - pooled utterance embedding, reused by the timeline model
    """

    def __init__(self, num_classes: int, encoder: SpeechEncoder, pool_heads: int = 16,
                 dropout: float = 0.1, head_hidden: int = 256, center_dim: int = 64,
                 supcon_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        dim = encoder.hidden_size
        self.num_classes = num_classes
        self.pool = FrameAttentionPool(dim, heads=pool_heads, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(dim, head_hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(head_hidden, num_classes),
        )
        self.center_proj = nn.Linear(dim, center_dim)
        self.frame_proj = nn.Sequential(nn.Linear(dim, supcon_dim), nn.GELU(), nn.Linear(supcon_dim, supcon_dim))
        self.vad_head = nn.Sequential(nn.Linear(dim, head_hidden // 2), nn.GELU(), nn.Linear(head_hidden // 2, 3))

    def forward(self, wave: torch.Tensor, attn: torch.Tensor, lengths: torch.Tensor | None = None) -> dict:
        frames, mask = self.encoder(wave, attn, lengths)
        pooled, frames, alpha = self.pool(frames, mask)
        h = self.frame_proj(frames)
        mask_f = mask.unsqueeze(-1).to(h.dtype)
        mean = (h * mask_f).sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        h = F.normalize(0.5 * (h + mean), dim=-1)           # context broadcasting
        return {
            "logits": self.classifier(pooled),
            "z": self.center_proj(pooled),
            "h": h,
            "mask": mask,
            "vad": torch.sigmoid(self.vad_head(pooled)),
            "emb": pooled,
            "alpha": alpha,
        }

    @torch.no_grad()
    def encoder_embedding(self, wave: torch.Tensor, attn: torch.Tensor,
                          lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Mean-pooled, uniform-layer-average embedding of the encoder alone (frozen baseline)."""
        frames, mask = self.encoder(wave, attn, lengths, uniform_layers=True)
        m = mask.unsqueeze(-1).to(frames.dtype)
        return (frames * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def build_ser_model(cfg, num_classes: int, pretrained: bool = True,
                    encoder_config: dict | None = None) -> SERModel:
    e, p = cfg.encoder, cfg.pool
    encoder = SpeechEncoder(
        name=e.name, pretrained=pretrained, encoder_config=encoder_config, tiny=e.get("tiny", False),
        freeze_feature_extractor=e.get("freeze_feature_extractor", True),
        layer_weighting=e.get("layer_weighting", True),
        gradient_checkpointing=e.get("gradient_checkpointing", False),
        layerdrop=e.get("layerdrop", 0.0), spec_augment=e.get("spec_augment", False),
        mask_time_prob=e.get("mask_time_prob", 0.05),
    )
    return SERModel(num_classes, encoder, pool_heads=p.heads, dropout=p.dropout,
                    head_hidden=p.get("head_hidden", 256), center_dim=p.get("center_dim", 64),
                    supcon_dim=p.get("supcon_dim", 128))

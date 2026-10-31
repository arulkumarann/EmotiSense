"""Multi-loss objective: KL + focal + center + frame-level SupCon (+ auxiliary VAD regression)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.encoder import FRAME_STRIDE


def kl_loss(logits: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    """KL(y || y_hat) against the (possibly mixed) soft label."""
    return F.kl_div(F.log_softmax(logits.float(), dim=-1), soft, reduction="batchmean")


def focal_loss(logits: torch.Tensor, soft: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """``-sum_c y_c (1 - p_c)^gamma log p_c`` averaged over the batch."""
    log_p = F.log_softmax(logits.float(), dim=-1)
    p = log_p.exp()
    return -(soft * (1.0 - p).pow(gamma) * log_p).sum(dim=-1).mean()


class CenterLoss(nn.Module):
    """``(1/B) sum_b ||z_b - c_{y_b}||^2`` with learnable class centers."""

    def __init__(self, num_classes: int, dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, dim) * 0.1)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return (z.float() - self.centers[labels]).pow(2).sum(dim=-1).mean()


def frame_labels(mask: torch.Tensor, hard: torch.Tensor, span: torch.Tensor,
                 span_label: torch.Tensor) -> torch.Tensor:
    """Per-frame class labels ``[B, T]``; frames inside a louder mixed-in segment take its label."""
    b, t = mask.shape
    labels = hard.unsqueeze(1).expand(b, t).clone()
    centers = torch.arange(t, device=mask.device) * FRAME_STRIDE + FRAME_STRIDE // 2
    inside = (centers[None, :] >= span[:, :1]) & (centers[None, :] < span[:, 1:2]) & (span_label[:, None] >= 0)
    labels = torch.where(inside, span_label[:, None].expand(b, t), labels)
    return labels


def supcon_loss(h: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, temperature: float = 0.1,
                max_frames: int = 32) -> torch.Tensor:
    """Supervised contrastive loss over frames sampled from every utterance in the batch."""
    feats, labs = [], []
    for b in range(h.size(0)):
        idx = mask[b].nonzero(as_tuple=True)[0]
        if idx.numel() > max_frames:
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[:max_frames]]
        feats.append(h[b, idx])
        labs.append(labels[b, idx])
    feats = torch.cat(feats).float()
    labs = torch.cat(labs)
    n = feats.size(0)
    if n < 2:
        return feats.sum() * 0.0
    sim = feats @ feats.t() / temperature
    self_mask = torch.eye(n, dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(self_mask, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (labs[:, None] == labs[None, :]) & ~self_mask
    n_pos = pos.sum(dim=1)
    valid = n_pos > 0
    if not valid.any():
        return feats.sum() * 0.0
    mean_log_prob_pos = (log_prob.masked_fill(~pos, 0.0)).sum(dim=1)[valid] / n_pos[valid]
    return -mean_log_prob_pos.mean()


class MultiLoss(nn.Module):
    """Weighted sum of the paper's four losses plus an optional VAD regression term."""

    def __init__(self, num_classes: int, center_dim: int, weights: dict, focal_gamma: float = 2.0,
                 supcon_temperature: float = 0.1, supcon_max_frames: int = 32):
        super().__init__()
        self.weights = dict(weights)
        self.focal_gamma = focal_gamma
        self.temperature = supcon_temperature
        self.max_frames = supcon_max_frames
        self.center = CenterLoss(num_classes, center_dim)

    def forward(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        w = self.weights
        soft, hard = batch["soft"], batch["hard"]
        parts: dict[str, torch.Tensor] = {}
        if w.get("kl", 0) > 0:
            parts["kl"] = kl_loss(out["logits"], soft)
        if w.get("focal", 0) > 0:
            parts["focal"] = focal_loss(out["logits"], soft, self.focal_gamma)
        if w.get("center", 0) > 0:
            parts["center"] = self.center(out["z"], hard)
        if w.get("supcon", 0) > 0:
            labels = frame_labels(out["mask"], hard, batch["span"], batch["span_label"])
            parts["supcon"] = supcon_loss(out["h"], labels, out["mask"], self.temperature, self.max_frames)
        if w.get("vad", 0) > 0:
            vm = batch["vad_mask"]
            err = (out["vad"].float() - batch["vad"]).pow(2) * vm
            parts["vad"] = err.sum() / vm.sum().clamp(min=1.0)
        if not parts:
            raise ValueError("All loss weights are zero.")
        total = sum(w[name] * value for name, value in parts.items())
        logs = {f"loss_{name}": float(value.detach()) for name, value in parts.items()}
        logs["loss"] = float(total.detach())
        return total, logs

"""Optimiser parameter groups and learning-rate schedules."""
from __future__ import annotations

import math
import re

import torch

_LAYER_RE = re.compile(r"encoder\.model\.encoder\.layers\.(\d+)\.")


def _depth(name: str, num_layers: int) -> int:
    """0 = input side of the encoder ... ``num_layers`` = top of the encoder."""
    m = _LAYER_RE.search(name)
    if m:
        return int(m.group(1)) + 1
    if "encoder.model.encoder.layer_norm" in name:
        return num_layers
    return 0


def build_param_groups(model: torch.nn.Module, criterion: torch.nn.Module, cfg_train) -> list[dict]:
    """Encoder / head / center groups, optionally with layer-wise LR decay on the encoder."""
    lr_enc, lr_head, lr_center = cfg_train.lr_encoder, cfg_train.lr_head, cfg_train.lr_center
    wd = cfg_train.weight_decay
    decay = float(cfg_train.get("layerwise_lr_decay", 1.0))
    num_layers = model.encoder.num_encoder_layers()
    groups: dict[tuple, dict] = {}

    def add(key: tuple, param: torch.Tensor, lr: float, weight_decay: float, kind: str):
        if key not in groups:
            groups[key] = {"params": [], "lr": lr, "weight_decay": weight_decay, "kind": kind}
        groups[key]["params"].append(param)

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_wd = p.ndim <= 1
        if name.startswith("encoder.model."):
            scale = decay ** (num_layers - _depth(name, num_layers))
            add(("enc", round(scale, 6), no_wd), p, lr_enc * scale, 0.0 if no_wd else wd, "encoder")
        else:
            add(("head", no_wd), p, lr_head, 0.0 if no_wd else wd, "head")
    for p in criterion.parameters():
        if p.requires_grad:
            add(("center",), p, lr_center, 0.0, "center")
    return list(groups.values())


def lr_factor(step: int, steps_per_epoch: int, total_steps: int, sched) -> float:
    """Multiplier applied to every group's base LR at optimiser step ``step``."""
    kind = sched.get("type", "exp")
    epoch = step // max(steps_per_epoch, 1)
    if kind == "exp":      # base paper: lr decays by gamma each epoch until epoch `decay_epochs`
        factor = float(sched.get("gamma", 0.875)) ** min(epoch, int(sched.get("decay_epochs", 20)))
    elif kind == "cosine":
        progress = min(step / max(total_steps, 1), 1.0)
        factor = max(0.5 * (1.0 + math.cos(math.pi * progress)), float(sched.get("min_factor", 0.01)))
    elif kind == "constant":
        factor = 1.0
    else:
        raise ValueError(f"Unknown schedule type {kind!r}")
    warmup = float(sched.get("warmup_epochs", 0)) * steps_per_epoch
    if warmup > 0 and step < warmup:
        factor *= (step + 1) / warmup
    return factor

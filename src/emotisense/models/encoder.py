"""Self-supervised speech encoder (WavLM / HuBERT / wav2vec 2.0) with learnable layer weighting."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, WavLMConfig, WavLMModel

FRAME_STRIDE = 320  # samples per encoder frame at 16 kHz (20 ms) for all supported models


def tiny_wavlm_config(hidden: int = 64, layers: int = 3) -> WavLMConfig:
    """A few-hundred-kilobyte random WavLM used for tests and the offline smoke run."""
    return WavLMConfig(
        hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=4,
        intermediate_size=hidden * 2, conv_dim=(16,) * 7, feat_extract_norm="layer",
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4, num_buckets=32,
        max_bucket_distance=64, do_stable_layer_norm=True, apply_spec_augment=False,
        layerdrop=0.0, vocab_size=32,
    )


class SpeechEncoder(nn.Module):
    """Wraps a Hugging Face speech model and returns frame features plus a frame mask.

    With ``layer_weighting`` the output is a softmax-weighted sum over all hidden states
    (the SUPERB recipe), which usually beats using only the last layer for emotion.
    """

    def __init__(self, name: str = "microsoft/wavlm-large", pretrained: bool = True,
                 encoder_config: dict | None = None, tiny: bool = False,
                 freeze_feature_extractor: bool = True, layer_weighting: bool = True,
                 gradient_checkpointing: bool = False, layerdrop: float = 0.0,
                 spec_augment: bool = False, mask_time_prob: float = 0.05):
        super().__init__()
        if encoder_config is not None:       # rebuilding from a saved checkpoint: no network access
            cfg = AutoConfig.for_model(**encoder_config)
            self.model = AutoModel.from_config(cfg)
        elif tiny:
            self.model = WavLMModel(tiny_wavlm_config())
        else:
            overrides = dict(layerdrop=layerdrop, apply_spec_augment=spec_augment,
                             mask_time_prob=mask_time_prob)
            if pretrained:
                self.model = AutoModel.from_pretrained(name, **overrides)
            else:
                self.model = AutoModel.from_config(AutoConfig.from_pretrained(name, **overrides))
        self.config = self.model.config
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers + 1   # embeddings + each transformer layer
        self.layer_weighting = layer_weighting
        if layer_weighting:
            self.layer_weights = nn.Parameter(torch.zeros(self.num_layers))
        if freeze_feature_extractor:
            self.model.freeze_feature_encoder()
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self._use_mask = getattr(self.config, "feat_extract_norm", "group") == "layer"

    def frame_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        return self.model._get_feat_extract_output_lengths(lengths)

    def forward(self, wave: torch.Tensor, attn: torch.Tensor | None = None,
                lengths: torch.Tensor | None = None, uniform_layers: bool = False
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(frames [B, T, D], frame_mask [B, T] bool)``.

        ``uniform_layers`` averages all layers equally (used for the frozen-embedding baseline).
        """
        if lengths is None:
            lengths = attn.sum(dim=1) if attn is not None else torch.full(
                (wave.size(0),), wave.size(1), device=wave.device)
        out = self.model(wave, attention_mask=attn if self._use_mask else None,
                         output_hidden_states=True)
        if self.layer_weighting or uniform_layers:
            hidden = torch.stack(out.hidden_states, dim=0)                    # [L, B, T, D]
            if uniform_layers or not self.layer_weighting:
                weights = torch.full((hidden.size(0),), 1.0 / hidden.size(0), device=hidden.device)
            else:
                weights = torch.softmax(self.layer_weights, dim=0)
            frames = (weights.view(-1, 1, 1, 1) * hidden).sum(dim=0)
        else:
            frames = out.last_hidden_state
        t = frames.size(1)
        frame_len = self.frame_lengths(lengths.to(wave.device)).clamp(min=1, max=t)
        mask = torch.arange(t, device=wave.device)[None, :] < frame_len[:, None]
        return frames, mask

    def num_encoder_layers(self) -> int:
        return self.config.num_hidden_layers

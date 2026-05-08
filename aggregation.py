"""
aggregation.py - Token aggregation and feature extraction.

Strategy:
  - Concatenate the last real-token hidden state from three mid-to-late
    transformer layers (the hallucination signal is well documented to
    live in mid-to-late representations).
  - Optional geometric features: per-layer L2 norms, inter-layer cosine
    drift, and log sequence length - cheap uncertainty proxies.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


SELECTED_LAYERS_FRAC = (0.5, 0.75, 1.0)


def _selected_layer_indices(n_layers: int) -> list[int]:
    return [max(1, int(round(f * (n_layers - 1)))) for f in SELECTED_LAYERS_FRAC]


def _last_real_index(attention_mask: torch.Tensor) -> int:
    real = attention_mask.nonzero(as_tuple=False)
    return int(real[-1].item())


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Concat last-token vectors from 3 selected layers."""
    n_layers = hidden_states.size(0)
    last_pos = _last_real_index(attention_mask)
    layer_idx = _selected_layer_indices(n_layers)

    parts = [hidden_states[i, last_pos] for i in layer_idx]
    return torch.cat(parts, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-layer norms + inter-layer cosine drift + log seq length."""
    n_layers, _, _ = hidden_states.shape
    last_pos = _last_real_index(attention_mask)

    last_token = hidden_states[:, last_pos, :]                         # (n_layers, hidden)
    norms = last_token.norm(dim=-1)                                    # (n_layers,)

    cos = F.cosine_similarity(last_token[:-1], last_token[1:], dim=-1) # (n_layers - 1,)

    seq_len = float(attention_mask.sum().item())
    log_len = torch.tensor(
        [math.log1p(seq_len)],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    return torch.cat([norms, cos, log_len], dim=0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    agg = aggregate(hidden_states, attention_mask)
    if use_geometric:
        geo = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg, geo], dim=0)
    return agg

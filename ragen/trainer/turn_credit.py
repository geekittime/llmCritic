"""Credit transforms for exact one-turn rollout rows.

The label-only ablation treats a judge score as an advantage directly.  For
experiments that instead interpret judge scores as transition rewards, this
module provides a return-to-go transform without changing the exact turn-PPO
probability calculation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


def _encode_episode_ids(episode_ids: Sequence[Any]) -> list[int]:
    codes: dict[Any, int] = {}
    encoded: list[int] = []
    for episode_id in np.asarray(episode_ids, dtype=object).reshape(-1).tolist():
        try:
            key = episode_id.item() if hasattr(episode_id, "item") else episode_id
            hash(key)
        except (TypeError, ValueError):
            key = repr(episode_id)
        if key not in codes:
            codes[key] = len(codes)
        encoded.append(codes[key])
    return encoded


def discounted_turn_returns(
    turn_scores: torch.Tensor,
    *,
    turn_ids: torch.Tensor,
    response_mask: torch.Tensor,
    episode_ids: Sequence[Any],
    trajectory_turn_ids: Sequence[int],
    gamma: float = 1.0,
) -> torch.Tensor:
    """Convert per-turn transition scores to episode return-to-go values.

    Exact turn traces expand each environment action into one batch row.  Rows
    may be reordered for sequence balancing or duplicated for FSDP divisibility,
    so grouping uses ``(episode_id, trajectory_turn_id)`` rather than row order.
    Duplicate padding rows are collapsed before the return is computed.
    """

    if turn_scores.shape != turn_ids.shape or response_mask.shape != turn_ids.shape:
        raise ValueError("turn_scores, turn_ids, and response_mask must have identical shapes")
    if turn_scores.ndim != 2:
        raise ValueError("turn credit tensors must be rank-2")
    if not np.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be finite and non-negative")

    batch_size = turn_scores.shape[0]
    episode_codes = _encode_episode_ids(episode_ids)
    trajectory_values = np.asarray(trajectory_turn_ids).reshape(-1)
    if len(episode_codes) != batch_size or trajectory_values.size != batch_size:
        raise ValueError("episode_ids and trajectory_turn_ids must match the batch size")

    valid = (turn_ids >= 0) & (response_mask > 0)
    if not torch.all(valid.any(dim=-1)):
        raise ValueError("every exact turn row must contain sampled action tokens")
    masked_ids = torch.where(valid, turn_ids, torch.full_like(turn_ids, -1))
    row_max_ids = masked_ids.max(dim=-1).values
    row_min_ids = torch.where(valid, turn_ids, row_max_ids.unsqueeze(-1)).min(dim=-1).values
    if not torch.equal(row_min_ids, row_max_ids):
        raise ValueError("discounted_turn_returns requires exactly one macro action per row")

    positions = torch.arange(turn_scores.shape[1], device=turn_scores.device).unsqueeze(0)
    endpoints = torch.where(valid, positions, -1).max(dim=-1).values
    row_scores = turn_scores.gather(1, endpoints.unsqueeze(-1)).squeeze(-1).float()

    pair_tensor = torch.stack(
        (
            torch.as_tensor(episode_codes, dtype=torch.long, device=turn_scores.device),
            torch.as_tensor(trajectory_values, dtype=torch.long, device=turn_scores.device),
        ),
        dim=-1,
    )
    unique_pairs, inverse = torch.unique(pair_tensor, sorted=True, return_inverse=True, dim=0)
    pair_sums = row_scores.new_zeros(unique_pairs.shape[0])
    pair_counts = row_scores.new_zeros(unique_pairs.shape[0])
    pair_sums.scatter_add_(0, inverse, row_scores)
    pair_counts.scatter_add_(0, inverse, torch.ones_like(row_scores))
    pair_scores = pair_sums / pair_counts.clamp_min(1.0)
    pair_min = torch.full_like(pair_scores, torch.inf)
    pair_max = torch.full_like(pair_scores, -torch.inf)
    pair_min.scatter_reduce_(0, inverse, row_scores, reduce="amin", include_self=True)
    pair_max.scatter_reduce_(0, inverse, row_scores, reduce="amax", include_self=True)
    if torch.any((pair_max - pair_min).abs() > 1e-6):
        raise ValueError("duplicate rows for one episode turn carry inconsistent judge scores")

    pair_returns = torch.zeros_like(pair_scores)
    for episode_code in torch.unique(unique_pairs[:, 0], sorted=True).tolist():
        pair_indexes = torch.nonzero(unique_pairs[:, 0] == episode_code, as_tuple=True)[0]
        order = torch.argsort(unique_pairs[pair_indexes, 1])
        ordered_indexes = pair_indexes[order]
        running = pair_scores.new_zeros(())
        for pair_index in reversed(ordered_indexes.tolist()):
            running = pair_scores[pair_index] + float(gamma) * running
            pair_returns[pair_index] = running

    row_returns = pair_returns[inverse]
    return torch.where(
        valid,
        row_returns.unsqueeze(-1),
        torch.zeros_like(row_returns.unsqueeze(-1)),
    ).to(dtype=turn_scores.dtype)


def assign_turn_credit(
    turn_scores: torch.Tensor,
    *,
    mode: str,
    turn_ids: torch.Tensor,
    response_mask: torch.Tensor,
    episode_ids: Sequence[Any],
    trajectory_turn_ids: Sequence[int],
    gamma: float = 1.0,
) -> torch.Tensor:
    """Apply the configured judge-score interpretation."""

    if mode == "direct":
        valid = (turn_ids >= 0) & (response_mask > 0)
        return torch.where(valid, turn_scores, torch.zeros_like(turn_scores))
    if mode == "discounted_return":
        return discounted_turn_returns(
            turn_scores,
            turn_ids=turn_ids,
            response_mask=response_mask,
            episode_ids=episode_ids,
            trajectory_turn_ids=trajectory_turn_ids,
            gamma=gamma,
        )
    raise ValueError(
        f"Unsupported algorithm.turn_credit_assignment={mode!r}; "
        "expected 'direct' or 'discounted_return'"
    )

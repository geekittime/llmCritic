import numpy as np
import pytest
import torch

from ragen.trainer.turn_credit import assign_turn_credit, discounted_turn_returns


def _expanded(values):
    return torch.tensor([[value, value, 0.0] for value in values], dtype=torch.float32)


def test_discounted_returns_group_reordered_and_duplicated_turn_rows():
    scores = _expanded([2.0, 1.0, 2.0, -1.0, 1.0])
    turn_ids = torch.tensor([[0, 0, -1]] * 5)
    response_mask = (turn_ids >= 0).float()

    result = discounted_turn_returns(
        scores,
        turn_ids=turn_ids,
        response_mask=response_mask,
        episode_ids=np.array([10, 10, 10, 20, 20]),
        trajectory_turn_ids=np.array([1, 0, 1, 0, 1]),
        gamma=0.5,
    )

    assert torch.equal(result, _expanded([2.0, 2.0, 2.0, -0.5, 1.0]))


def test_direct_credit_preserves_exact_judge_scores():
    scores = _expanded([-1.0, 0.0, 1.0])
    turn_ids = torch.tensor([[0, 0, -1]] * 3)
    response_mask = (turn_ids >= 0).float()

    result = assign_turn_credit(
        scores,
        mode="direct",
        turn_ids=turn_ids,
        response_mask=response_mask,
        episode_ids=np.arange(3),
        trajectory_turn_ids=np.zeros(3),
    )

    assert torch.equal(result, scores)


def test_discounted_returns_reject_multi_action_rows():
    scores = _expanded([1.0])
    turn_ids = torch.tensor([[0, 1, -1]])

    with pytest.raises(ValueError, match="one macro action"):
        discounted_turn_returns(
            scores,
            turn_ids=turn_ids,
            response_mask=(turn_ids >= 0).float(),
            episode_ids=[0],
            trajectory_turn_ids=[0],
        )


def test_discounted_returns_reject_inconsistent_padding_copies():
    scores = _expanded([1.0, -1.0])
    turn_ids = torch.tensor([[0, 0, -1]] * 2)

    with pytest.raises(ValueError, match="inconsistent judge scores"):
        discounted_turn_returns(
            scores,
            turn_ids=turn_ids,
            response_mask=(turn_ids >= 0).float(),
            episode_ids=[0, 0],
            trajectory_turn_ids=[0, 0],
        )


def test_turn_credit_rejects_unknown_mode():
    scores = _expanded([1.0])
    turn_ids = torch.tensor([[0, 0, -1]])

    with pytest.raises(ValueError, match="turn_credit_assignment"):
        assign_turn_credit(
            scores,
            mode="unknown",
            turn_ids=turn_ids,
            response_mask=(turn_ids >= 0).float(),
            episode_ids=[0],
            trajectory_turn_ids=[0],
        )

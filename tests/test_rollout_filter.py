import numpy as np
import pytest
import torch
from tensordict import TensorDict

try:
    from verl import DataProto
except ImportError:  # pragma: no cover - the project test env provides verl
    pytest.skip("verl is required for rollout-filter tests", allow_module_level=True)


from ragen.trainer.rollout_filter import (
    RolloutFilterConfig,
    RewardRolloutFilter,
    EntropyRolloutFilter,
)


def _make_reward_batch(num_groups: int, group_size: int, traj_len: int):
    total = num_groups * group_size
    rm_scores = torch.arange(total * traj_len, dtype=torch.float32).reshape(total, traj_len)
    loss_mask = torch.ones(total, traj_len)
    batch = TensorDict(
        {
            "original_rm_scores": rm_scores,
            "loss_mask": loss_mask,
        },
        batch_size=[total],
    )
    non_tensor_batch = {"uids": np.arange(total)}
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={})


def test_reward_variance_filter_reduces_batch_size():
    num_groups, group_size, traj_len = 4, 2, 3
    batch = _make_reward_batch(num_groups, group_size, traj_len)

    rollout_filter = RewardRolloutFilter(
        RolloutFilterConfig(
            ratio=0.5,
            filter_type="largest",
            num_groups=num_groups,
            group_size=group_size,
        )
    )

    filtered_batch, metrics = rollout_filter.filter(batch)

    assert filtered_batch.batch["original_rm_scores"].shape[0] == group_size * max(int(0.5 * num_groups), 1)
    assert "rollout/in_group_std" in metrics


def test_entropy_variance_filter_uses_compute_log_prob():
    num_groups, group_size, traj_len = 2, 3, 4
    batch = _make_reward_batch(num_groups, group_size, traj_len)

    entropies = torch.linspace(0.1, 1.0, steps=num_groups * group_size * traj_len).reshape(num_groups * group_size, traj_len)
    old_log_probs = -entropies

    def fake_compute_log_prob(data_proto):
        td = TensorDict(
            {
                "old_log_probs": old_log_probs,
                "entropys": entropies,
            },
            batch_size=[num_groups * group_size],
        )
        return DataProto(batch=td, non_tensor_batch={}, meta_info={})

    rollout_filter = EntropyRolloutFilter(
        RolloutFilterConfig(
            ratio=0.5,
            filter_type="largest",
            num_groups=num_groups,
            group_size=group_size,
            metric="entropy",
        ),
        compute_log_prob=fake_compute_log_prob,
    )

    filtered_batch, metrics = rollout_filter.filter(batch)

    expected = group_size * max(int(0.5 * num_groups), 1)
    assert filtered_batch.batch["loss_mask"].shape[0] == expected
    assert "old_log_probs" in filtered_batch.batch.keys()
    assert "rollout/in_group_entropy_std" in metrics


def test_reward_metric_selects_high_mean_group():
    num_groups, group_size, traj_len = 2, 2, 1
    batch = _make_reward_batch(num_groups, group_size, traj_len)

    # Overwrite scores: first group has higher mean, second has higher variance.
    batch.batch["original_rm_scores"] = torch.tensor(
        [
            [10.0],
            [11.0],
            [0.0],
            [5.0],
        ]
    )

    rollout_filter = RewardRolloutFilter(
        RolloutFilterConfig(
            ratio=0.5,
            filter_type="largest",
            num_groups=num_groups,
            group_size=group_size,
            metric="reward",
        )
    )

    filtered_batch, _ = rollout_filter.filter(batch)

    # Highest mean group is the first one, so we expect its entries to remain.
    retained = filtered_batch.batch["original_rm_scores"].squeeze(-1)
    assert torch.allclose(retained, torch.tensor([10.0, 11.0]))


def test_turn_filter_aggregates_all_rows_of_each_episode():
    # Exact turn traces contain one row per turn, and episodes can have
    # different numbers of turns.  The terminal score must be aggregated by
    # episode before comparing groups; taking the first row would select the
    # wrong group here.
    scores = torch.tensor([[0.0], [1.0], [0.0], [2.0], [0.0], [1.0]])
    batch = TensorDict(
        {
            "original_rm_scores": scores,
            "loss_mask": torch.ones_like(scores),
        },
        batch_size=[6],
    )
    proto = DataProto(
        batch=batch,
        non_tensor_batch={
            "episode_ids": np.array([0, 0, 1, 2, 2, 3]),
            "group_ids": np.array([0, 0, 0, 1, 1, 1]),
            "env_ids": np.arange(6),
        },
        meta_info={},
    )
    rollout_filter = RewardRolloutFilter(
        RolloutFilterConfig(
            ratio=0.5,
            filter_type="largest",
            num_groups=2,
            group_size=2,
            metric="reward",
        )
    )

    filtered, _ = rollout_filter.filter(proto)

    assert filtered.non_tensor_batch["env_ids"].tolist() == [3, 4, 5]

import numpy as np
import torch
from tensordict import TensorDict
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.metric_utils import process_validation_metrics

from ragen.trainer.agent_trainer import (
    _aggregate_episode_scores,
    _episode_row_groups,
    _messages_to_prompt_text,
    _validation_sample_uids,
    adjust_batch,
    training_rollout_seed,
    validation_rollout_seed,
)


def _batch(size: int) -> DataProto:
    return DataProto(
        batch=TensorDict(
            {"row_id": torch.arange(size, dtype=torch.long)},
            batch_size=[size],
        ),
        non_tensor_batch={"name": np.asarray([f"row-{index}" for index in range(size)])},
        meta_info={},
    )


def test_adjust_batch_copy_preserves_each_source_rows_total_weight():
    adjusted = adjust_batch(_batch(3), size_divisor=8, mode="copy")

    assert adjusted.batch.batch_size[0] == 8
    assert torch.allclose(adjusted.batch["sample_weights"].sum(), torch.tensor(3.0))
    for row_id in range(3):
        mask = adjusted.batch["row_id"] == row_id
        assert torch.allclose(adjusted.batch["sample_weights"][mask].sum(), torch.tensor(1.0))


def test_adjust_batch_delete_keeps_tensor_and_non_tensor_rows_aligned():
    np.random.seed(7)
    adjusted = adjust_batch(_batch(5), size_divisor=4, mode="delete")

    assert adjusted.batch.batch_size[0] == 4
    for row_id, name in zip(
        adjusted.batch["row_id"].tolist(),
        adjusted.non_tensor_batch["name"].tolist(),
        strict=True,
    ):
        assert name == f"row-{row_id}"


def test_validation_rollout_seeds_are_fixed_non_overlapping_blocks():
    config = OmegaConf.create(
        {
            "seed": {"val": 123},
            "es_manager": {"val": {"env_groups": 16}},
        }
    )

    first_checkpoint = [validation_rollout_seed(config, step) for step in range(3)]
    second_checkpoint = [validation_rollout_seed(config, step) for step in range(3)]

    assert first_checkpoint == [123, 139, 155]
    assert second_checkpoint == first_checkpoint


def test_training_rollout_seed_is_resume_stable():
    config = OmegaConf.create(
        {
            "seed": {"train": 10000},
            "es_manager": {"train": {"env_groups": 16}},
        }
    )

    assert training_rollout_seed(config, 1) == 10000
    assert training_rollout_seed(config, 51) == 10800
    assert training_rollout_seed(config, 51) == training_rollout_seed(config, 51)


def test_episode_row_groups_selects_final_turn_for_exact_traces():
    grouped = _episode_row_groups(
        {
            "episode_ids": np.array([10, 10, 11, 11, 11]),
            "trajectory_turn_ids": np.array([0, 1, 0, 1, 2]),
        },
        batch_size=5,
    )

    assert grouped == (
        [10, 11],
        {10: [0, 1], 11: [2, 3, 4]},
        [1, 4],
    )


def test_episode_row_groups_uses_last_repeated_episode_reward_without_turn_ids():
    grouped = _episode_row_groups(
        {"episode_ids": np.array([4, 4, 7])},
        batch_size=3,
    )

    assert grouped == ([4, 7], {4: [0, 1], 7: [2]}, [1, 2])


def test_validation_prompt_grouping_excludes_sampled_action_and_followup_state():
    shared_prefix = [
        {"role": "system", "content": "solve"},
        {"role": "user", "content": "initial board"},
    ]
    left = shared_prefix + [
        {"role": "assistant", "content": "Left"},
        {"role": "user", "content": "left board"},
    ]
    right = shared_prefix + [
        {"role": "assistant", "content": "Right"},
        {"role": "user", "content": "right board"},
    ]

    assert _messages_to_prompt_text(left) == _messages_to_prompt_text(right)
    assert "Left" not in _messages_to_prompt_text(left)


def test_validation_uids_do_not_merge_distinct_groups_with_identical_prompts():
    non_tensor_batch = {"group_ids": np.array([3, 3, 9, 9])}
    uids = _validation_sample_uids(
        2,
        non_tensor_batch,
        ["CoordSokoban"] * 4,
        [0, 1, 2, 3],
    )

    assert uids[0] == uids[1]
    assert uids[2] == uids[3]
    assert uids[0] != uids[2]
    assert all("validation_step=2" in uid for uid in uids)

    metrics = process_validation_metrics(
        np.array(["CoordSokoban"] * 4),
        uids,
        {"acc": [1.0, 0.0, 0.0, 0.0]},
    )
    assert "mean@2" in metrics["CoordSokoban"]["acc"]
    assert "mean@4" not in metrics["CoordSokoban"]["acc"]


def test_episode_score_aggregation_supports_terminal_and_turn_reward_modes():
    groups = (
        [10, 11],
        {10: [0, 1], 11: [2]},
        [1, 2],
    )
    scores = [0.25, -0.1, 1.0]

    assert _aggregate_episode_scores(scores, groups, sum_turn_scores=False) == [-0.1, 1.0]
    assert _aggregate_episode_scores(scores, groups, sum_turn_scores=True) == [0.15, 1.0]

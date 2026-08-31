import numpy as np
import torch
from tensordict import TensorDict
from omegaconf import OmegaConf

from verl import DataProto

from ragen.trainer.agent_trainer import (
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

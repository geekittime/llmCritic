import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto

from ragen.trainer.agent_trainer import adjust_batch


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
    assert adjusted.batch["sample_weights"].sum().item() == 3.0
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

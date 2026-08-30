import inspect
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss

from ragen.llm_agent.ctx_manager import build_legacy_turn_metadata, build_turn_token_metadata
from ragen.trainer.agent_trainer import (
    broadcast_outcome_to_turns,
    collapse_turn_scores,
    compose_turn_advantages,
    normalize_turn_scores,
    place_turn_endpoint_rewards,
    trajectory_outcomes,
)
from ragen.trainer.core_algos import compute_turn_gae_advantage_return
from ragen.workers.actor.dp_actor import (
    DataParallelPPOActor,
    compute_turn_micro_batch_scale,
    compute_turn_policy_loss,
)
from ragen.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
from ragen.utils import redact_config


def test_tracker_config_redacts_nested_credentials_without_mutating_input():
    config = {
        "generative_critic": {
            "deepseek_api_key": "sk-not-for-logs",
            "deepseek_api_key_env": "DEEPSEEK_API_KEY",
            "deepseek_api_key_file": "/protected/secrets.env",
            "max_concurrency": 8,
        },
        "wandb_token": "wb-not-for-logs",
    }

    redacted = redact_config(config)

    assert redacted["generative_critic"]["deepseek_api_key"] == "<redacted>"
    assert redacted["generative_critic"]["deepseek_api_key_env"] == "<redacted>"
    assert redacted["generative_critic"]["deepseek_api_key_file"] == "<redacted>"
    assert redacted["wandb_token"] == "<redacted>"
    assert redacted["generative_critic"]["max_concurrency"] == 8
    assert config["generative_critic"]["deepseek_api_key"] == "sk-not-for-logs"


def test_token_trace_metadata_is_causal_aligned_and_left_padding_safe():
    # Two exact prompt/response traces, with the first one left padded.
    input_ids = torch.tensor(
        [[0, 0, 10, 11, 12, 20, 21], [30, 31, 40, 41, 42, 43, 44]],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]],
        dtype=torch.long,
    )
    turn_ids, value_mask, value_ids, end_mask = build_turn_token_metadata(
        input_ids, attention_mask, prompt_lengths=[3, 2], response_lengths=[2, 5]
    )

    assert turn_ids.tolist() == [[-1, -1, -1, -1, 0, 0], [-1, 0, 0, 0, 0, 0]]
    assert torch.nonzero(value_mask[0]).flatten().tolist() == [4]
    assert torch.nonzero(end_mask[0]).flatten().tolist() == [5]
    assert value_ids[0, 4].item() == 0


def test_legacy_metadata_excludes_assistant_role_header():
    class TinyTokenizer:
        name_or_path = "qwen-tiny"

        def encode(self, text, add_special_tokens=True):
            return {"<|im_start|>": [1], "<|im_end|>": [2]}[text]

        def decode(self, ids, skip_special_tokens=False):
            pieces = {1: "<|im_start|>", 2: "<|im_end|>", 3: "system", 4: "user", 5: "assistant", 6: "\n", 7: "state", 8: "act"}
            return "".join(pieces[int(token)] for token in ids)

    # system/user/assistant chat template; source index 9 predicts the first
    # sampled action token, while the role header at indices 7-8 is excluded.
    ids = torch.tensor([[1, 3, 6, 2, 1, 4, 6, 7, 2, 1, 5, 6, 8, 2]])
    mask = torch.ones_like(ids)
    turn_ids, value_mask, value_ids, end_mask = build_legacy_turn_metadata(ids, mask, TinyTokenizer())
    valid = torch.nonzero(turn_ids[0] >= 0, as_tuple=True)[0].tolist()
    assert valid == [11, 12]
    assert torch.nonzero(value_mask[0]).flatten().tolist() == [11]
    assert value_ids[0, 11].item() == 0
    assert torch.nonzero(end_mask[0]).flatten().tolist() == [12]


def test_turn_gae_uses_one_endpoint_scalar_without_length_multiplication():
    rewards = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    values = torch.tensor([[0.5, 0.0, 0.0], [0.2, 0.0, 0.0]])
    response_mask = torch.ones_like(rewards)
    turn_ids = torch.tensor([[0, 0, 0], [0, 0, -1]])
    value_mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    value_ids = torch.tensor([[0, -1, -1], [0, -1, -1]])

    advantages, returns = compute_turn_gae_advantage_return(
        rewards,
        values,
        response_mask,
        turn_ids,
        value_mask,
        value_ids,
        gamma=1.0,
        lam=1.0,
        normalize_advantages=False,
    )

    assert torch.allclose(advantages[0], torch.full((3,), 0.5))
    assert torch.allclose(advantages[1, :2], torch.full((2,), -1.2))
    assert returns[0, 0].item() == 1.0
    assert returns[1, 0].item() == -1.0
    assert torch.count_nonzero(returns[0, 1:]) == 0


def test_macro_ratio_sums_token_log_probabilities():
    old_log_prob = torch.zeros((1, 5))
    log_prob = torch.tensor([[0.1, 0.2, 9.0, -0.1, 0.3]])
    advantages = torch.tensor([[0.0, 2.0, 0.0, 0.0, 3.0]])
    response_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0]])
    turn_ids = torch.tensor([[0, 0, -1, 1, 1]])

    loss, *_ = compute_turn_policy_loss(
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        turn_ids,
        cliprange=10.0,
        cliprange_low=10.0,
        cliprange_high=10.0,
        clip_ratio_c=3.0,
    )
    expected = -(2 * torch.exp(torch.tensor(0.3)) + 3 * torch.exp(torch.tensor(0.2))) / 2
    assert torch.allclose(loss, expected)


def _reference_turn_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    turn_ids,
    sample_weights=None,
):
    packed_old_log_prob = []
    packed_log_prob = []
    packed_advantages = []
    packed_weights = []
    if sample_weights is None:
        sample_weights = torch.ones(turn_ids.size(0), dtype=log_prob.dtype)

    for row in range(turn_ids.size(0)):
        for turn_id in torch.unique(turn_ids[row][turn_ids[row] >= 0], sorted=True):
            token_mask = (turn_ids[row] == turn_id) & (response_mask[row] > 0)
            if not torch.any(token_mask):
                continue
            token_indices = torch.nonzero(token_mask, as_tuple=True)[0]
            packed_old_log_prob.append(old_log_prob[row][token_mask].sum())
            packed_log_prob.append(log_prob[row][token_mask].sum())
            packed_advantages.append(advantages[row, token_indices[-1]])
            packed_weights.append(sample_weights[row])

    if not packed_log_prob:
        zero = log_prob.new_tensor(0.0)
        return zero, zero, zero, zero

    return compute_policy_loss(
        old_log_prob=torch.stack(packed_old_log_prob).unsqueeze(-1),
        log_prob=torch.stack(packed_log_prob).unsqueeze(-1),
        advantages=torch.stack(packed_advantages).unsqueeze(-1),
        response_mask=torch.stack(packed_weights).unsqueeze(-1),
        cliprange=0.2,
        cliprange_low=0.1,
        cliprange_high=0.3,
        clip_ratio_c=3.0,
        loss_agg_mode="token-mean",
    )


def test_vectorized_single_turn_path_handles_padding_and_empty_rows():
    old_log_prob = torch.zeros((3, 4))
    log_prob = torch.tensor(
        [
            [0.1, 99.0, 0.2, 99.0],
            [99.0, 99.0, 99.0, 99.0],
            [0.1, 99.0, 0.3, 99.0],
        ]
    )
    advantages = torch.tensor(
        [
            [10.0, 10.0, 2.0, 10.0],
            [10.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 4.0, 10.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    turn_ids = torch.tensor(
        [
            [0, -1, 0, -1],
            [-1, -1, -1, -1],
            [0, 0, 0, -1],
        ]
    )

    actual = compute_turn_policy_loss(
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        turn_ids,
        cliprange=0.2,
        cliprange_low=0.1,
        cliprange_high=0.3,
        clip_ratio_c=3.0,
    )
    expected = _reference_turn_policy_loss(
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        turn_ids,
    )

    assert all(torch.allclose(got, want) for got, want in zip(actual, expected))
    assert DataParallelPPOActor._count_turns(turn_ids, response_mask=response_mask) == 2.0


def test_turn_policy_loss_returns_zero_for_all_padding_rows():
    shape = (2, 3)
    outputs = compute_turn_policy_loss(
        torch.full(shape, float("nan")),
        torch.full(shape, float("nan")),
        torch.full(shape, float("nan")),
        torch.zeros(shape),
        torch.full(shape, -1),
        cliprange=0.2,
        cliprange_low=0.1,
        cliprange_high=0.3,
        clip_ratio_c=3.0,
    )

    assert all(output.item() == 0.0 for output in outputs)
    assert DataParallelPPOActor._count_turns(torch.full(shape, -1)) == 0.0


def test_vectorized_multi_turn_path_matches_reference_with_row_weights():
    old_log_prob = torch.tensor(
        [
            [0.0, -0.1, 9.0, 0.2, 0.0, -0.2, 9.0],
            [9.0, 0.1, -0.4, 0.0, 9.0, 0.3, -0.1],
            [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
        ]
    )
    log_prob = old_log_prob + torch.tensor(
        [
            [0.05, 0.02, 8.0, -0.03, 0.01, 0.04, 8.0],
            [8.0, -0.02, 0.05, 0.03, 8.0, -0.01, 0.02],
            [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
        ]
    )
    advantages = torch.tensor(
        [
            [1.0, 2.0, 0.0, -1.0, -2.0, 3.0, 0.0],
            [0.0, 4.0, 99.0, -3.0, 0.0, -2.0, -5.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    # Turn 2 reappears after turn 7, exercising grouping by id rather than
    # assuming each id occupies exactly one contiguous run.
    turn_ids = torch.tensor(
        [
            [2, 2, -1, 7, 7, 2, -1],
            [-1, 4, 4, 9, -1, 9, 9],
            [-1, -1, -1, -1, -1, -1, -1],
        ]
    )
    sample_weights = torch.tensor([0.25, 2.0, 10.0])

    actual = compute_turn_policy_loss(
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        turn_ids,
        cliprange=0.2,
        cliprange_low=0.1,
        cliprange_high=0.3,
        clip_ratio_c=3.0,
        sample_weights=sample_weights,
    )
    expected = _reference_turn_policy_loss(
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        turn_ids,
        sample_weights=sample_weights,
    )

    assert all(torch.allclose(got, want) for got, want in zip(actual, expected))
    assert DataParallelPPOActor._count_turns(
        turn_ids,
        sample_weights=sample_weights,
        response_mask=response_mask,
    ) == 4.5


def test_inverse_multiplicity_weights_remove_copied_row_bias():
    original_log_prob = torch.tensor([[0.1, 0.2], [-0.2, 0.1]])
    original_advantages = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
    original_turn_ids = torch.tensor([[0, 0], [0, 0]])
    original_mask = torch.ones_like(original_log_prob)

    original = compute_turn_policy_loss(
        torch.zeros_like(original_log_prob),
        original_log_prob,
        original_advantages,
        original_mask,
        original_turn_ids,
        cliprange=10.0,
        cliprange_low=10.0,
        cliprange_high=10.0,
        clip_ratio_c=3.0,
    )[0]

    copied_rows = torch.tensor([0, 0, 1])
    copied = compute_turn_policy_loss(
        torch.zeros_like(original_log_prob[copied_rows]),
        original_log_prob[copied_rows],
        original_advantages[copied_rows],
        original_mask[copied_rows],
        original_turn_ids[copied_rows],
        cliprange=10.0,
        cliprange_low=10.0,
        cliprange_high=10.0,
        clip_ratio_c=3.0,
        sample_weights=torch.tensor([0.5, 0.5, 1.0]),
    )[0]

    assert torch.allclose(copied, original)


def test_inverse_multiplicity_weights_survive_multiple_optimizer_steps():
    old_log_prob = torch.zeros((3, 1))
    log_prob = torch.tensor([[0.10], [-0.15], [0.20]])
    advantages = torch.tensor([[1.0], [-0.5], [2.0]])
    response_mask = torch.ones((3, 1))
    turn_ids = torch.zeros((3, 1), dtype=torch.long)

    full_loss = compute_turn_policy_loss(
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        turn_ids,
        cliprange=10.0,
        cliprange_low=10.0,
        cliprange_high=10.0,
        clip_ratio_c=3.0,
    )[0]

    copied_indices = torch.tensor([0, 1, 2, 2])
    copied_weights = torch.tensor([1.0, 1.0, 0.5, 0.5])
    accumulated_loss = torch.zeros_like(full_loss)
    for start in (0, 2):
        indices = copied_indices[start : start + 2]
        weights = copied_weights[start : start + 2]
        mini_loss = compute_turn_policy_loss(
            old_log_prob[indices],
            log_prob[indices],
            advantages[indices],
            response_mask[indices],
            turn_ids[indices],
            cliprange=10.0,
            cliprange_low=10.0,
            cliprange_high=10.0,
            clip_ratio_c=3.0,
            sample_weights=weights,
        )[0]
        scale = compute_turn_micro_batch_scale(
            micro_turn_weight=float(weights.sum()),
            global_turn_weight=3.0,
            num_mini_batches=2,
            data_parallel_world_size=1,
        )
        accumulated_loss = accumulated_loss + mini_loss * scale

    # Two optimizer steps together carry two copies of the global mean. The
    # duplicated third row must not receive a full extra step of its own.
    assert torch.allclose(accumulated_loss, 2.0 * full_loss)


def test_zero_entropy_coefficient_uses_placeholder_without_entropy_reduction(monkeypatch):
    actor = object.__new__(DataParallelPPOActor)
    actor.config = SimpleNamespace(entropy_coeff=0.0)
    actor.ulysses_sequence_parallel_size = 1
    actor.actor_module = SimpleNamespace(
        _fsdp_wrapped_module=object(),
        eval=lambda: None,
    )
    entropy_requests = []
    moved_micro_batches = []

    original_to = TensorDict.to

    def tracking_to(self, *args, **kwargs):
        moved_micro_batches.append((args, kwargs))
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(TensorDict, "to", tracking_to)
    monkeypatch.setattr("ragen.workers.actor.dp_actor.get_device_id", lambda: torch.device("cpu"))

    def fake_forward(self, micro_batch, temperature, calculate_entropy=False):
        entropy_requests.append(calculate_entropy)
        log_probs = micro_batch["responses"].float() / temperature
        return None, log_probs

    actor._forward_micro_batch = MethodType(fake_forward, actor)
    responses = torch.tensor([[1, 2, 3], [4, 5, 6]])
    data = DataProto.from_dict(
        tensors={
            "responses": responses,
            "input_ids": responses,
            "attention_mask": torch.ones_like(responses),
            "position_ids": torch.arange(3).expand_as(responses),
        },
        meta_info={
            "micro_batch_size": 1,
            "temperature": 2.0,
            "use_dynamic_bsz": False,
        },
    )

    log_probs, entropys = actor.compute_log_prob(data, calculate_entropy=True)

    assert entropy_requests == [False, False]
    assert len(moved_micro_batches) == 2
    assert torch.equal(log_probs, responses.float() / 2.0)
    assert torch.equal(entropys, torch.zeros_like(log_probs))


def test_turn_reward_helpers_are_scalar_and_endpoint_based():
    turn_ids = torch.tensor([[0, 0, -1, 1, 1]])
    response_mask = (turn_ids >= 0).float()
    token_scores = torch.tensor([[1.0, 3.0, 99.0, -2.0, -4.0]])
    turn_scores = collapse_turn_scores(token_scores, turn_ids, response_mask)
    assert torch.allclose(turn_scores, torch.tensor([[2.0, 2.0, 0.0, -3.0, -3.0]]))
    summed_scores = collapse_turn_scores(token_scores, turn_ids, response_mask, reduction="sum")
    assert torch.allclose(summed_scores, torch.tensor([[4.0, 4.0, 0.0, -6.0, -6.0]]))

    end_mask = torch.tensor([[0.0, 1.0, 0.0, 0.0, 1.0]])
    endpoint_scores = place_turn_endpoint_rewards(
        turn_scores, turn_ids, response_mask, turn_end_mask=end_mask
    )
    assert torch.allclose(endpoint_scores, torch.tensor([[0.0, 2.0, 0.0, 0.0, -3.0]]))

    all_turns = broadcast_outcome_to_turns(
        torch.tensor([1.0]), turn_ids, response_mask, mode="all_turns"
    )
    last_turn = broadcast_outcome_to_turns(
        torch.tensor([-1.0]), turn_ids, response_mask, mode="last_turn"
    )
    no_outcome = broadcast_outcome_to_turns(
        torch.tensor([1.0]), turn_ids, response_mask, mode="none"
    )
    assert torch.allclose(all_turns, torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0]]))
    assert torch.allclose(last_turn, torch.tensor([[0.0, 0.0, 0.0, -1.0, -1.0]]))
    assert torch.count_nonzero(no_outcome) == 0


def test_label_only_turn_advantage_is_exact_judge_output():
    labels = torch.tensor([[1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    outcomes = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])

    advantages = compose_turn_advantages(
        labels,
        outcomes,
        mode="label_only",
        label_weight=99.0,
        outcome_weight=99.0,
    )

    assert torch.equal(advantages, labels)


def test_weighted_turn_advantage_keeps_existing_behavior():
    labels = torch.tensor([[1.0, 0.0, -1.0]])
    outcomes = torch.tensor([[-1.0, -1.0, -1.0]])

    advantages = compose_turn_advantages(
        labels,
        outcomes,
        mode="weighted",
        label_weight=2.0,
        outcome_weight=0.5,
    )

    assert torch.allclose(advantages, torch.tensor([[1.5, -0.5, -2.5]]))


def test_last_turn_outcome_uses_episode_metadata_for_expanded_rows():
    outcomes = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0])
    turn_ids = torch.tensor([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    response_mask = torch.ones_like(turn_ids, dtype=torch.float32)

    result = broadcast_outcome_to_turns(
        outcomes,
        turn_ids,
        response_mask,
        mode="last_turn",
        episode_ids=np.array([10, 10, 10, 11, 11]),
        trajectory_turn_ids=np.array([0, 1, 2, 0, 1]),
    )

    assert result.tolist() == [
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [-1.0, -1.0],
    ]


def test_turn_normalization_ignores_token_length_and_copied_rows():
    turn_scores = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    turn_end_mask = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    normalized = normalize_turn_scores(
        turn_scores,
        response_mask,
        turn_end_mask,
        sample_weights=torch.tensor([1.0, 0.5, 0.5]),
    )

    assert torch.allclose(normalized[0, :2], torch.ones(2))
    assert torch.allclose(normalized[1, :], -torch.ones(4))
    assert torch.allclose(normalized[2, :], -torch.ones(4))
    assert torch.count_nonzero(normalized[0, 2:]) == 0


def test_trajectory_outcomes_are_binary_and_failure_safe():
    data = SimpleNamespace(
        non_tensor_batch={"trajectory_success": np.array([1.0, 0.0, np.nan], dtype=np.float32)},
        batch={},
    )
    outcomes, source = trajectory_outcomes(
        data, batch_size=3, device=torch.device("cpu")
    )
    assert source == "trajectory_success"
    assert outcomes.tolist() == [1.0, -1.0, -1.0]

    missing = SimpleNamespace(non_tensor_batch={}, batch={})
    outcomes, source = trajectory_outcomes(
        missing, batch_size=2, device=torch.device("cpu")
    )
    assert source == "missing"
    assert outcomes.tolist() == [-1.0, -1.0]


def test_fsdp_workers_install_ragen_wrappers():
    actor_source = inspect.getsource(ActorRolloutRefWorker.init_model)
    critic_source = inspect.getsource(CriticWorker.init_model)
    assert "RagenPPOActor" in actor_source
    assert "RagenPPOCritic" in critic_source

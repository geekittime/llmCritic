import inspect
from types import SimpleNamespace

import numpy as np
import torch

from ragen.llm_agent.ctx_manager import build_legacy_turn_metadata, build_turn_token_metadata
from ragen.trainer.agent_trainer import (
    broadcast_outcome_to_turns,
    collapse_turn_scores,
    place_turn_endpoint_rewards,
    trajectory_outcomes,
)
from ragen.trainer.core_algos import compute_turn_gae_advantage_return
from ragen.workers.actor.dp_actor import compute_turn_policy_loss
from ragen.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
from ragen.utils import redact_config


def test_tracker_config_redacts_nested_credentials_without_mutating_input():
    config = {
        "generative_critic": {
            "deepseek_api_key": "sk-not-for-logs",
            "deepseek_api_key_env": "DEEPSEEK_API_KEY",
            "max_concurrency": 8,
        },
        "wandb_token": "wb-not-for-logs",
    }

    redacted = redact_config(config)

    assert redacted["generative_critic"]["deepseek_api_key"] == "<redacted>"
    assert redacted["generative_critic"]["deepseek_api_key_env"] == "<redacted>"
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


def test_turn_reward_helpers_are_scalar_and_endpoint_based():
    turn_ids = torch.tensor([[0, 0, -1, 1, 1]])
    response_mask = (turn_ids >= 0).float()
    token_scores = torch.tensor([[1.0, 3.0, 99.0, -2.0, -4.0]])
    turn_scores = collapse_turn_scores(token_scores, turn_ids, response_mask)
    assert torch.allclose(turn_scores, torch.tensor([[2.0, 2.0, 0.0, -3.0, -3.0]]))

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
    assert torch.allclose(all_turns, torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0]]))
    assert torch.allclose(last_turn, torch.tensor([[0.0, 0.0, 0.0, -1.0, -1.0]]))


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

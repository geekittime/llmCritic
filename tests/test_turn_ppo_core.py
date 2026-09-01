import inspect
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss

from ragen.llm_agent.ctx_manager import build_legacy_turn_metadata, build_turn_token_metadata
from ragen.trainer.agent_trainer import (
    CriticAuditJsonlWriter,
    RayAgentTrainer,
    broadcast_outcome_to_turns,
    build_turn_label_observability,
    classify_turn_action,
    collapse_turn_scores,
    compose_turn_advantages,
    compute_exact_trace_observability_metrics,
    normalize_turn_scores,
    place_turn_endpoint_rewards,
    trajectory_outcomes,
    validate_deepseek_batch_health,
)
from ragen.trainer.generative_critic import FrozenGenerativeCritic, JudgePromptItem
from ragen.trainer.core_algos import compute_turn_gae_advantage_return
from ragen.workers.actor.dp_actor import (
    DataParallelPPOActor,
    compute_turn_micro_batch_scale,
    compute_turn_policy_loss,
    finalize_turn_policy_metrics,
)
from ragen.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
from ragen.utils import redact_config


def test_score_only_validation_skips_invalid_binary_confusion_target():
    trainer = object.__new__(RayAgentTrainer)
    trainer.generative_critic = SimpleNamespace(response_format="score_only")

    metrics = trainer._compute_critic_confusion_eval_metrics(batch=None)

    assert metrics == {"gen_critic/eval/confusion/skipped_no_turn_level_targets": 1.0}


def test_turn_label_observability_records_raw_output_and_action_cross_metrics():
    critic = FrozenGenerativeCritic(
        OmegaConf.create(
            {
                "generative_critic": {
                    "enable": True,
                    "backend": "deepseek_api",
                    "response_format": "score_only",
                    "parse_fail_score": 0,
                },
                "custom_envs": {},
            }
        )
    )
    before = (
        "Board size: 3 rows x 3 cols (zero-indexed).\n"
        "Boxes: (1, 1)\nPlayer: (2, 1)"
    )
    after = (
        "Board size: 3 rows x 3 cols (zero-indexed).\n"
        "Boxes: (1, 1)\nPlayer: (2, 2)"
    )
    messages = [
        {"role": "system", "content": "Solve the puzzle."},
        {"role": "user", "content": f"Turn 1\nState:\n{before}\nYou have 5 actions left"},
        {
            "role": "assistant",
            "content": "Right",
            "transition_metadata": {
                "state_before": before,
                "state_after": after,
                "is_cycle": False,
                "action_is_valid": True,
                "shortest_solution_length_before": 5,
                "shortest_solution_length_after": 4,
                "deadlock_before": False,
                "deadlock_after": False,
                "termination": {
                    "done": False,
                    "terminated": False,
                    "truncated": False,
                    "success": False,
                    "reason": None,
                },
            },
        },
        {"role": "user", "content": f"Reward:\n-0.1\nState:\n{after}\nYou have 4 actions left"},
    ]
    turn_ids = torch.tensor([[0, 0, -1]])
    labels = torch.tensor([[1.0, 1.0, 0.0]])

    metrics, records = build_turn_label_observability(
        critic=critic,
        messages_list=[messages],
        turn_ids=turn_ids,
        label_tensor=labels,
        raw_outputs=["FINAL_SCORE: 1"],
        non_tensor_batch={
            "episode_ids": np.array([7]),
            "trajectory_turn_ids": np.array([2]),
            "env_ids": np.array([11]),
        },
    )

    assert metrics["train/label_nonzero_rate"] == pytest.approx(1.0)
    assert metrics["train/action_type/move_rate"] == pytest.approx(1.0)
    assert metrics["train/action_label/move_positive_conditional_rate"] == pytest.approx(1.0)
    assert metrics["train/cycle/metadata_available_rate"] == pytest.approx(1.0)
    assert metrics["train/solver_relation/closer_rate"] == pytest.approx(1.0)
    assert metrics[
        "train/solver_relation_label/closer_positive_conditional_rate"
    ] == pytest.approx(1.0)
    assert records[0]["episode_id"] == 7
    assert records[0]["trajectory_turn_id"] == 2
    assert records[0]["action"] == "Right"
    assert records[0]["action_type"] == "move"
    assert records[0]["parse_valid"] is True
    assert records[0]["raw_output"] == "FINAL_SCORE: 1"
    assert records[0]["solver_progress_relation"] == "closer"
    assert records[0]["shortest_solution_length_before"] == 5
    assert records[0]["shortest_solution_length_after"] == 4
    assert records[0]["solution_effort_delta"] == -1
    assert records[0]["deadlock_before"] is False
    assert records[0]["deadlock_after"] is False
    assert records[0]["termination_done"] is False
    assert records[0]["termination_terminated"] is False
    assert records[0]["termination_truncated"] is False
    assert records[0]["termination_success"] is False
    assert records[0]["termination_reason"] is None
    assert len(records[0]["prompt_sha256"]) == 64


def test_turn_label_observability_records_farther_terminal_failure():
    critic = FrozenGenerativeCritic(
        OmegaConf.create(
            {
                "generative_critic": {
                    "enable": True,
                    "backend": "deepseek_api",
                    "response_format": "score_only",
                    "parse_fail_score": 0,
                },
                "custom_envs": {},
            }
        )
    )
    messages = [
        {"role": "system", "content": "Solve the puzzle."},
        {"role": "user", "content": "Turn 1\nState:\nbefore"},
        {
            "role": "assistant",
            "content": "Left",
            "transition_metadata": {
                "state_before": "before",
                "state_after": "after",
                "shortest_solution_length_before": 4,
                "shortest_solution_length_after": 5,
                "deadlock_before": False,
                "deadlock_after": False,
                "termination": {
                    "done": True,
                    "terminated": False,
                    "truncated": True,
                    "success": False,
                    "reason": "max_actions",
                },
            },
        },
        {"role": "user", "content": "State:\nafter"},
    ]

    metrics, records = build_turn_label_observability(
        critic=critic,
        messages_list=[messages],
        turn_ids=torch.tensor([[0, 0]]),
        label_tensor=torch.tensor([[-1.0, -1.0]]),
        raw_outputs=["FINAL_SCORE: -1"],
    )

    assert metrics["train/solver_relation/farther_rate"] == pytest.approx(1.0)
    assert metrics[
        "train/solver_relation_label/farther_negative_conditional_rate"
    ] == pytest.approx(1.0)
    assert records[0]["solver_progress_relation"] == "farther"
    assert records[0]["solution_effort_delta"] == 1
    assert records[0]["termination_done"] is True
    assert records[0]["termination_terminated"] is False
    assert records[0]["termination_truncated"] is True
    assert records[0]["termination_success"] is False
    assert records[0]["termination_reason"] == "max_actions"


def test_action_classifier_separates_push_noop_and_invalid():
    before = "Boxes: (1, 1)\nPlayer: (2, 1)"
    pushed = "Boxes: (0, 1)\nPlayer: (1, 1)"
    assert classify_turn_action(
        action="Up", state_before=before, state_after=pushed
    ) == "push"
    assert classify_turn_action(
        action="Up", state_before=before, state_after=before
    ) == "no_op"
    assert classify_turn_action(
        action="<no action executed>", state_before=before, state_after=before
    ) == "invalid"


def test_action_classifier_prefers_transition_facts_over_rendered_state():
    same_state = "Boxes: (1, 1)\nPlayer: (2, 1)"
    misleading_box_change = "Boxes: (0, 1)\nPlayer: (1, 1)"

    assert classify_turn_action(
        action="Up",
        state_before=same_state,
        state_after=same_state,
        assistant_message={"transition_metadata": {"moved_box": True}},
    ) == "push"
    assert classify_turn_action(
        action="Up",
        state_before=same_state,
        state_after=misleading_box_change,
        assistant_message={
            "transition_metadata": {
                "moved_box": False,
                "moved_player": False,
                "action_is_effective": False,
                "action_is_valid": False,
            }
        },
    ) == "no_op"
    assert classify_turn_action(
        action="Up",
        state_before=same_state,
        state_after=same_state,
        assistant_message={
            "transition_metadata": {"action": {"mapping_valid": False}}
        },
    ) == "invalid"


def test_critic_audit_writer_caps_deduplicates_and_redacts_secrets(tmp_path):
    audit_path = tmp_path / "critic.jsonl"
    writer = CriticAuditJsonlWriter(
        str(audit_path), sample_rate=1.0, max_records=1
    )
    record = {
        "prompt_sha256": "a" * 64,
        "episode_id": 1,
        "trajectory_turn_id": 0,
        "turn_id": 0,
        "raw_output": "accidentally echoed sk-abcdefghijklmnop",
    }

    assert writer.write([record, record], step=3) == 1
    assert writer.write([record], step=3) == 0
    writer.close()

    payload = audit_path.read_text(encoding="utf-8")
    assert "sk-abcdefghijklmnop" not in payload
    assert "<redacted>" in payload
    assert len(payload.splitlines()) == 1
    assert audit_path.stat().st_mode & 0o077 == 0


def _make_trainable_label_trainer(prompt_items, generated_outputs):
    trainer = object.__new__(RayAgentTrainer)
    trainer.generative_critic = FrozenGenerativeCritic(
        OmegaConf.create(
            {
                "generative_critic": {
                    "enable": True,
                    "backend": "transformers",
                    "response_format": "score_only",
                    "parse_fail_score": 0,
                },
                "custom_envs": {},
            }
        )
    )
    trainer.generative_critic.build_judge_prompts = lambda **_: list(prompt_items)
    trainer.config = OmegaConf.create(
        {"trainer": {"n_gpus_per_node": 1, "nnodes": 1}}
    )
    trainer._build_generative_critic_prompt_batch = lambda prompts: list(prompts)

    class FakeCriticWorker:
        def __init__(self):
            self.requests = []

        def generate_critic_sequences(self, prompts):
            self.requests.append(list(prompts))
            return SimpleNamespace(
                batch=None,
                non_tensor_batch={
                    "response_texts": np.asarray(generated_outputs, dtype=object)
                },
            )

    trainer.critic_wg = FakeCriticWorker()
    return trainer


def test_trainable_critic_forced_items_skip_generation_and_keep_output_order(monkeypatch):
    prompt_items = [
        JudgePromptItem(0, 0, "forced", forced_score=-1, force_reason="invalid"),
        JudgePromptItem(1, 0, "request-one"),
        JudgePromptItem(2, 0, "request-two"),
    ]
    trainer = _make_trainable_label_trainer(
        prompt_items,
        generated_outputs=["FINAL_SCORE: 1"],
    )
    monkeypatch.setattr(
        "ragen.trainer.agent_trainer.pad_dataproto_to_divisor",
        lambda value, size_divisor: (value, 0),
    )
    monkeypatch.setattr(
        "ragen.trainer.agent_trainer.unpad_dataproto",
        lambda value, _: value,
    )
    turn_ids = torch.tensor([[0, 0], [0, 0], [0, 0]])

    labels, metrics, _, raw_outputs = trainer._infer_labels_with_trainable_critic(
        messages_list=[[], [], []],
        turn_ids=turn_ids,
    )

    assert trainer.critic_wg.requests == [["request-one", "request-two"]]
    assert labels[:, 0].tolist() == [-1.0, 1.0, 0.0]
    assert raw_outputs == ["", "FINAL_SCORE: 1", ""]
    assert metrics["gen_critic/submitted_prompt_count"] == 2.0
    assert metrics["gen_critic/rule_forced_negative_count"] == 1.0
    assert metrics["gen_critic/model_output_count_mismatch"] == 1.0
    assert metrics["gen_critic/parse_fail_rate"] == pytest.approx(1.0 / 3.0)


def test_trainable_critic_all_forced_never_builds_or_calls_model():
    prompt_items = [
        JudgePromptItem(0, 0, "forced-a", forced_score=-1, force_reason="invalid"),
        JudgePromptItem(1, 0, "forced-b", forced_score=-1, force_reason="invalid"),
    ]
    trainer = _make_trainable_label_trainer(prompt_items, generated_outputs=[])
    trainer._build_generative_critic_prompt_batch = lambda _: pytest.fail(
        "all-forced batch must not build model inputs"
    )
    turn_ids = torch.tensor([[0], [0]])

    labels, metrics, generation_time, raw_outputs = (
        trainer._infer_labels_with_trainable_critic(
            messages_list=[[], []],
            turn_ids=turn_ids,
        )
    )

    assert trainer.critic_wg.requests == []
    assert labels[:, 0].tolist() == [-1.0, -1.0]
    assert raw_outputs == ["", ""]
    assert generation_time == 0.0
    assert metrics["gen_critic/submitted_prompt_count"] == 0.0
    assert metrics["gen_critic/rule_forced_negative_count"] == 2.0
    assert metrics["gen_critic/parse_fail_rate"] == 0.0


def test_trainable_critic_truncates_extra_model_outputs(monkeypatch):
    prompt_items = [
        JudgePromptItem(0, 0, "forced", forced_score=-1, force_reason="invalid"),
        JudgePromptItem(1, 0, "request"),
    ]
    trainer = _make_trainable_label_trainer(
        prompt_items,
        generated_outputs=["FINAL_SCORE: -1", "FINAL_SCORE: 1", "FINAL_SCORE: 1"],
    )
    monkeypatch.setattr(
        "ragen.trainer.agent_trainer.pad_dataproto_to_divisor",
        lambda value, size_divisor: (value, 0),
    )
    monkeypatch.setattr(
        "ragen.trainer.agent_trainer.unpad_dataproto",
        lambda value, _: value,
    )

    labels, metrics, _, raw_outputs = trainer._infer_labels_with_trainable_critic(
        messages_list=[[], []],
        turn_ids=torch.tensor([[0], [0]]),
    )

    assert labels[:, 0].tolist() == [-1.0, -1.0]
    assert raw_outputs == ["", "FINAL_SCORE: -1"]
    assert metrics["gen_critic/model_output_count"] == 3.0
    assert metrics["gen_critic/model_output_count_mismatch"] == 2.0


def test_tracker_config_redacts_nested_credentials_without_mutating_input():
    config = {
        "generative_critic": {
            "deepseek_api_key": "sk-not-for-logs",
            "deepseek_api_key_env": "DEEPSEEK_API_KEY",
            "deepseek_api_key_file": "/protected/secrets.env",
            "max_concurrency": 8,
        },
        "wandb_token": "wb-not-for-logs",
        "tokenizer_path": "/models/tokenizer",
        "max_tokens": 40,
        "ppo_max_token_len_per_gpu": 12288,
        "deepseek_abort_on_auth_failure": True,
        "authorization_header": "Bearer secret-for-test",
    }

    redacted = redact_config(config)

    assert redacted["generative_critic"]["deepseek_api_key"] == "<redacted>"
    assert redacted["generative_critic"]["deepseek_api_key_env"] == "<redacted>"
    assert redacted["generative_critic"]["deepseek_api_key_file"] == "<redacted>"
    assert redacted["wandb_token"] == "<redacted>"
    assert redacted["generative_critic"]["max_concurrency"] == 8
    assert redacted["tokenizer_path"] == "/models/tokenizer"
    assert redacted["max_tokens"] == 40
    assert redacted["ppo_max_token_len_per_gpu"] == 12288
    assert redacted["deepseek_abort_on_auth_failure"] is True
    assert redacted["authorization_header"] == "<redacted>"
    assert config["generative_critic"]["deepseek_api_key"] == "sk-not-for-logs"


def test_exact_trace_observability_uses_action_mask_not_shifted_sequence():
    batch = DataProto(
        batch=TensorDict(
            {
                "attention_mask": torch.tensor(
                    [[0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]],
                    dtype=torch.long,
                ),
                "response_mask": torch.tensor(
                    [[0, 0, 0, 0, 1, 1], [0, 1, 1, 1, 1, 1]],
                    dtype=torch.float32,
                ),
                "sample_weights": torch.ones(2, dtype=torch.float32),
            },
            batch_size=2,
        )
    )

    metrics = compute_exact_trace_observability_metrics(
        batch,
        {"gen": 14.0},
        max_response_length=5,
        max_model_len=10,
    )

    assert metrics["response_length/mean"] == pytest.approx(3.5)
    assert metrics["response_length/max"] == pytest.approx(5.0)
    assert metrics["response_length/clip_ratio"] == pytest.approx(0.5)
    assert metrics["prompt_length/mean"] == pytest.approx(2.5)
    assert metrics["prompt_length/max"] == pytest.approx(3.0)
    assert metrics["perf/generated_action_tokens"] == pytest.approx(7.0)
    assert metrics["timing_per_token_ms/gen"] == pytest.approx(2000.0)


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


def test_turn_policy_metrics_weight_dynamic_microbatches_and_copied_rows():
    micro_metrics = [
        {
            "actor/pg_loss": 1.0,
            "actor/pg_clipfrac": 0.0,
            "actor/ppo_kl": 0.2,
            "actor/pg_clipfrac_lower": 0.4,
        },
        {
            "actor/pg_loss": 5.0,
            "actor/pg_clipfrac": 1.0,
            "actor/ppo_kl": 0.8,
            "actor/pg_clipfrac_lower": 0.1,
        },
    ]
    # The first dynamic micro-batch contains one half-weight padding copy;
    # the second contains two original turns plus its matching half-copy.
    micro_weights = [0.5, 2.5]
    metric_sums = {
        key: sum(values[key] * weight for values, weight in zip(micro_metrics, micro_weights, strict=True))
        for key in micro_metrics[0]
    }

    metrics = finalize_turn_policy_metrics(metric_sums, sum(micro_weights))

    for key in micro_metrics[0]:
        expected = sum(
            values[key] * weight
            for values, weight in zip(micro_metrics, micro_weights, strict=True)
        ) / sum(micro_weights)
        assert metrics[key] == pytest.approx(expected)
    assert metrics["actor/pg_loss"] != pytest.approx(3.0)


def test_turn_policy_metrics_sum_unequal_fsdp_rank_weights(monkeypatch):
    local_sums = {
        "actor/pg_loss": 1.0,
        "actor/pg_clipfrac": 0.25,
        "actor/ppo_kl": 0.5,
        "actor/pg_clipfrac_lower": 0.75,
    }
    local_weight = 1.0
    remote_packed = torch.tensor(
        [9.0, 2.25, 4.5, 6.75, 3.0],
        dtype=torch.float64,
    )

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fake_all_reduce(packed, op):
        assert op == torch.distributed.ReduceOp.SUM
        packed.add_(remote_packed)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    metrics = finalize_turn_policy_metrics(
        local_sums,
        local_weight,
        collective_device=torch.device("cpu"),
    )

    assert metrics == {
        "actor/pg_loss": pytest.approx(2.5),
        "actor/pg_clipfrac": pytest.approx(0.625),
        "actor/ppo_kl": pytest.approx(1.25),
        "actor/pg_clipfrac_lower": pytest.approx(1.875),
    }


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


def test_deepseek_batch_health_allows_isolated_negative_fallbacks():
    validate_deepseek_batch_health(
        {
            "gen_critic/api_failure_rate": 0.01,
            "gen_critic/parse_fail_rate": 0.02,
            "gen_critic/api_auth_failure_count": 0.0,
        }
    )


@pytest.mark.parametrize(
    "metrics",
    [
        {"gen_critic/api_auth_failure_count": 1.0},
        {"gen_critic/api_missing_key": 1.0},
        {"gen_critic/api_failure_rate": 0.251},
        {"gen_critic/parse_fail_rate": 0.251},
    ],
)
def test_deepseek_batch_health_stops_systemic_failures(metrics):
    with pytest.raises(RuntimeError, match="rejected before actor update"):
        validate_deepseek_batch_health(metrics)


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

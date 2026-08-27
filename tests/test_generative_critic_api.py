"""Focused tests for the signed DeepSeek critic protocol.

The tests use an in-memory OpenAI-compatible client and never contact the
network.  They intentionally exercise malformed output and repeated prompts,
which are the two failure modes that can silently corrupt an RL rollout.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from ragen.trainer.generative_critic import FrozenGenerativeCritic


def _config(**critic_overrides):
    critic = {
        "enable": True,
        "backend": "deepseek_api",
        "deepseek_api_key": "test-only-key",
        "deepseek_max_retries": 0,
        "deepseek_max_concurrency": 4,
        "deepseek_cache_enable": True,
        "deepseek_cache_size": 16,
        "max_new_tokens": 8,
        "response_format": "score_only",
    }
    critic.update(critic_overrides)
    return OmegaConf.create({"model_path": "unused", "generative_critic": critic})


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Useful setup.\nFINAL_SCORE: 1", 1),
        ("No progress.\nFINAL_SCORE: 0", 0),
        ("Deadlock.\nFINAL_SCORE: -1", -1),
        ("1", 1),
        ("```\nFINAL_SCORE: -1\n```", -1),
        ("<answer>0</answer>", 0),
        ("The coordinates include 1 and 0.\nNo final score.", -1),
        ("FINAL_SCORE: 2", -1),
        ("###label: False", -1),
        ("", -1),
    ],
)
def test_parse_score_requires_explicit_final_line(text, expected):
    assert FrozenGenerativeCritic.parse_score(text) == expected


def test_score_prompt_contains_transition_and_machine_contract():
    critic = FrozenGenerativeCritic(_config())
    prompt = critic._build_single_prompt(
        state_before="before-state",
        action_text="Right",
        state_after="after-state",
        observed_reward="-0.1",
        turn_number=3,
        has_after_state=True,
        env_instruction="Solve the puzzle",
        critic_instruction=None,
        response_format="score_only",
    )
    assert "before-state" in prompt
    assert "Right" in prompt
    assert "after-state" in prompt
    assert "-0.1" in prompt
    assert "FINAL_SCORE: 1" in prompt
    assert "FINAL_SCORE: 0" in prompt
    assert "FINAL_SCORE: -1" in prompt


def test_api_backend_forces_integer_protocol_when_base_config_is_legacy():
    config = _config(response_format="structured")
    critic = FrozenGenerativeCritic(config)
    assert critic.response_format == "score_only"


def test_environment_key_takes_precedence_over_config(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-key")
    critic = FrozenGenerativeCritic(_config(deepseek_api_key="config-key"))
    assert critic.deepseek_api_key == "environment-key"


class _FakeCompletions:
    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(0)
        result = self.responder(kwargs)
        if isinstance(result, BaseException):
            raise result
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=result))]
        )


class _FakeClient:
    def __init__(self, responder):
        self.chat = SimpleNamespace(completions=_FakeCompletions(responder))


def test_api_requests_are_parallel_deduplicated_and_cached():
    fake = _FakeClient(lambda kwargs: "FINAL_SCORE: 1")
    critic = FrozenGenerativeCritic(_config())
    critic._deepseek_client = fake

    first = critic._generate_texts(["same", "other", "same"])
    assert first == ["FINAL_SCORE: 1"] * 3
    assert len(fake.chat.completions.calls) == 2
    assert critic._last_generation_metadata["gen_critic/api_deduplicated_count"] == 1.0

    second = critic._generate_texts(["same", "new"])
    assert second == ["FINAL_SCORE: 1", "FINAL_SCORE: 1"]
    assert len(fake.chat.completions.calls) == 3
    assert critic._last_generation_metadata["gen_critic/api_cache_hit_count"] == 1.0


def test_malformed_api_output_is_not_cached():
    fake = _FakeClient(lambda kwargs: "rationale without a final score")
    critic = FrozenGenerativeCritic(_config())
    critic._deepseek_client = fake
    assert critic._generate_texts(["same"]) == ["rationale without a final score"]
    assert critic._generate_texts(["same"]) == ["rationale without a final score"]
    assert len(fake.chat.completions.calls) == 2


def test_auth_failure_does_not_retry_and_maps_to_negative():
    class _AuthError(Exception):
        status_code = 401

    fake = _FakeClient(lambda kwargs: _AuthError())
    critic = FrozenGenerativeCritic(_config(deepseek_max_retries=5))
    critic._deepseek_client = fake
    outputs = critic._generate_texts(["one", "two"])
    assert outputs == ["", ""]
    assert len(fake.chat.completions.calls) == 2
    assert critic._last_generation_metadata["gen_critic/api_auth_failure_count"] == 2.0


def _messages(action: str):
    return [
        {"role": "system", "content": "Solve the task."},
        {
            "role": "user",
            "content": "Turn 0\nState:\nbefore\nYou have 1 action.",
        },
        {"role": "assistant", "content": action},
    ]


def test_infer_turn_labels_uses_signed_scores_and_api_failure_is_negative():
    def responder(kwargs):
        prompt = kwargs["messages"][1]["content"]
        if "ACTION_POS" in prompt:
            return "reason\nFINAL_SCORE: 1"
        if "ACTION_NEU" in prompt:
            return "reason\nFINAL_SCORE: 0"
        if "ACTION_BAD" in prompt:
            return "reason mentions 1 but no marker"
        return "reason\nFINAL_SCORE: -1"

    critic = FrozenGenerativeCritic(_config())
    critic._deepseek_client = _FakeClient(responder)
    messages = [_messages("ACTION_POS"), _messages("ACTION_NEU"), _messages("ACTION_BAD")]
    turn_ids = torch.zeros((3, 2), dtype=torch.long)

    scores, metrics, outputs = critic.infer_turn_labels(messages, turn_ids)

    assert scores[:, 0].tolist() == [1.0, 0.0, -1.0]
    assert scores[:, 1].tolist() == [1.0, 0.0, -1.0]
    assert metrics["gen_critic/parse_fail_rate"] == pytest.approx(1 / 3)
    assert len(outputs) == 3


def test_missing_api_key_preserves_batch_shape_and_reports_failure(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    critic = FrozenGenerativeCritic(
        OmegaConf.create(
            {
                "generative_critic": {
                    "enable": True,
                    "backend": "deepseek_api",
                    "deepseek_cache_enable": False,
                }
            }
        )
    )
    outputs = critic._generate_texts(["a", "b"])
    assert outputs == ["", ""]
    assert critic._last_generation_metadata["gen_critic/api_batch_failed"] == 1.0

"""Focused tests for the signed DeepSeek critic protocol.

The tests use an in-memory OpenAI-compatible client and never contact the
network.  They intentionally exercise malformed output and repeated prompts,
which are the two failure modes that can silently corrupt an RL rollout.
"""

from __future__ import annotations

import asyncio
import inspect
import time
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
        critic_instruction="Prefer legal moves. Output ###label: True or ###label: False.",
        response_format="score_only",
    )
    assert "before-state" in prompt
    assert "Right" in prompt
    assert "after-state" in prompt
    assert "-0.1" in prompt
    assert "FINAL_SCORE: 1" in prompt
    assert "FINAL_SCORE: 0" in prompt
    assert "FINAL_SCORE: -1" in prompt
    assert "Output exactly one line and nothing else" in prompt
    assert "ignore any conflicting True/False output instruction" in prompt
    assert "rationale" not in prompt.lower()

    system_message = critic._build_deepseek_messages(prompt)[0]["content"]
    assert "Return exactly one line" in system_message
    assert "rationale" not in system_message.lower()


def test_api_backend_forces_integer_protocol_when_base_config_is_legacy():
    config = _config(response_format="structured")
    critic = FrozenGenerativeCritic(config)
    assert critic.response_format == "score_only"


def test_deepseek_request_disables_thinking_by_default():
    critic = FrozenGenerativeCritic(_config())
    assert critic.deepseek_model == "deepseek-v4-flash"
    assert critic.deepseek_thinking == "disabled"
    fake = _FakeClient(lambda kwargs: "FINAL_SCORE: 0")
    critic._deepseek_client = fake
    critic._generate_texts(["one"])
    request = fake.chat.completions.calls[0]
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}


def test_environment_key_takes_precedence_over_config(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-key")
    critic = FrozenGenerativeCritic(_config(deepseek_api_key="config-key"))
    assert critic.deepseek_api_key == "environment-key"


class _FakeCompletions:
    def __init__(self, responder, usage=None):
        self.responder = responder
        self.usage = usage
        self.calls = []
        self.event_loop_ids = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        self.event_loop_ids.append(id(asyncio.get_running_loop()))
        await asyncio.sleep(0)
        result = self.responder(kwargs)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, BaseException):
            raise result
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=result))],
            usage=self.usage,
        )


class _FakeClient:
    def __init__(self, responder, usage=None):
        self.chat = SimpleNamespace(completions=_FakeCompletions(responder, usage=usage))
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1


def test_api_requests_are_parallel_deduplicated_and_cached():
    usage = SimpleNamespace(prompt_tokens=11, completion_tokens=3)
    fake = _FakeClient(lambda kwargs: "FINAL_SCORE: 1", usage=usage)
    critic = FrozenGenerativeCritic(_config())
    critic._deepseek_client = fake

    first = critic._generate_texts(["same", "other", "same"])
    assert first == ["FINAL_SCORE: 1"] * 3
    assert len(fake.chat.completions.calls) == 2
    assert critic._last_generation_metadata["gen_critic/api_deduplicated_count"] == 1.0
    assert critic._last_generation_metadata["gen_critic/api_deduplicated_rate"] == pytest.approx(1 / 3)
    assert critic._last_generation_metadata["gen_critic/api_cache_hit_rate"] == 0.0
    assert critic._last_generation_metadata["gen_critic/api_input_token_count"] == 22.0
    assert critic._last_generation_metadata["gen_critic/api_output_token_count"] == 6.0
    assert critic._last_generation_metadata["gen_critic/api_usage_reported_request_count"] == 2.0
    assert critic._last_generation_metadata["gen_critic/api_wall_time_s"] > 0.0
    assert critic._last_generation_metadata["gen_critic/api_labels_per_second"] > 0.0

    second = critic._generate_texts(["same", "new"])
    assert second == ["FINAL_SCORE: 1", "FINAL_SCORE: 1"]
    assert len(fake.chat.completions.calls) == 3
    assert critic._last_generation_metadata["gen_critic/api_cache_hit_count"] == 1.0
    assert critic._last_generation_metadata["gen_critic/api_cache_hit_rate"] == 0.5
    assert critic._last_generation_metadata["gen_critic/api_request_avoidance_rate"] == 0.5


def test_cache_and_dedupe_use_the_actual_truncated_prompt():
    fake = _FakeClient(lambda kwargs: "FINAL_SCORE: 0")
    critic = FrozenGenerativeCritic(_config(deepseek_max_prompt_chars=40))
    critic._deepseek_client = fake
    prefix = "p" * 20
    suffix = "s" * 20
    first_prompt = prefix + ("a" * 100) + suffix
    second_prompt = prefix + ("b" * 100) + suffix

    assert critic._generate_texts([first_prompt, second_prompt]) == ["FINAL_SCORE: 0"] * 2
    assert len(fake.chat.completions.calls) == 1
    sent_prompt = fake.chat.completions.calls[0]["messages"][1]["content"]
    assert sent_prompt == critic._truncate_deepseek_prompt(first_prompt)
    assert critic._last_generation_metadata["gen_critic/api_deduplicated_count"] == 1.0

    assert critic._generate_texts([second_prompt]) == ["FINAL_SCORE: 0"]
    assert len(fake.chat.completions.calls) == 1
    assert critic._last_generation_metadata["gen_critic/api_cache_hit_rate"] == 1.0


def test_persistent_loop_and_owned_client_are_reused_and_closed(monkeypatch):
    import openai

    created_clients = []

    def make_client(**kwargs):
        client = _FakeClient(lambda request: "FINAL_SCORE: 1")
        created_clients.append(client)
        return client

    monkeypatch.setattr(openai, "AsyncOpenAI", make_client)
    critic = FrozenGenerativeCritic(_config(deepseek_cache_enable=False))

    assert critic._generate_texts(["first"]) == ["FINAL_SCORE: 1"]
    assert critic._generate_texts(["second"]) == ["FINAL_SCORE: 1"]
    assert len(created_clients) == 1
    loop_ids = created_clients[0].chat.completions.event_loop_ids
    assert len(loop_ids) == 2
    assert len(set(loop_ids)) == 1

    critic.close()
    assert created_clients[0].close_calls == 1
    assert critic._deepseek_event_loop is None


def test_sync_api_call_is_safe_inside_an_active_event_loop():
    fake = _FakeClient(lambda kwargs: "FINAL_SCORE: -1")
    critic = FrozenGenerativeCritic(_config())
    critic._deepseek_client = fake

    async def invoke_sync_api():
        return critic._generate_texts(["inside-running-loop"])

    assert asyncio.run(invoke_sync_api()) == ["FINAL_SCORE: -1"]


def test_retry_delay_includes_small_bounded_jitter(monkeypatch):
    monkeypatch.setattr("ragen.trainer.generative_critic.random.uniform", lambda low, high: high)
    assert FrozenGenerativeCritic._retry_delay(0) == pytest.approx(1.1)
    assert FrozenGenerativeCritic._retry_delay(3) == pytest.approx(8.25)


def test_batch_deadline_keeps_completed_results_and_times_out_pending_requests():
    async def responder(kwargs):
        prompt = kwargs["messages"][1]["content"]
        if prompt == "slow":
            await asyncio.sleep(1.0)
            return "FINAL_SCORE: -1"
        return "FINAL_SCORE: 1"

    fake = _FakeClient(responder)
    critic = FrozenGenerativeCritic(
        _config(deepseek_batch_timeout=0.02, deepseek_max_retries=0, deepseek_cache_enable=False)
    )
    critic._deepseek_client = fake

    started_at = time.perf_counter()
    outputs = critic._generate_texts(["fast", "slow", "slow"])
    wall_time = time.perf_counter() - started_at

    assert outputs == ["FINAL_SCORE: 1", "", ""]
    assert wall_time < 0.5
    assert critic._last_generation_metadata["gen_critic/api_batch_timeout_count"] == 2.0
    assert critic._last_generation_metadata["gen_critic/api_batch_timeout"] == 1.0
    assert critic._last_generation_metadata["gen_critic/api_timeout_count"] == 2.0
    assert critic._last_generation_metadata["gen_critic/api_failure_count"] == 2.0
    assert critic._last_generation_metadata["gen_critic/api_batch_failed"] == 0.0


def test_nonpositive_batch_deadline_is_disabled():
    async def responder(kwargs):
        await asyncio.sleep(0.01)
        return "FINAL_SCORE: 0"

    fake = _FakeClient(responder)
    critic = FrozenGenerativeCritic(
        _config(deepseek_batch_timeout=0, deepseek_max_retries=0, deepseek_cache_enable=False)
    )
    critic._deepseek_client = fake

    assert critic._generate_texts(["wait-for-result"]) == ["FINAL_SCORE: 0"]
    assert critic._last_generation_metadata["gen_critic/api_batch_timeout"] == 0.0


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
    assert critic._last_generation_metadata["gen_critic/api_request_count"] == 0.0

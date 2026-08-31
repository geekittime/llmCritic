import pytest
from ragen.llm_agent.ctx_manager import ContextManager
from omegaconf import OmegaConf
from verl.protocol import DataProto

class DummyTokenizer:
    name_or_path = "qwen"  # or "llama-3" or any string your code expects

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return " ".join([msg["content"] for msg in messages])

    def __call__(self, texts, return_tensors, padding, padding_side, truncation):
        import torch
        batch_size = len(texts) if isinstance(texts, list) else 1
        class DummyOutput:
            input_ids = torch.tensor([[1, 2, 3]]).repeat(batch_size, 1)
            attention_mask = torch.tensor([[1, 1, 1]]).repeat(batch_size, 1)
        return DummyOutput()

    def encode(self, text):
        # Return a dummy list of token ids; must be at least length 1 for [0] indexing
        return [42, 43]


class CountingTokenizer:
    name_or_path = "qwen-counting"

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        text = "|".join(message["content"] for message in messages)
        return text + ("|GEN|" if add_generation_prompt else "")

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text)))}

    def encode(self, text, **kwargs):
        return list(range(max(len(text), 1)))

@pytest.fixture
def dummy_config():
    cfg = OmegaConf.create({
        "agent_proxy": {
            "context_window_mode": "limited_multi_turn",
            "max_context_window": 2,
            # This unit fixture predates exact rollout token traces and
            # intentionally exercises the explicit legacy text path.
            "allow_legacy_retokenization": True,
            "enable_think": False,
            "use_turn_scores": False,
            "action_sep": "|",
            "max_actions_per_turn": 1,
            "reward_normalization": {
                "grouping": "batch",
                "method": "identity"
            }
        },
        "enable_response_mask": False,
        "es_manager": {
            "train": {
                "env_configs": {
                    "n_groups": [1],
                    "tags": ["sokoban"]
                },
                "group_size": 1
            }
        },
        "custom_envs": {
            "sokoban": {
                "env_type": "sokoban",
                "max_actions_per_traj": 10
            }
        },
        "actor_rollout_ref": {
            "rollout": {
                "response_length": 128
            }
        }
    })
    return cfg

def test_context_window_truncation(dummy_config):
    tokenizer = DummyTokenizer()
    ctx = ContextManager(config=dummy_config, tokenizer=tokenizer, mode="train")
    ctx.prefix_lookup = {0: "Initial prompt"}
    ctx.env_config_lookup = {0: {"max_tokens": 128}}
    ctx.env_nums = {"": 1}  # For metrics

    env_outputs = [{
        "env_id": 0,
        "group_id": 0,
        "tag": "sokoban",
        "history": [
            {"state": "S1", "llm_response": "R1", "reward": 0.1, "actions_left": 5},
            {"state": "S2", "llm_response": "R2", "reward": 0.2, "actions_left": 4},
            {"state": "S3", "llm_response": "R3", "reward": 0.3, "actions_left": 3},
        ],
        "metrics": {},
    }]

    lm_inputs: DataProto = ctx.get_lm_inputs(env_outputs, prepare_for_update=True)
    messages = lm_inputs.non_tensor_batch["messages_list"][-1]

    # Ensure only last 2 turns are present
    assert "S1" not in str(messages)
    assert "S2" in str(messages)
    assert "S3" in str(messages)


def test_response_parser_flags_extra_actions_and_unparsed_suffix(dummy_config):
    ctx = ContextManager(config=dummy_config, tokenizer=DummyTokenizer(), mode="train")

    canonical, actions, metadata = ctx._parse_response(
        "<answer>Left | Right</answer>"
    )
    assert canonical == "<answer>Left</answer>"
    assert actions == ["Left"]
    assert metadata["response_format_valid"] is True
    assert metadata["action_count_exceeded"] is True
    assert metadata["raw_action_count"] == 2

    raw = "<answer>Left</answer> trailing text"
    canonical, actions, metadata = ctx._parse_response(raw)
    assert canonical == raw
    assert actions == []
    assert metadata["response_format_valid"] is False

    canonical, actions, metadata = ctx._parse_response("<answer>Left |</answer>")
    assert actions == ["Left"]
    assert metadata["action_format_valid"] is False

    canonical, actions, metadata = ctx._parse_response(
        "<answer>Left</answer><answer>Right</answer>"
    )
    assert metadata["action_format_valid"] is False


def test_prompt_budget_and_single_action_instruction_match_real_rollout(dummy_config):
    dummy_config.actor_rollout_ref.rollout.response_length = 40
    dummy_config.custom_envs.sokoban.max_tokens = 120
    ctx = ContextManager(config=dummy_config, tokenizer=DummyTokenizer(), mode="train")

    assert ctx.env_config_lookup[0]["max_tokens"] == 40
    assert "exactly one valid action" in ctx.prefix_lookup[0]


def test_generation_truncation_reserves_the_full_completion_budget(dummy_config):
    dummy_config.actor_rollout_ref.rollout.max_model_len = 80
    dummy_config.actor_rollout_ref.rollout.response_length = 10
    ctx = ContextManager(config=dummy_config, tokenizer=CountingTokenizer(), mode="train")
    messages = [
        {"role": "system", "content": "S" * 10},
        {"role": "user", "content": "O" * 40},
        {"role": "assistant", "content": "A" * 5},
        {"role": "user", "content": "Reward: " + "R" * 5},
        {"role": "user", "content": "C" * 20},
    ]

    truncated = ctx._apply_max_length(messages, add_generation_prompt=True)
    prompt_text = ctx._render_messages_for_length(truncated, add_generation_prompt=True)

    assert len(prompt_text) <= 70
    assert len(prompt_text) + dummy_config.actor_rollout_ref.rollout.response_length <= 80
    assert truncated[0] == messages[0]
    assert truncated[-1] == messages[-1]


def test_non_zero_metrics_skip_binary_rates(dummy_config):
    ctx = ContextManager(config=dummy_config, tokenizer=DummyTokenizer(), mode="train")
    ctx.env_nums = {"sokoban": 2}
    env_outputs = [
        {
            "metrics": {
                "sokoban/success": 0.0,
                "sokoban/pass@2": 0.0,
                "sokoban/episodic_return": 0.0,
            }
        },
        {
            "metrics": {
                "sokoban/success": 1.0,
                "sokoban/pass@2": 1.0,
                "sokoban/episodic_return": 1.0,
            }
        },
    ]

    metrics = ctx._compute_metrics(env_outputs, response_length=5.0)

    assert "sokoban/non-zero/success" not in metrics
    assert "sokoban/non-zero/pass@2" not in metrics
    assert metrics["sokoban/non-zero/episodic_return"] == pytest.approx(1.0)


def test_generation_truncation_fails_when_current_turn_cannot_fit(dummy_config):
    dummy_config.actor_rollout_ref.rollout.max_model_len = 30
    dummy_config.actor_rollout_ref.rollout.response_length = 10
    ctx = ContextManager(config=dummy_config, tokenizer=CountingTokenizer(), mode="train")
    messages = [
        {"role": "system", "content": "S" * 15},
        {"role": "user", "content": "C" * 15},
    ]

    with pytest.raises(ValueError, match="current turn alone"):
        ctx._apply_max_length(messages, add_generation_prompt=True)

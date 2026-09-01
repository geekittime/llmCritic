"""CPU-only integration coverage for the exact turn-credit pipeline."""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto

from ragen.env.sokoban.env import SokobanEnv
from ragen.llm_agent.ctx_manager import ContextManager
from ragen.llm_agent.es_manager import EnvStateManager
from ragen.trainer.generative_critic import FrozenGenerativeCritic


class _TraceTokenizer:
    """Decode one fixed rollout while leaving its original token IDs intact."""

    name_or_path = "qwen-test"
    pad_token_id = 0
    eos_token_id = 0

    def batch_decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return ["Right</answer>"] * len(token_ids)


def _pipeline_config():
    return OmegaConf.create(
        {
            "seed": {"train": 123},
            "enable_response_mask": True,
            "es_manager": {
                "format_penalty": -0.1,
                "train": {
                    "env_groups": 1,
                    "group_size": 1,
                    "env_configs": {
                        "tags": ["CoordSokoban"],
                        "n_groups": [1],
                    },
                },
            },
            "custom_envs": {
                "CoordSokoban": {
                    "env_type": "sokoban",
                    "max_actions_per_traj": 1,
                    "env_instruction": "Solve Sokoban by pushing the box onto the target.",
                    "score_critic_instruction": "Use the exact solver and termination facts.",
                    "max_tokens": 16,
                    "parallel_friendly": False,
                    "max_workers": 1,
                    "env_config": {
                        "dim_x": 6,
                        "dim_y": 6,
                        "num_boxes": 1,
                        "max_steps": 100,
                        "search_depth": 10,
                        "observation_format": "grid_coord",
                    },
                }
            },
            "agent_proxy": {
                "context_window_mode": "full",
                "max_context_window": -1,
                "allow_legacy_retokenization": False,
                "action_sep": "||",
                "max_actions_per_turn": 1,
                "enable_think": False,
                "use_turn_scores": False,
                "reward_normalization": {
                    "grouping": "state",
                    "method": "identity",
                },
            },
            "actor_rollout_ref": {
                "rollout": {
                    "response_length": 8,
                    "max_model_len": 256,
                }
            },
            "generative_critic": {
                "enable": True,
                "backend": "deepseek_api",
                "deepseek_api_key": "test-only-key",
                "deepseek_max_retries": 0,
                "deepseek_max_concurrency": 1,
                "deepseek_cache_enable": False,
                "deepseek_max_tokens": 8,
                "response_format": "score_only",
                "parse_fail_score": 0,
            },
        }
    )


def _install_fixed_room(env: SokobanEnv) -> None:
    """Install a small solvable room where Right is effective but not terminal."""

    env._solution_facts_cache.clear()
    env.room_fixed = np.ones((6, 6), dtype=np.uint8)
    env.room_fixed[[0, -1], :] = 0
    env.room_fixed[:, [0, -1]] = 0
    env.room_fixed[1, 1] = 2
    env.room_state = np.array(env.room_fixed, copy=True)
    env.room_state[3, 3] = 4
    env.room_state[4, 2] = 5
    env.player_position = np.array([4, 2])
    env.num_env_steps = 0
    env.reward_last = 0
    env.boxes_on_target = 0
    env.box_mapping = {}


def test_sokoban_exact_trace_reaches_turn_aligned_critic_label(monkeypatch):
    config = _pipeline_config()
    manager = EnvStateManager(config, mode="train")
    env = manager.envs[0]["env"]
    assert isinstance(env, SokobanEnv)

    def fixed_reset(seed=None, mode=None):
        del seed, mode
        _install_fixed_room(env)
        return env.render()

    monkeypatch.setattr(env, "reset", fixed_reset)
    try:
        initial = manager.reset(seed=123)
        state_before = initial[0]["history"][0]["state"]

        context = ContextManager(config, _TraceTokenizer(), mode="train")
        actor_output = DataProto()
        actor_output.batch = TensorDict(
            {
                "prompts": torch.tensor([[11, 12, 13]], dtype=torch.long),
                "responses": torch.tensor([[21, 22]], dtype=torch.long),
            },
            batch_size=[1],
        )
        actor_output.non_tensor_batch = {"env_ids": np.asarray([0], dtype=int)}

        env_inputs = context.get_env_inputs(actor_output)
        assert env_inputs[0]["actions"] == ["Right"]
        assert env_inputs[0]["prompt_token_ids"] == [11, 12, 13]
        assert env_inputs[0]["response_token_ids"] == [21, 22]

        # The real Sokoban step consumes the sole primitive-action budget.
        assert manager.step(env_inputs) == []
        rollouts = manager.get_rollout_states()
        transition = rollouts[0]["history"][0]["transition_metadata"]
        assert transition["state_before"] == state_before
        assert transition["state_after"] != state_before
        assert transition["action"]["executed_ids"] == [4]
        assert transition["action"]["executed_texts"] == ["Right"]
        assert transition["action"]["effective"] is True
        assert transition["termination"] == {
            "environment_done": False,
            "done": True,
            "terminated": False,
            "truncated": True,
            "success": False,
            "reason": "max_actions",
        }
        assert rollouts[0]["metrics"]["CoordSokoban/action_budget_exhausted"] == 1.0

        actor_batch = context.formulate_rollouts(rollouts)
        assert actor_batch.batch["input_ids"].tolist() == [[11, 12, 13, 21, 22]]
        assert actor_batch.non_tensor_batch["episode_ids"].tolist() == [0]
        assert actor_batch.non_tensor_batch["trajectory_turn_ids"].tolist() == [0]

        turn_ids = actor_batch.batch["turn_ids"]
        actor_mask = turn_ids == 0
        assert actor_mask.sum().item() == 2
        assert torch.equal(actor_batch.batch["loss_mask"].bool(), actor_mask)
        assert actor_batch.batch["turn_end_mask"][actor_mask].tolist() == [0.0, 1.0]

        messages = actor_batch.non_tensor_batch["messages_list"]
        assistant = messages[0][2]
        assert assistant["role"] == "assistant"
        assert assistant["content"] == "Right"
        assert assistant["transition_metadata"] == transition

        captured_prompts = []
        critic = FrozenGenerativeCritic(config)

        def mock_generate(prompts):
            captured_prompts.extend(prompts)
            return ["FINAL_SCORE: -1"] * len(prompts)

        monkeypatch.setattr(critic, "_generate_texts", mock_generate)
        labels, metrics, raw_outputs = critic.infer_turn_labels(messages, turn_ids)

        assert raw_outputs == ["FINAL_SCORE: -1"]
        assert metrics["gen_critic/submitted_prompt_count"] == 1.0
        assert metrics["gen_critic/parse_fail_rate"] == 0.0
        assert labels[actor_mask].tolist() == [-1.0, -1.0]
        assert labels[~actor_mask].tolist() == [0.0, 0.0]
        assert len(captured_prompts) == 1
        assert "[s_t: state before action]" in captured_prompts[0]
        assert "[s_{t+1}: state after action]" in captured_prompts[0]
        assert "termination_reason: max_actions" in captured_prompts[0]
        assert "actions_left_after: 0" in captured_prompts[0]
    finally:
        manager.close()

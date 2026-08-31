import pytest
from types import SimpleNamespace
from omegaconf import OmegaConf
from ragen.llm_agent.es_manager import EnvStateManager, EnvStatus


def make_cfg(env_groups=1):
    return OmegaConf.create({
        'seed': {'train': 7},
        'es_manager': {
            'train': {
                'env_groups': env_groups,
                'group_size': 1,
                'env_configs': {'tags': ['Bandit'], 'n_groups': [env_groups]},
            }
        },
        'custom_envs': {
            'Bandit': {
                'env_type': 'bandit',
                'max_actions_per_traj': 1,
                'env_config': None
            }
        }
    })


def test_seed_iteration():
    cfg = make_cfg()
    es = EnvStateManager(cfg, mode='train')
    es.reset()
    first_seed = es.envs[0]['status'].seed
    es.reset()
    second_seed = es.envs[0]['status'].seed
    assert first_seed == 7
    assert second_seed == 8


def test_explicit_train_seed_advances_past_every_consumed_group():
    es = EnvStateManager(make_cfg(env_groups=2), mode='train')

    es.reset(seed=20)
    assert [entry['status'].seed for entry in es.envs] == [20, 21]
    es.reset()
    assert [entry['status'].seed for entry in es.envs] == [22, 23]


def test_finalize_unfinished_marks_only_active_episodes_as_truncated():
    manager = object.__new__(EnvStateManager)
    active = EnvStatus(num_actions=2)
    succeeded = EnvStatus(terminated=True, truncated=False, num_actions=1)
    already_truncated = EnvStatus(terminated=True, truncated=True, num_actions=3)
    manager.envs = [
        {'status': active},
        {'status': succeeded},
        {'status': already_truncated},
    ]
    manager.rollout_cache = [
        {'history': [{}, {}, {}]},
        {'history': [{}, {}]},
        {'history': [{}, {}, {}, {}], 'termination_reason': 'environment'},
    ]

    manager.finalize_unfinished(reason='max_turn')

    assert active.terminated is True
    assert active.truncated is True
    assert manager.rollout_cache[0]['termination_reason'] == 'max_turn'
    assert len(manager.rollout_cache[0]['history']) == 3
    assert succeeded.terminated is True and succeeded.truncated is False
    assert 'termination_reason' not in manager.rollout_cache[1]
    assert manager.rollout_cache[2]['termination_reason'] == 'environment'


def test_step_records_executed_action_and_forces_protocol_violation_negative():
    class FakeEnv:
        config = SimpleNamespace(action_lookup={0: "Left", 1: "Right"})

        def step(self, action):
            assert action == 0
            return None, 0.0, False, {"raw_reward": 0.0}

        def render(self):
            return "next state"

    manager = object.__new__(EnvStateManager)
    manager.sys_config = SimpleNamespace(
        es_manager=SimpleNamespace(format_penalty=-0.1)
    )
    manager._executors = {}
    manager.envs = [
        {
            "env_id": 0,
            "group_id": 0,
            "tag": "Puzzle",
            "env": FakeEnv(),
            "status": EnvStatus(seed=1),
            "max_actions_per_traj": 3,
            "parallel_friendly": False,
            "max_workers": 1,
        }
    ]
    manager.rollout_cache = [
        {
            "env_id": 0,
            "group_id": 0,
            "tag": "Puzzle",
            "history": [{"state": "start", "actions_left": 3}],
            "penalty": 0.0,
        }
    ]

    manager.step(
        [
            {
                "env_id": 0,
                "llm_raw_response": "<answer>Left | Right</answer>",
                "llm_response": "<answer>Left</answer>",
                "actions": ["Left"],
                "response_format_valid": True,
                "action_count_exceeded": True,
            }
        ]
    )

    turn = manager.rollout_cache[0]["history"][0]
    assert turn["actions"] == [0]
    assert turn["executed_action_texts"] == ["Left"]
    assert turn["action_count_exceeded"] is True
    assert turn["judge_force_negative"] is True
    assert turn["judge_force_reason"] == "max_actions_per_turn_exceeded"

import pytest
import numpy as np
from types import SimpleNamespace
from omegaconf import OmegaConf
from ragen.llm_agent.es_manager import EnvStateManager, EnvStatus
from ragen.env.sokoban.config import SokobanEnvConfig
from ragen.env.sokoban.env import SokobanEnv
from ragen.env.sokoban.utils import get_shortest_solution_length


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
    already_truncated = EnvStatus(terminated=False, truncated=True, num_actions=3)
    manager.envs = [
        {'status': active},
        {'status': succeeded},
        {'status': already_truncated},
    ]
    manager.rollout_cache = [
        {'history': [{}, {}, {}]},
        {'history': [{}, {}]},
        {'history': [{}, {}, {}, {}], 'termination_reason': 'environment_failure'},
    ]

    manager.finalize_unfinished(reason='max_turn')

    assert active.terminated is False
    assert active.truncated is True
    assert manager.rollout_cache[0]['termination_reason'] == 'max_turn'
    assert len(manager.rollout_cache[0]['history']) == 3
    assert succeeded.terminated is True and succeeded.truncated is False
    assert 'termination_reason' not in manager.rollout_cache[1]
    assert manager.rollout_cache[2]['termination_reason'] == 'environment_failure'


def test_action_budget_exhaustion_is_recorded_and_not_terminated():
    class FakeEnv:
        config = SimpleNamespace(action_lookup={0: "Left"})

        def step(self, action):
            assert action == 0
            return None, 0.0, False, {"raw_reward": 0.0}

        def render(self):
            return "next state"

    manager = object.__new__(EnvStateManager)
    manager.sys_config = SimpleNamespace(es_manager=SimpleNamespace(format_penalty=-0.1))
    manager._executors = {}
    manager.group_size = 1
    manager.envs = [
        {
            "env_id": 0,
            "group_id": 0,
            "tag": "Puzzle",
            "env": FakeEnv(),
            "status": EnvStatus(seed=1),
            "max_actions_per_traj": 1,
            "parallel_friendly": False,
            "max_workers": 1,
        }
    ]
    manager.rollout_cache = [
        {
            "env_id": 0,
            "group_id": 0,
            "tag": "Puzzle",
            "history": [{"state": "start", "actions_left": 1}],
            "penalty": 0.0,
        }
    ]

    active_outputs = manager.step(
        [
            {
                "env_id": 0,
                "llm_raw_response": "<answer>Left</answer>",
                "llm_response": "<answer>Left</answer>",
                "actions": ["Left"],
                "response_format_valid": True,
                "action_format_valid": True,
                "action_count_exceeded": False,
            }
        ]
    )

    assert active_outputs == []
    assert manager.envs[0]["status"].terminated is False
    assert manager.envs[0]["status"].truncated is True
    assert manager.rollout_cache[0]["termination_reason"] == "max_actions"

    state = manager.get_rollout_states()[0]
    metrics = state["metrics"]
    assert metrics["Puzzle/trajectory_done"] == 1.0
    assert metrics["Puzzle/trajectory_terminated"] == 0.0
    assert metrics["Puzzle/trajectory_truncated"] == 1.0
    assert metrics["Puzzle/action_budget_exhausted"] == 1.0
    assert metrics["Puzzle/turn_budget_exhausted"] == 0.0
    assert metrics["Puzzle/rollout_budget_exhausted"] == 1.0


@pytest.mark.parametrize(
    ("success_value", "expected_terminated", "expected_reason"),
    [
        (True, True, "environment_success"),
        (False, False, "environment_failure"),
        ("False", False, "environment_failure"),
    ],
)
def test_environment_completion_has_strict_success_and_precedes_action_budget(
    success_value,
    expected_terminated,
    expected_reason,
):
    class FakeEnv:
        config = SimpleNamespace(action_lookup={0: "Left"})

        def step(self, action):
            return None, 1.0, True, {"success": success_value, "raw_reward": 1.0}

        def render(self):
            return "done"

    manager = object.__new__(EnvStateManager)
    manager.sys_config = SimpleNamespace(es_manager=SimpleNamespace(format_penalty=-0.1))
    manager._executors = {}
    manager.group_size = 1
    manager.envs = [
        {
            "env_id": 0,
            "group_id": 0,
            "tag": "Puzzle",
            "env": FakeEnv(),
            "status": EnvStatus(seed=1),
            "max_actions_per_traj": 1,
            "parallel_friendly": False,
            "max_workers": 1,
        }
    ]
    manager.rollout_cache = [
        {
            "env_id": 0,
            "group_id": 0,
            "tag": "Puzzle",
            "history": [{"state": "start", "actions_left": 1}],
            "penalty": 0.0,
        }
    ]

    manager.step(
        [
            {
                "env_id": 0,
                "llm_raw_response": "<answer>Left</answer>",
                "llm_response": "<answer>Left</answer>",
                "actions": ["Left"],
                "response_format_valid": True,
                "action_format_valid": True,
                "action_count_exceeded": False,
            }
        ]
    )

    status = manager.envs[0]["status"]
    assert status.terminated is expected_terminated
    assert status.truncated is (not expected_terminated)
    assert manager.rollout_cache[0]["termination_reason"] == expected_reason
    metrics = manager.get_rollout_states()[0]["metrics"]
    assert metrics["Puzzle/action_budget_exhausted"] == 0.0


def test_step_rejects_duplicate_and_completed_environment_ids():
    manager = object.__new__(EnvStateManager)
    manager.envs = [{"status": EnvStatus()}]
    duplicate = [{"env_id": 0}, {"env_id": 0}]

    with pytest.raises(ValueError, match="duplicate env_id"):
        manager.step(duplicate)

    manager.envs[0]["status"] = EnvStatus(truncated=True)
    with pytest.raises(RuntimeError, match="completed env_id"):
        manager.step([{"env_id": 0}])


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


def test_step_records_structured_transition_history_and_finalizes_termination():
    class FakeEnv:
        config = SimpleNamespace(action_lookup={0: "Left", 1: "Right"})

        def __init__(self):
            self.state = "A"

        def step(self, action):
            self.state = "B" if action == 1 else "A"
            return None, -0.1, False, {
                "raw_reward": -0.1,
                "action_is_valid": True,
                "action_is_effective": True,
                "action_is_blocked": False,
                "action.moved_player": True,
                "action.moved_box": False,
            }

        def render(self):
            return self.state

    manager = object.__new__(EnvStateManager)
    manager.sys_config = SimpleNamespace(
        es_manager=SimpleNamespace(format_penalty=-0.1),
        agent_proxy=SimpleNamespace(max_turn=3),
    )
    manager._executors = {}
    manager.envs = [{
        "env_id": 0,
        "group_id": 0,
        "tag": "Puzzle",
        "env": FakeEnv(),
        "status": EnvStatus(seed=1),
        "max_actions_per_traj": 3,
        "parallel_friendly": False,
        "max_workers": 1,
    }]
    manager.rollout_cache = [{
        "env_id": 0,
        "group_id": 0,
        "tag": "Puzzle",
        "history": [{"state": "A", "actions_left": 3}],
        "penalty": 0.0,
    }]

    def action_input(action):
        return {
            "env_id": 0,
            "llm_raw_response": f"<answer>{action}</answer>",
            "llm_response": f"<answer>{action}</answer>",
            "actions": [action],
            "response_format_valid": True,
            "action_format_valid": True,
            "action_count_exceeded": False,
        }

    manager.step([action_input("Right")])
    manager.step([action_input("Left")])

    first = manager.rollout_cache[0]["history"][0]["transition_metadata"]
    second = manager.rollout_cache[0]["history"][1]["transition_metadata"]
    assert first["state_before"] == "A" and first["state_after"] == "B"
    assert first["actions_left_before"] == 3 and first["actions_left_after"] == 2
    assert first["turn_number"] == 1 and first["turns_left"] == 2
    assert first["previous_action"] == []
    assert first["is_cycle"] is False
    assert second["previous_action"] == "Right"
    assert second["is_inverse"] is True
    assert second["is_cycle"] is True
    assert second["history"]["cycle_length"] == 2
    assert second["state_seen_count"] == 1
    assert second["action_is_effective"] is True
    assert second["moved_box"] is False

    manager.finalize_unfinished(reason="max_turn")
    assert second["truncated"] is True
    assert second["termination_reason"] == "max_turn"
    assert second["termination"]["done"] is True


def test_multi_action_turn_aggregates_transition_facts_across_primitives():
    class FakeEnv:
        config = SimpleNamespace(action_lookup={0: "Left", 1: "Right"})

        def __init__(self):
            self.state = "S0"
            self.calls = 0

        def step(self, action):
            self.calls += 1
            self.state = f"S{self.calls}"
            info = {
                "raw_reward": -0.1,
                "action_is_valid": self.calls == 1,
                "action_is_effective": self.calls == 1,
                "action_is_blocked": self.calls == 2,
                "action.moved_player": self.calls == 1,
                "action.moved_box": self.calls == 1,
                "shortest_solution_length_before": 5 - (self.calls - 1),
                "shortest_solution_length_after": 5 - self.calls,
                "deadlock_before": False,
                "deadlock_after": self.calls == 2,
                "solver_status_before": "solvable",
                "solver_status_after": "deadlock" if self.calls == 2 else "solvable",
                "success": False,
            }
            return None, -0.1, False, info

        def render(self):
            return self.state

    manager = object.__new__(EnvStateManager)
    manager.sys_config = SimpleNamespace(
        es_manager=SimpleNamespace(format_penalty=-0.1),
        agent_proxy=SimpleNamespace(max_turn=2),
    )
    manager._executors = {}
    manager.envs = [{
        "env_id": 0,
        "group_id": 0,
        "tag": "Puzzle",
        "env": FakeEnv(),
        "status": EnvStatus(seed=1),
        "max_actions_per_traj": 4,
        "parallel_friendly": False,
        "max_workers": 1,
    }]
    manager.rollout_cache = [{
        "env_id": 0,
        "group_id": 0,
        "tag": "Puzzle",
        "history": [{"state": "S0", "actions_left": 4}],
        "penalty": 0.0,
    }]

    manager.step([{
        "env_id": 0,
        "llm_raw_response": "<answer>Right</answer><answer>Left</answer>",
        "llm_response": "<answer>Right</answer><answer>Left</answer>",
        "actions": ["Right", "Left"],
        "response_format_valid": True,
        "action_format_valid": True,
        "action_count_exceeded": False,
    }])

    metadata = manager.rollout_cache[0]["history"][0]["transition_metadata"]
    assert metadata["state_before"] == "S0"
    assert metadata["state_after"] == "S2"
    assert metadata["shortest_solution_length_before"] == 5
    assert metadata["shortest_solution_length_after"] == 3
    assert metadata["deadlock_before"] is False
    assert metadata["deadlock_after"] is True
    assert metadata["environment_info"]["solver_status_before"] == "solvable"
    assert metadata["environment_info"]["solver_status_after"] == "deadlock"
    assert metadata["action_is_effective"] is True
    assert metadata["action_is_valid"] is False
    assert metadata["moved_player"] is True
    assert metadata["moved_box"] is True
    assert metadata["action"]["blocked"] is True
    assert metadata["observed_reward"] == pytest.approx(-0.2)


def test_sokoban_solver_distance_and_blocked_action_facts():
    config = SokobanEnvConfig(dim_room=(5, 5), num_boxes=1, max_steps=20)
    env = SokobanEnv(config)
    env.room_fixed = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 2, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ])
    env.room_state = np.array([
        [0, 0, 0, 0, 0],
        [0, 5, 1, 2, 0],
        [0, 1, 4, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ])
    env.player_position = np.array([1, 1])
    env.num_env_steps = 0
    env.reward_last = 0
    env.boxes_on_target = 0
    env._solution_facts_cache.clear()

    before_distance = get_shortest_solution_length(
        env.room_fixed, env.room_state, max_depth=100
    )
    _, _, _, info = env.step(1)  # Up is mapped but blocked by the wall.

    assert before_distance is not None and before_distance > 0
    assert info["action_is_mapped"] is True
    assert info["action_is_valid"] is False
    assert info["action_is_effective"] is False
    assert info["action_is_blocked"] is True
    assert info["shortest_solution_length_before"] == before_distance
    assert info["shortest_solution_length_after"] == before_distance

    solved_state = env.room_state.copy()
    solved_state[2, 2] = env.room_fixed[2, 2]
    solved_state[1, 3] = 3
    assert get_shortest_solution_length(env.room_fixed, solved_state) == 0

    deadlocked_state = env.room_state.copy()
    deadlocked_state[2, 2] = env.room_fixed[2, 2]
    deadlocked_state[1, 1] = 4
    deadlocked_state[2, 1] = 5
    assert get_shortest_solution_length(env.room_fixed, deadlocked_state) is None


def test_rollout_metrics_ignore_categorical_info_without_empty_metric():
    manager = object.__new__(EnvStateManager)
    manager.group_size = 1
    manager.envs = [
        {
            "tag": "Puzzle",
            "group_id": 0,
            "env": SimpleNamespace(),
            "status": EnvStatus(terminated=True, truncated=False, num_actions=1),
        }
    ]
    manager.rollout_cache = [
        {
            "tag": "Puzzle",
            "group_id": 0,
            "history": [
                {"info": {"action.name": "push left", "raw_reward": 1.0}},
                {"state": "done"},
            ],
        }
    ]

    states = manager.get_rollout_states()

    assert "Puzzle/action.name" not in states[0]["metrics"]
    assert states[0]["metrics"]["Puzzle/raw_reward"] == pytest.approx(1.0)
    assert states[0]["metrics"]["Puzzle/episodic_return"] == pytest.approx(1.0)
    assert states[0]["history"][-1]["metrics"] == {"raw_reward": [1.0]}

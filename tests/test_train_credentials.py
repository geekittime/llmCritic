import os

import pytest
from omegaconf import OmegaConf

from train import (
    _load_task_credentials,
    _validate_credential_file,
    add_dependency_and_validate_config,
)


def _write_credentials(path, text):
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)
    return str(path)


def test_task_credential_file_overrides_stale_inherited_values(tmp_path, monkeypatch):
    credential_path = _write_credentials(
        tmp_path / "secrets.env",
        "DEEPSEEK_API_KEY='rotated-key'\nWANDB_API_KEY=wandb-key\n",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "revoked-key")
    monkeypatch.setenv("WANDB_API_KEY", "stale-wandb-key")

    _load_task_credentials(credential_path)

    assert os.environ["DEEPSEEK_API_KEY"] == "rotated-key"
    assert os.environ["WANDB_API_KEY"] == "wandb-key"


def test_credential_file_rejects_group_or_other_access(tmp_path):
    credential_path = tmp_path / "secrets.env"
    credential_path.write_text("DEEPSEEK_API_KEY=test-key\n", encoding="utf-8")
    credential_path.chmod(0o640)

    with pytest.raises(PermissionError, match="mode 600"):
        _validate_credential_file(credential_path)


def test_task_credential_file_rejects_non_allowlisted_assignments(tmp_path):
    credential_path = _write_credentials(
        tmp_path / "secrets.env",
        "DEEPSEEK_API_KEY=test-key\nCUDA_DEVICES=7\n",
    )

    with pytest.raises(ValueError, match="Unsupported credential assignment"):
        _load_task_credentials(credential_path)


def _label_only_config():
    return OmegaConf.create(
        {
            "micro_batch_size_per_gpu": 2,
            "ppo_mini_batch_size": 32,
            "model_path": "/models/Qwen2.5-3B-Instruct",
            "enable_response_mask": True,
            "system": {"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            "trainer": {"n_gpus_per_node": 4},
            "actor_rollout_ref": {
                "actor": {
                    "ppo_mini_batch_size": 32,
                    "ulysses_sequence_parallel_size": 1,
                    "entropy_coeff": 0.0,
                    "use_kl_loss": False,
                },
                "rollout": {"rollout_filter_ratio": 1.0},
            },
            "agent_proxy": {
                "context_window_mode": "full",
                "max_turn": 5,
                "use_turn_scores": False,
            },
            "es_manager": {"train": {"env_groups": 4, "group_size": 8}},
            "algorithm": {
                "bi_level_gae": False,
                "adv_estimator": "gae",
                "use_label_outcome_advantage": True,
                "turn_advantage_mode": "label_only",
                "turn_score_reduction": "mean",
                "normalize_turn_advantage": False,
                "use_kl_in_reward": False,
                "add_kl_to_turn_advantage": False,
            },
            "critic": {"enable": False},
            "generative_critic": {
                "enable": True,
                "backend": "deepseek_api",
                "deepseek_api_key": "test-key",
                "deepseek_api_key_env": "DEEPSEEK_API_KEY",
            },
            "data": {},
        }
    )


def test_label_only_config_accepts_exact_unregularized_judge_advantage():
    config = add_dependency_and_validate_config(_label_only_config())

    assert config.data.train_batch_size == 32


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"algorithm.normalize_turn_advantage": True}, "normalize"),
        (
            {
                "algorithm.use_kl_in_reward": True,
                "algorithm.add_kl_to_turn_advantage": True,
            },
            "KL reward",
        ),
        ({"generative_critic.enable": False}, "judge"),
        ({"algorithm.turn_score_reduction": "sum"}, "reduction"),
    ],
)
def test_label_only_config_rejects_advantage_rewrites(updates, message):
    config = _label_only_config()
    for path, value in updates.items():
        OmegaConf.update(config, path, value)

    with pytest.raises(ValueError, match=message):
        add_dependency_and_validate_config(config)

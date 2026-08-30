import os

import pytest

from train import _load_task_credentials, _validate_credential_file


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

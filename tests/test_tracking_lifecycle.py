import sys
from types import MethodType, SimpleNamespace

import pytest

from ragen.tracking import Tracking, VerlTracking
from ragen.trainer.agent_trainer import RayAgentTrainer


class _FakeBackend:
    def __init__(self):
        self.exit_codes = []

    def finish(self, exit_code=0):
        self.exit_codes.append(exit_code)


class _FlakyBackend(_FakeBackend):
    def finish(self, exit_code=0):
        super().finish(exit_code=exit_code)
        if len(self.exit_codes) == 1:
            raise RuntimeError("temporary transport error")


def test_tracking_finish_is_explicit_and_idempotent():
    backend = _FakeBackend()
    tracker = object.__new__(Tracking)
    tracker._finished = False
    tracker.logger = {"wandb": backend, "console": object()}

    tracker.finish(exit_code=7)
    tracker.finish(exit_code=9)
    tracker.__del__()

    assert backend.exit_codes == [7]
    assert "wandb" not in tracker.logger
    assert "console" in tracker.logger


def test_tracking_finish_can_retry_only_the_failed_backend():
    backend = _FlakyBackend()
    tracker = object.__new__(Tracking)
    tracker._finished = False
    tracker.logger = {"wandb": backend}

    with pytest.raises(RuntimeError, match="temporary transport error"):
        tracker.finish(exit_code=1)

    assert tracker._finished is False
    assert tracker.logger["wandb"] is backend

    tracker.finish(exit_code=1)

    assert backend.exit_codes == [1, 1]
    assert tracker._finished is True
    assert "wandb" not in tracker.logger


def test_failed_finish_destructor_retry_preserves_failure_exit_code():
    backend = _FlakyBackend()
    tracker = object.__new__(Tracking)
    tracker._finished = False
    tracker._destructor_exit_code = 0
    tracker.logger = {"wandb": backend}

    with pytest.raises(RuntimeError, match="temporary transport error"):
        tracker.finish(exit_code=1)
    tracker.__del__()

    assert backend.exit_codes == [1, 1]
    assert tracker._finished is True


def test_tracking_initialization_failure_marks_started_backend_failed(monkeypatch):
    backend = _FakeBackend()

    def failing_init(self, *args, **kwargs):
        self.logger = {"wandb": backend}
        raise RuntimeError("secondary backend init failed")

    monkeypatch.setattr(VerlTracking, "__init__", failing_init)

    with pytest.raises(RuntimeError, match="secondary backend init failed"):
        Tracking("project", "run")

    assert backend.exit_codes == [1]


@pytest.mark.parametrize(("exit_code", "status"), [(0, "FINISHED"), (1, "FAILED")])
def test_tracking_explicitly_finishes_mlflow(monkeypatch, exit_code, status):
    statuses = []
    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        SimpleNamespace(end_run=lambda status: statuses.append(status)),
    )
    tracker = object.__new__(Tracking)
    tracker._finished = False
    tracker.logger = {"mlflow": object()}

    tracker.finish(exit_code=exit_code)

    assert statuses == [status]
    assert "mlflow" not in tracker.logger


def test_agent_trainer_finishes_tracking_after_resource_shutdown():
    events = []
    tracker = SimpleNamespace(finish=lambda exit_code: events.append(("finish", exit_code)))
    trainer = object.__new__(RayAgentTrainer)

    def fake_fit_impl(self):
        events.append(("fit", None))
        self._tracking_logger = tracker
        return "done"

    trainer._fit_impl = MethodType(fake_fit_impl, trainer)
    trainer._shutdown_frozen_critic = MethodType(
        lambda self: events.append(("shutdown", None)), trainer
    )

    assert RayAgentTrainer.fit(trainer) == "done"
    assert events == [("fit", None), ("shutdown", None), ("finish", 0)]


def test_agent_trainer_marks_failed_run_and_preserves_training_error():
    events = []
    tracker = SimpleNamespace(finish=lambda exit_code: events.append(("finish", exit_code)))
    trainer = object.__new__(RayAgentTrainer)

    def failing_fit_impl(self):
        self._tracking_logger = tracker
        raise ValueError("training failed")

    trainer._fit_impl = MethodType(failing_fit_impl, trainer)
    trainer._shutdown_frozen_critic = MethodType(
        lambda self: events.append(("shutdown", None)), trainer
    )

    with pytest.raises(ValueError, match="training failed"):
        RayAgentTrainer.fit(trainer)

    assert events == [("shutdown", None), ("finish", 1)]


def test_resource_shutdown_does_not_mask_training_error(capsys):
    events = []
    tracker = SimpleNamespace(finish=lambda exit_code: events.append(("finish", exit_code)))
    trainer = object.__new__(RayAgentTrainer)

    def failing_fit_impl(self):
        self._tracking_logger = tracker
        raise ValueError("primary training error")

    def failing_shutdown(self):
        events.append(("shutdown", None))
        raise RuntimeError("secondary shutdown error")

    trainer._fit_impl = MethodType(failing_fit_impl, trainer)
    trainer._shutdown_frozen_critic = MethodType(failing_shutdown, trainer)

    with pytest.raises(ValueError, match="primary training error"):
        RayAgentTrainer.fit(trainer)

    assert events == [("shutdown", None), ("finish", 1)]
    assert "secondary shutdown error" in capsys.readouterr().out


def test_resource_shutdown_failure_marks_otherwise_successful_run_failed():
    events = []
    tracker = SimpleNamespace(finish=lambda exit_code: events.append(("finish", exit_code)))
    trainer = object.__new__(RayAgentTrainer)

    def fake_fit_impl(self):
        self._tracking_logger = tracker
        return "unreachable"

    def failing_shutdown(self):
        events.append(("shutdown", None))
        raise RuntimeError("shutdown failed")

    trainer._fit_impl = MethodType(fake_fit_impl, trainer)
    trainer._shutdown_frozen_critic = MethodType(failing_shutdown, trainer)

    with pytest.raises(RuntimeError, match="shutdown failed"):
        RayAgentTrainer.fit(trainer)

    assert events == [("shutdown", None), ("finish", 1)]

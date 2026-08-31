"""RAGEN tracking lifecycle helpers.

VERL's tracker only finalizes backends from ``__del__``.  At interpreter
shutdown that can run after W&B has already closed its service transport, so
the final history row is never flushed.  This wrapper adds an idempotent,
explicit ``finish`` method while retaining the same public logging API.
"""

from __future__ import annotations

from typing import Any

from verl.utils.tracking import Tracking as VerlTracking


class Tracking(VerlTracking):
    """VERL tracker with deterministic, idempotent backend finalization."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._finished = False
        self._destructor_exit_code = 0
        self.logger = {}
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            self._destructor_exit_code = 1
            try:
                self.finish(exit_code=1)
            except Exception:
                # Preserve the backend initialization error. Any backend
                # that failed to finish remains available for __del__ retry.
                pass
            raise

    def finish(self, exit_code: int = 0) -> None:
        if self._finished:
            return
        if exit_code != 0 and getattr(self, "_destructor_exit_code", 0) == 0:
            # A failed run must stay failed if explicit finalization raises
            # and __del__ performs the best-effort retry.
            self._destructor_exit_code = exit_code

        errors: list[tuple[str, Exception]] = []
        exit_code_backends = {"wandb", "vemlp_wandb"}
        mlflow_backends = {"mlflow"}
        no_arg_backends = {
            "swanlab",
            "tensorboard",
            "clearml",
            "trackio",
            "file",
        }
        loggers = getattr(self, "logger", {})
        for backend in exit_code_backends | mlflow_backends | no_arg_backends:
            logger = loggers.get(backend)
            if logger is None:
                continue
            try:
                if backend in exit_code_backends:
                    logger.finish(exit_code=exit_code)
                elif backend in mlflow_backends:
                    import mlflow

                    mlflow.end_run(status="FINISHED" if exit_code == 0 else "FAILED")
                else:
                    logger.finish()
            except Exception as exc:  # Finalize the remaining backends first.
                errors.append((backend, exc))
            else:
                loggers.pop(backend, None)

        if errors:
            details = "; ".join(f"{backend}: {exc}" for backend, exc in errors)
            raise RuntimeError(f"Failed to finalize tracking backend(s): {details}") from errors[0][1]
        self._finished = True

    def __del__(self) -> None:
        try:
            self.finish(exit_code=getattr(self, "_destructor_exit_code", 1))
        except Exception:
            # Destructors must not raise during interpreter shutdown.  Normal
            # trainer exits call finish explicitly and report any failure.
            pass

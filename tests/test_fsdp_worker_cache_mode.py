import inspect
from types import SimpleNamespace

from verl import DataProto

from ragen.workers.fsdp_workers import ActorRolloutRefWorker


def test_skipped_multiturn_end_restores_trainer_mode(monkeypatch):
    worker = object.__new__(ActorRolloutRefWorker)
    worker._is_rollout = True
    worker._is_actor = True
    calls = []

    async def trainer_mode():
        calls.append("trainer_mode")

    def unexpected_generation(*args, **kwargs):
        raise AssertionError("skip_generation must not invoke the rollout engine")

    worker.trainer_mode = trainer_mode
    worker.rollout = SimpleNamespace(generate_sequences=unexpected_generation)
    prompts = DataProto(meta_info={"mode": "multiturn-end", "skip_generation": True})

    monkeypatch.setattr("ragen.workers.fsdp_workers.log_gpu_memory_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr("ragen.workers.fsdp_workers.torch.cuda.empty_cache", lambda: None)

    generate_sequences = inspect.unwrap(ActorRolloutRefWorker.generate_sequences)
    output = generate_sequences(worker, prompts)

    assert output.meta_info == prompts.meta_info
    assert calls == ["trainer_mode"]

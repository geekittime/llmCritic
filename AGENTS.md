# AGENTS Guide for GenPPO (RAGEN)
Guide for coding agents in `/home/kangshijia/GenPPO`.

## 1) Repository Overview
- Core package: `ragen/`
- Main tests: `tests/` and `test_sudoku.py`
- Train entrypoint: `train.py`
- Eval entrypoint: `python -m ragen.llm_agent.agent_proxy`
- Configs: `config/*.yaml` (Hydra overrides are standard)
- Setup scripts: `scripts/setup_ragen.sh`, `scripts/setup_webshop.sh`
- Submodules: `verl/`, `external/webshop-minimal/`, `external/kimina-lean-server/`, `ragen/env/spatial/Base/`

## 2) Cursor / Copilot Rules
Checked rule paths:
- `.cursor/rules/`
- `.cursorrules`
- `.github/copilot-instructions.md`

Current status: no Cursor/Copilot instruction files found.
If added later, treat them as higher-priority constraints.

## 3) Build and Setup Commands
Recommended full setup:
```bash
bash scripts/setup_ragen.sh
```

Optional WebShop setup:
```bash
bash scripts/setup_webshop.sh
```

Manual editable setup:
```bash
pip install -U pip setuptools wheel
pip install -e ./verl --no-dependencies
pip install -e .
pip install -r requirements.txt
```

Optional extras:
```bash
pip install -e ".[webshop]"
pip install -e ".[lean]"
pip install -e ".[all]"
```

Quick syntax/import check:
```bash
python -m compileall ragen tests
```

## 4) Lint / Static Checks
No enforced root config for `ruff`, `black`, `isort`, `mypy`, or `pyright`.

Recommended baseline checks:
```bash
python -m compileall ragen tests
pytest -q
```

Notes:
- Avoid mass reformatting unrelated files.
- If editing `verl/`, use `verl/.pre-commit-config.yaml` for submodule checks.

## 5) Test Commands (Single Test Emphasis)
Run all tests:
```bash
pytest
```

Quiet run:
```bash
pytest -q
```

Run one file:
```bash
pytest tests/llm_agent/test_context_window.py -q
```

Run one function:
```bash
pytest tests/llm_agent/test_context_window.py::test_context_window_truncation -q
```

Another single-test example:
```bash
pytest tests/test_rollout_filter.py::test_reward_metric_selects_high_mean_group -q
```

Filter by keyword:
```bash
pytest -k context_window -q
```

Run top-level single file:
```bash
pytest test_sudoku.py -q
```

Fast-fail loop:
```bash
pytest --maxfail=1 -x -q
```

## 6) Training / Evaluation Commands
Base training:
```bash
python train.py --config-name base
```

Common env configs:
```bash
python train.py --config-name _2_sokoban
python train.py --config-name _3_frozen_lake
```

Hydra override pattern:
```bash
python train.py --config-name _2_sokoban trainer.experiment_name=my_exp system.CUDA_VISIBLE_DEVICES="0"
```

Evaluation command:
```bash
python -m ragen.llm_agent.agent_proxy --config-name eval
```

## 7) Code Style Guidelines
Follow file-local style first; use these defaults for new edits.

### Imports
- Prefer absolute imports rooted at `ragen`.
- Group imports: stdlib, third-party, local.
- Prefer one import per line unless grouping helps readability.

### Formatting
- Follow PEP 8.
- Keep lines readable; avoid giant chains.
- Avoid formatting-only diffs unless requested.

### Types
- Add type hints for new public functions/non-trivial helpers.
- Prefer explicit return types in trainer/rollout/env core paths.
- Use practical types (`Optional[T]`, `Dict[str, Any]`) when strict typing is unstable.

### Naming
- `snake_case`: vars/functions/modules.
- `PascalCase`: classes.
- `UPPER_SNAKE_CASE`: constants.
- Prefer descriptive names over abbreviations.

### Error Handling
- Validate config and shape assumptions early.
- Use clear assertion messages for invariants.
- Raise explicit exceptions for file/config/runtime failures.

### Logging
- Prefer structured logging in reusable modules.
- `print` is acceptable in scripts and debug-heavy entrypoints.
- Avoid noisy logs in hot loops unless debug-gated.

### Hydra / Config Conventions
- Add new knobs under `config/*.yaml`.
- Reuse namespaces like `agent_proxy.*`, `es_manager.*`, `actor_rollout_ref.*`.
- Keep defaults backward compatible.

### Testing Conventions
- Use `pytest` with `test_*` names.
- Keep tests deterministic with explicit seeds when possible.
- Add focused unit tests near impacted modules.

### Performance / Concurrency
- Be careful in rollout and worker hot paths.
- Preserve batch semantics and tensor shape invariants.
- Document determinism trade-offs when introducing parallelism.

### Security
- Never commit API keys, tokens, or credentials.
- Replace hardcoded secrets with env vars/config values.

## 8) Agent Checklist Before Handoff
1. Run relevant single tests first.
2. Run broader `pytest -q` when scope is medium/large.
3. Verify config compatibility and defaults.
4. Keep diffs minimal; avoid unrelated edits.
5. Summarize behavior changes and test evidence clearly.

## 9) Ongoing Plan: Trainable Generative Critic (RLVR)

This section records the active implementation plan and progress for converting
the current frozen generative critic into a trainable critic.

### Goal
- Keep critic independent from actor (separate model/optimizer/checkpoint path).
- Critic predicts per-turn label via generation (`###label: True/False`).
- Train critic with RLVR-style reward:
  - trajectory success -> all turn labels target `True`
  - trajectory failure -> all turn labels target `False`
- Reward validates label correctness; policy update applies to full generated output.

### Planned File-by-File Changes (Step 2)
1. `config/base.yaml`
   - Add generative critic training switches and reward weights.
   - Add rollout/sampling knobs for critic train path.

2. `ragen/trainer/generative_critic.py`
   - Add helper methods to construct train samples for critic.
   - Add label-target extraction from trajectory-level success/failure.
   - Add RLVR reward shaping utilities (format + label correctness).

3. `ragen/workers/fsdp_workers.py`
   - Add a trainable generative critic worker reusing actor PPO update structure.
   - Expose methods for critic generation, logprob recompute, and policy update.

4. `train.py`
   - Route `Role.Critic` to generative critic worker when enabled.
   - Keep legacy value critic route as fallback.

5. `ragen/trainer/agent_trainer.py`
   - Integrate critic rollout and RLVR reward assignment.
   - Build critic mini-batches and call critic policy update.
   - Log critic-specific metrics (`label_acc`, `format_rate`, RLVR reward stats).

### Current Progress Log
- Completed: frozen generative critic inference path.
- Completed: actor training signal path using `label + outcome` rewards.
- Completed: vLLM reuse path for critic inference and chat-template alignment.
- Completed: trainable generative critic worker path and RLVR training loop integration.
- Completed: trajectory success direct field (`trajectory_success`) from env metrics to trainer.
- In progress: stabilization and metric-driven validation for trainable generative critic.

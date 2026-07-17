# LLM L2 Patch Fix Report

**Branch:** `feature/tri-level-autoresearch`  
**Date:** 2026-07-17  
**Scope:** Schedule L2 (ouroboros) + Runner L2 (gpu_bench ablation)

## Root causes

### 1. Schedule L2 (ouroboros) — codegen shape mismatch

DeepSeek often emitted full helper classes (e.g. `ValueCascadeAnalyzer`) alongside or instead of a bare `decide()` method. Schema checks rejected some outputs; attempt 2 passed schema but **validate failed** because generated code referenced nonexistent `self.min_discard_rate_threshold` and used bare attribute access on `l2_sessions` items (`s.applied`, `s.blocked_by_tabu`).

### 2. Misleading failure labels

Controllers and tabu registry labeled **all** validate failures as `import_fail`, even when the actual error was `AttributeError`, schema violation, or patch apply error. Session `06_summary.json` did not record the real `validation_error`.

### 3. Runner L2 (gpu_bench) — broken validate path

`GpuBenchMechanismResearcher.validate()` used a subprocess import of a path derived from `run_dir` copy layout, which failed even when patches were syntactically valid. The tri-level controller duplicated validation via `_validate_runner_module` instead of the researcher.

### 4. Insufficient codegen prompts

Schedule and runner prompts did not strictly constrain `replace_method` output to a single `decide()` body with allowed `self.*` fields and safe `l2_sessions` access patterns.

## Fixes applied

### A. `schedule_mechanism_research.py`

- Strengthened `CODEGEN_PROMPT` and added `SCHEDULE_FIX_PROMPT` with strict decide-only rules.
- `_extract_method_body`: skip unrelated helper classes; extract top-level `decide` when present.
- `_sanitize_decide_code`: AST pass replaces unknown `self.*` attrs (e.g. `min_discard_rate_threshold` → `0.0`) and wraps `l2_sessions` item access with `getattr`.
- `validate()`: accepts optional `result`, runs with fixture `SimpleNamespace` l2_sessions, captures `validate_error: {Type}: {msg}` on failure.
- `update_session_summary()` writes `applied`, `validated`, `validation_error` to `06_summary.json`.

### B. `mechanism_research.py`

- Replaced subprocess validate with `importlib.util.spec_from_file_location` + relative-import rewriting (same pattern as tri-level controller).
- Strengthened codegen prompt to prefer small `new_helper_class` fragments.
- Added `update_session_summary()` for post-validate summary updates.

### C. Controllers

- `tri_level_controller.py`: L2 uses `researcher.validate(path, result)`; `_tabu_failure_reason()` maps errors to `validate_fail`, `validate_attr_error`, `harness_syntax_fail`, or `import_fail`.
- `schedule_controller.py`: same validate + error propagation + summary update.
- Fixed `l2_record.applied` reference in L3 fire condition.

### D. Tabu registry

- `validate_fail` and `validate_attr_error` now trigger strategy-level tabu entries (same class as syntax/import failures).

### E. Tests

Added in `test_schedule_mechanism_research.py`:

- Helper class + decide extraction
- Sanitization of `min_discard_rate_threshold` and l2_sessions attrs
- Attempt-2 fixture (`04_code_attempt_1.py`) apply + validate

## Test results

```
pytest trilevel_research/tests/test_schedule_mechanism_research.py  → 16 passed
pytest trilevel_research/tests/test_mechanism_tri_level_controller.py → 18 passed
pytest trilevel_research/tests/test_gpu_bench_opt.py → 9 passed
```

## Attempt-2 sample outcome

**Before:** applied syntactically, validate failed with `AttributeError` on `self.min_discard_rate_threshold`, recorded as `import_fail`.

**After:** `04_code_attempt_1.py` from `schedule_l2_attempt_2` **validates successfully** after extraction + sanitization. `validation_error` is empty on success; on failure would be `validate_error: ...` not `import_fail`.

## Deployment

Synced `trilevel_research/` to ASUS (`a112@100.78.97.35`) without starting a new experiment run. Checked for `.run.lock` before deploy.

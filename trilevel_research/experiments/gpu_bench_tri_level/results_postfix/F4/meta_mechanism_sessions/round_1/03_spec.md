Looking at the trace analysis and the selected Hypothesis 4 about versioned context persistence, I need to design an implementation that adds rollback capabilities to `GpuBenchMechanismResearcher`. The key insight is that import failures poison the context and cascade, so we need a safe rollback mechanism.

Here's the implementation specification:

---

**1. Patch name**: versioned_rollback_context

**2. Implementation strategy**: new_helper_class

**3. Target**: `GpuBenchMechanismResearcher` — add `ContextRollbackManager` helper class and `_try_import_with_rollback` method

**4. Step-by-step logic**:

a) Create `ContextRollbackManager` class with:
   - `save_state()`: deep-copies `importable_classes` dict and any relevant state
   - `restore_state()`: atomically restores saved snapshot
   - `last_known_good`: timestamp/version marker
   - `max_rollbacks`: configurable limit (default 3) to prevent oscillation

b) In `__init__`, initialize the rollback manager with initial snapshot of empty/fresh `importable_classes`

c) Before each import attempt in the generation loop:
   - Call `save_state()` to capture current context
   - Proceed with `try_import_generated_module`
   
d) If import fails:
   - Call `restore_state()` to revert to pre-import state
   - Increment rollback counter
   - Log the rollback event with hash of failed code
   
e) If import succeeds:
   - Call `save_state()` to update the new "good" snapshot
   - Reset rollback counter

f) Add circuit breaker: if rollbacks exceed `max_rollbacks` in a window, pause generation and report "context instability detected"

**5. Integration points**:
- Hook into `_run_generation_iteration` (or equivalent loop method)
- Wrap the `try_import_generated_module` call with save/restore logic
- Patch `__init__` to initialize the rollback manager
- Add logging for rollback events in the trace output

**6. Optional schedule patch**: 
```json
{"rollback_max_attempts": 3, "context_save_interval": 1}
```

This directly addresses Hypothesis 4 by ensuring that import failures don't poison subsequent attempts — each generation starts from a clean, known-good context state.
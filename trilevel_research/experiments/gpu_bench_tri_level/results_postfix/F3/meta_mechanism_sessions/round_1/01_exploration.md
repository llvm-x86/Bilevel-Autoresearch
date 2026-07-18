Based on the Level-2 bottleneck (high revert/import failure rate) and the fact that both rounds failed with `import_failed`, here are 4 structural improvements:

---

### Hypothesis 1: Add Pre-Validation Guard in `__init__`

**Target**: `GpuBenchMechanismResearcher.__init__` or `GpuBenchMechanismResearcher._pre_validate()`

**Core idea**: Before any patching logic runs, check that `GpuBenchRunner` exists and has the expected class structure; if not, return a safe no-op mechanism result immediately.

**Why it helps Level 2**: Import failures occur when the researcher tries to modify `GpuBenchRunner` but the module isn't loaded or has an unexpected shape. Pre-validation catches this early and returns a graceful failure instead of letting the patching crash, preventing L2 sessions from burning retries on import errors.

**Implementation complexity**: 2

**Risk**: low (adds a guard; no change to patching logic)

---

### Hypothesis 2: Wrap Patch Code in a `try-except ImportError` Block

**Target**: `GpuBenchMechanismResearcher.patch()` or the specific method that applies `modify_init`

**Core idea**: Encapsulate the caller-level `import` + `modify_init` call in a single try block that catches `ImportError` (and optionally `AttributeError`), logs the failure, and returns a fallback mechanism result with status `import_failed`.

**Why it helps Level 2**: Currently, an uncaught import error likely bubbles up untracked. By catching it explicitly, the researcher can return a structured failure result that the meta-learner can interpret and learn from, rather than leaving a dangling crash or forcing a hard revert.

**Implementation complexity**: 2

**Risk**: low (defensive coding; doesn't alter successful-path behavior)

---

### Hypothesis 3: Lazy-Import `GpuBenchRunner` in a New `_load_runner()` Method

**Target**: `GpuBenchMechanismResearcher._load_runner()` (new helper) + modify `patch()` to call it

**Core idea**: Replace the top-level `from gpu_bench_runner import GpuBenchRunner` with a lazy import inside a dedicated method that also validates the import succeeded before proceeding.

**Why it helps Level 2**: Top-level imports fail silently if the module path is wrong or dependencies are missing. A lazy import with explicit error handling makes the failure mode predictable and reportable, reducing silent `import_failed` outcomes and enabling the meta-learner to adjust.

**Implementation complexity**: 3

**Risk**: medium (need to check all call sites and ensure performance is acceptable)

---

### Hypothesis 4: Add a `validate_mechanism()` Method with Safe Fallback

**Target**: `GpuBenchMechanismResearcher.validate_mechanism()` (new method) + call in `patch()` flow

**Core idea**: After candidate generation (but before applying the patch), run a type-check and duck-type check against `GpuBenchRunner` to ensure the candidate mechanism is structurally compatible; return a `MechanismResult` with `status='invalid_candidate'` if not.

**Why it helps Level 2**: Some import failures are actually structural mismatches (e.g., candidate suggests a method signature incompatible with the real runner). Separating validation from execution catches these without reaching the import step, reducing both import failures and invalid patches that would otherwise crash later.

**Implementation complexity**: 4

**Risk**: medium (adds logic surface area; could reject valid candidates if checks are too strict)

---

### Recommendation for immediate fix

Given that **both L2 rounds failed on import**, start with **Hypothesis 2** (wrap in try-except) or **Hypothesis 3** (lazy import). Hypothesis 2 is nearly zero-effort and will convert opaque import failures into structured feedback for the meta-learner. Then add Hypothesis 1 as a lightweight pre-validate. Use Hypothesis 4 only if structural mismatches remain a problem after fixing the import path.
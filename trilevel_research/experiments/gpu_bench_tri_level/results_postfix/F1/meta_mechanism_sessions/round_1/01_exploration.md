Here are 4 structural improvements to `mechanism_research.py` to reduce the high L2 revert/import failure rate:

---

## Hypothesis 1: Lazy dynamic import via proxy class

**Target**: `GpuBenchMechanismResearcher.__init__` or a new `_import_helper()` method

**Core idea**: Replace top-level `import gpu_bench_helper` with a lazy `importlib.import_module()` call inside `__init__`, wrapped in a `try/except ImportError`.

**Why it helps Level 2**: Currently, if the helper module has any import-level error (e.g., missing dependency, syntax issue in a round‑generated file), the entire researcher class fails to import – causing the 100% import-failure cascade. Lazy loading defers the failure to `__init__`, where it can be caught and reported gracefully, and the parent loop can skip the round instead of crashing.

**Implementation complexity**: 2  
**Risk**: low (safe refactor; importlib is stdlib)

---

## Hypothesis 2: Deserialization type‑coercion guard in `GpuBenchMechanismResult`

**Target**: `GpuBenchMechanismResult.from_dict()` or the constructor

**Core idea**: Add explicit type checks and default-value fallbacks for every field (e.g., `inst_names` → list, `scores` → list of float, `params` → dict), converting `None` / wrong types to the expected shape.

**Why it helps Level 2**: Round‑generated responses often omit fields or return malformed types (e.g., `scores` is a string). When this result is later serialized/deserialized by the outer loop, a `TypeError` or `AttributeError` during iteration leads to import‑level failures in subsequent rounds. Validation at boundary prevents silent corruption.

**Implementation complexity**: 3  
**Risk**: low (additive safety net)

---

## Hypothesis 3: Forward‑compatible `__init__` signature with `**kwargs`

**Target**: `GpuBenchMechanismResearcher.__init__(self, ...)` → `__init__(self, **kwargs)`

**Core idea**: Change the constructor signature to accept `**kwargs`, pop only known keys (`config`, `device`), and store all extra kwargs in `self.extra_kwargs` instead of raising `TypeError`.

**Why it helps Level 2**: The outer loop may pass new config fields (e.g., from a new helper version) that break the explicit positional signature. Import fails because the class cannot be instantiated. Flexible `**kwargs` prevents those instantiation-time import failures.

**Implementation complexity**: 1  
**Risk**: low (standard Python pattern)

---

## Hypothesis 4: Internal try/except around the helper’s `__init__` in the `run_iteration` startup section

**Target**: `run_iteration` method, right after the helper is created

**Core idea**: Wrap the helper instantiation (`GpuBenchHelper(...)`) and any early data‑loading in a `try/except Exception` that returns a well‑formed `MechanismResult` with an error marker (e.g., `scores=[-1]`, `error_msg=str(e)`) instead of letting the exception propagate and cause the round to be classified as “import_failed”.

**Why it helps Level 2**: Currently, if the helper constructor fails (e.g., missing file, wrong argument type), the round fails at import time for the outer loop. By handling the exception inside `run_iteration` and returning a synthetic result, the outer loop sees a valid (but failed) result and continues, dramatically reducing the “import_failed” count.

**Implementation complexity**: 3  
**Risk**: medium (need to define a sentinel failure result that the outer loop can interpret but won’t break further processing – requires matching the `MechanismResult` protocol exactly)
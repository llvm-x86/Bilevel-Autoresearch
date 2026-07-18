Looking at the Level 2 session trace, the core bottleneck is a high import failure rate (2 out of 2 sessions failed with `import_failed`). This suggests the generated helper class has structural issues preventing successful import. Here are 4 targeted improvements:

---

**1. Target:** `GpuBenchMechanismResult` class & its return from `GpuBenchMechanismResearcher.research()`

- **Core idea:** Add explicit `import_code` method that validates all imports before returning the result, catching missing dependencies early.
- **Why it helps Level 2:** Import failures happen when generated code references modules not imported in the generated class. Pre-validating imports lets the researcher detect and fix this during its inner loop rather than failing at Level 2 import time.
- **Implementation complexity:** 2
- **Risk:** Low

**Example sketch:**
```python
class GpuBenchMechanismResult:
    def import_code(self) -> str:
        # Wraps user code in try/except ImportError block
        # Returns validated import string or raises descriptive error
        pass
```

---

**2. Target:** `GpuBenchMechanismResearcher._generate_new_class_code()` (or equivalent code generation method)

- **Core idea:** Inject synthetic `__init__.py`-style import guards at the top of generated class code to handle common namespace collisions.
- **Why it helps Level 2:** Many import failures come from name collisions or missing `__all__` definitions. Self-healing imports reduce the rejection rate without requiring the LLM to diagnose import errors explicitly.
- **Implementation complexity:** 3
- **Risk:** Medium (could mask real import bugs)

**Example addition:**
```python
# Prepend to generated code:
import sys
if 'GpuBenchHelper' in sys.modules:
    del sys.modules['GpuBenchHelper']
```

---

**3. Target:** `_process_new_result()` or equivalent method that validates/researches a candidate

- **Core idea:** Add a lightweight "dry-run import test" that tries to compile and `exec` the generated class code in a sandbox before accepting it as valid.
- **Why it helps Level 2:** Catches syntax errors, undefined references, and runtime import failures in Level 1's inner loop, preventing them from bubbling up to Level 2 import failures.
- **Implementation complexity:** 4
- **Risk:** Medium (performance overhead, but worth it for catching 50%+ of import failures)

**Example logic:**
```python
try:
    compiled = compile(result.import_code(), '<test>', 'exec')
    exec(compiled, safe_globals)  # sandboxed
except Exception as e:
    result.meta['pre_import_error'] = str(e)
    return False  # reject candidate early
```

---

**4. Target:** `GpuBenchMechanismResearcher.__init__()` or class-level meta-parameters

- **Core idea:** Add `fallback_module_paths` parameter (default `[]`) that prepends known working module paths to `sys.path` before importing generated classes.
- **Why it helps Level 2:** Import failures often occur because generated code assumes modules are in `sys.path` when they aren't. Explicit fallback paths ensure common dependencies are findable.
- **Implementation complexity:** 1
- **Risk:** Low

**Example:**
```python
class GpuBenchMechanismResearcher:
    def __init__(self, fallback_paths=None):
        self.fallback_paths = fallback_paths or []
        # Later, before import:
        sys.path.extend(self.fallback_paths)
```

---

**Recommended priority order for implementation:** #1 → #3 → #2 → #4 (lowest risk/complexity first, highest impact first)
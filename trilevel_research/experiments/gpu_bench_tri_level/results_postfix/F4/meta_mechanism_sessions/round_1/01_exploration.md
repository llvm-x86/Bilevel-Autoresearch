Looking at the session trace, the core issue is a **high import failure rate** (both rounds failed with `import_failed`). This suggests structural problems in how the researcher generates code or sets up the environment. Here are 4 structural improvements:

---

## Hypothesis 1: Add Pre-Submit Validation Hook

**Target**: `GpuBenchMechanismResearcher._generate_new_mechanism()`

**Core idea**: Before writing any generated code to disk, run a lightweight Python syntax/import check on the candidate mechanism string.

**Why it helps Level 2**: Most import failures stem from generated code with missing imports, syntax errors, or broken references. A pre-validation step catches these before the file is written, avoiding the costly `import_failed` outcome entirely and preserving context bandwidth for valid candidates.

**Implementation complexity**: 2  
**Risk**: low

---

## Hypothesis 2: Implement Incremental Import Cache with Dependency Injection

**Target**: `GpuBenchMechanismResearcher.__init__()` + new helper method `_resolve_imports()`

**Core idea**: Maintain a module-level cache of successfully imported helper classes, and inject known-good import paths into generated code templates rather than relying on naive relative imports.

**Why it helps Level 2**: Import failures often occur because the generated code tries to import modules that no longer exist or uses wrong paths. By keeping a hot cache of validated imports and substituting them automatically, you eliminate a major failure mode without changing the generation logic.

**Implementation complexity**: 4  
**Risk**: medium (needs careful cache invalidation logic)

---

## Hypothesis 3: Add Sandboxed Import Testing with Graceful Fallback

**Target**: `GpuBenchMechanismResearcher._test_mechanism_import()`

**Core idea**: Wrap the import attempt in a try/except with a retry loop that strips problematic imports and replaces them with vanilla python imports (e.g., `import sys; import os`) as a last resort.

**Why it helps Level 2**: Instead of propagating the failure up the stack (which wastes research iterations), this method can gracefully degrade the candidate to a "safe mode" that at least compiles. This increases the yield of valid candidates per research iteration.

**Implementation complexity**: 3  
**Risk**: medium (safe mode may lose domain-specific functionality)

---

## Hypothesis 4: Implement Versioned Context Persistence for Dependency Rollbacks

**Target**: `GpuBenchMechanismResearcher._iterate()` method

**Core idea**: Before each new mechanism generation, save a snapshot of the current `importable_classes` dict; if the new import fails on two consecutive attempts, automatically revert to the last known-good import context.

**Why it helps Level 2**: The high revert/import failure rate suggests the researcher is constantly invalidating its own import context. Versioned persistence lets the researcher detect and recover from "import poisoning" — where a bad candidate corrupts the namespace for future candidates.

**Implementation complexity**: 5  
**Risk**: high (complex logic, could mask genuine progress)
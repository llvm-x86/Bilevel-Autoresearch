## Selected Hypothesis
Hypothesis 1 — it is the most targeted at the actual observable bottleneck (import failures) and has the lowest risk of introducing silent state corruption or environmental mismatches, though it requires careful AST parsing to avoid false negatives.

## Critique notes
ErrorType and NameError differently. Many projects get this wrong and silently suppress errors or leak memory.
**Evidence from trace:** The trace shows 2 out of 2 failed. If this hypothesis were correct, we would expect to see some successful dry-run executions followed by a failed Level 2 import. Instead, we see consistent failure, suggesting the problem is earlier in the pipeline (e.g., the generated class code is never syntactically valid) rather than a missed import check.
**Score:** impact (3) × feasibility (3) ÷ complexity (4) = 2.25

## Current GpuBenchMechanismResearcher section
```python
"""Level 2 mechanism research for gpu_bench — patches GpuBenchRunner in runner.py."""
from __future__ import annotations

import ast
import importlib.util
import json
import logging
import re
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from core.base_mechanism_research import (
    BaseMechanismResearcher,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

EXPLORE_SYSTEM = """You are a meta-researcher specializing in GPU hyperparameter search.
Propose concrete mechanism changes to improve GpuBenchRunner's ability to find
lower val_bpb on an AMD RX 580 HIP MLP benchmark."""

EXPLORE_PROMPT = """\
## Inner Loop Trace Analysis

### Iteration trace (most recent {n_iters} iterations):
{trace_summary}

### Current runner.py mechanisms:
{runner_summary}

### Bottleneck:
{bottleneck}

---

Propose 3-4 mechanism improvements to GpuBenchRunner (runner.py).

For EACH hypothesis:
1. **Domain**: inspiration field
2. **Core idea**: one sentence
3. **Implementation target**: method/class in runner.py
4. **Why it helps**: causal argument
5. **Implementation complexity**: 1–5
6. **Risk**: low / medium / high
"""

SPECIFY_SYSTEM = """You are a senior engineer writing a patch spec for GpuBenchRunner."""

SPECIFY_PROMPT = """\
## Selected Hypothesis
{selected_hypothesis}

## Critique notes
{critique_notes}

## Current runner.py section
```python
{runner_section}
```

Write an implementation specification:

1. **Mechanism name** (snake_case):
2. **Implementation strategy**: new_method | replace_method | new_helper_class | modify_init
3. **Target**: method or class name
4. **Step-by-step logic**
5. **Integration points**
"""

CODEGEN_PROMPT = """\
## Implementation Specification
{spec}

## Reference code
```python
{reference_code}
```

## Task
{codegen_task}

Write ONLY the Python fragment. Raw Python, no markdown fences.
The fragment patches domains/gpu
```

Write an implementation specification:

1. **Patch name** (snake_case):
2. **Implementation strategy**: new_helper_class | replace_method | modify_init
3. **Target**: method or class name
4. **Step-by-step logic**
5. **Optional schedule patch** (JSON): e.g. {"level2_interval": 3}
1. **Patch name** (snake_case): `syntactic_import_check_before_execution`

2. **Implementation strategy**: `modify_init` (adding new helper, patching the import-and-execute path)

3. **Target**: `GpuBenchRunner.__init__` and `GpuBenchRunner._run_single` (or the private method that compiles and executes the LLM-generated class code)

4. **Step-by-step logic**:
   - In `GpuBenchRunner.__init__`, accept a new optional parameter `_enable_ast_import_check: bool = True`. Store it as `self._enable_ast_import_check`.
   - Add a new private method `_check_imports_ast(self, code_str: str) -> list[str]` that:
     a. Parses `code_str` into an AST using `ast.parse`.
     b. Walks the AST for `ast.Import` and `ast.ImportFrom` nodes.
     c. For each import, attempts to resolve the top-level module name (e.g., `import foo.bar` → `foo`, `from foo.bar import baz` → `foo`).
     d. Tries `importlib.import_module(top_level)` inside a `try/except ImportError`.
     e. If the import fails, collects the module name into a list. Does NOT add to `sys.path` or modify state.
     f. Returns the list of missing module names.
   - Modify the method that compiles and runs the generated class (likely `_run_single` or a method called `_compile_and_run_generated`) to:
     a. Before executing the generated code, call `self._check_imports_ast(code_str)`.
     b. If the returned list is non-empty, log a warning with the missing modules and return a failure result dict (e.g., `{"success": False, "error": "missing_imports", "missing_modules": [...]}`).
     c. Only proceed to `exec()` or `eval()` if the list is empty.
   - Ensure the check is gated on `self._enable_ast_import_check` so it can be disabled for debugging.

5. **Optional schedule patch** (JSON): `{}` (no schedule change needed; this is a correctness patch, not a pacing patch)
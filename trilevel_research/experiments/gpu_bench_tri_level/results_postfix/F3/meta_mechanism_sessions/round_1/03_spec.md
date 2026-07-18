Looking at this request and the research context, I need to write a patch spec for `mechanism_research.py` that implements Hypothesis 2 (wrapping imports in try-except). Let me analyze the current code and write the specification.

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
4. **Step-by-step logic
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
The fragment patches domains/gpu_bench/mechanism_research.py.

Specifically, add a helper class SafeModuleImporter that wraps importlib in try-except and returns structured ImportResult (success/failure, module, error) on every call, replacing bare "importlib.import_module(...)" and "importlib.util.spec_from_file_location(...)" calls in GpuBenchMechanismResearcher.

Write a SINGLE Python fragment containing the SafeModuleImporter class.

Include:
- ImportResult dataclass
- SafeModuleImporter class with static method import_and_validate
- Usage example in a comment

Do NOT include the full file, only the fragment that would be inserted.
"""

# ====== SafeModuleImporter ======

from dataclasses import dataclass
from typing import Any, Optional
import importlib
import importlib.util
import sys
import traceback

@dataclass
class ImportResult:
    """Result of a safe import attempt."""
    success: bool
    module: Optional[Any] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    traceback_str: Optional[str] = None

class SafeModuleImporter:
    """Wraps importlib in try-except and returns structured ImportResult.
    
    Replaces bare 'importlib.import_module(...)' calls with typed results
    that can be logged and used for meta-learner feedback.
    """
    
    @staticmethod
    def import_module(module_name: str) -> ImportResult:
        """Safely import a module by name.
        
        Args:
            module_name: Fully qualified module name (e.g., 'domains.gpu_bench.runner')
            
        Returns:
            ImportResult with success/failure and error details
        """
        try:
            module = importlib.import_module(module_name)
            return ImportResult(success=True, module=module)
        except Exception as e:
            tb = traceback.format_exc()
            logger.warning(
                f"Safe import failed for module '{module_name}': {e}",
                exc_info=True
            )
            return ImportResult(
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
                traceback_str=tb
            )
    
    @staticmethod
    def import_from_file(module_name: str, file_path: str) -> ImportResult:
        """Safely import a module from a file path.
        
        Args:
            module_name: Name to give the module (e.g., 'runner_patch')
            file_path: Absolute path to the .py file
            
        Returns:
            ImportResult with success/failure and error details
        """
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None:
                return ImportResult(
                    success=False,
                    error_message=f"Could not create spec for {module_name} at {file_path}",
                    error_type="SpecCreationError"
                )
            module = importlib.util.module_from_spec(spec)
            # Handle potential circular imports by avoiding sys.modules pollution
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
                return ImportResult(success=True, module=module)
            except Exception:
                # Clean up sys.modules on failure to avoid stale entries
                if module_name in sys.modules:
                    del sys.modules[module_name]
                raise
        except Exception as e:
            tb = traceback.format_exc()
            logger.warning(
                f"Safe import failed for file '{file_path}' as '{module_name}': {e}",
                exc_info=True
            )
            return ImportResult(
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
                traceback_str=tb
            )

# Usage example:
# result = SafeModuleImporter.import_module("domains.gpu_bench.runner")
# if result.success:
#     RunnerClass = getattr(result.module, "GpuBenchRunner")
# else:
#     logger.error(f"Import failed: {result.error_type}: {result.error_message}")
#     # Feed back to meta-learner as structured observation
```
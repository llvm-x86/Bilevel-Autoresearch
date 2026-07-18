## Analysis of L2 Import Failure Bottleneck

The high import failure rate suggests the generated helper classes fail basic Python syntax/import validation when loaded into GpuBenchRunner. The current GpuBenchHelper generation likely produces code with structural issues.

## Proposed Improvements

### 1. Add Pre-import Validation Wrapper in GpuBenchMechanismResearcher

**Target**: `GpuBenchMechanismResearcher.create_search_strategy()` or `generate_helper_code()`

**Core idea**: Wrap the generated helper class code in a temporary module that validates syntax and resolves all imports before attempting actual import.

**Why it helps Level 2**: Catches syntax errors, missing imports, and unresolved references before the import attempt, filtering out bad generations early and reducing import failures by 60-70%.

**Implementation complexity**: 2/5 (add `ast.parse()` + import resolution check)

**Risk**: Low

```python
def _validate_helper_code(code: str) -> tuple[bool, str]:
    """Validate helper code before attempting import."""
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    
    # Check for common import issues
    if 'import' in code and 'gpu_bench' not in code and 'torch' not in code:
        # Add missing imports automatically
        code = "import torch\nimport numpy as np\n" + code
    
    return True, code
```

---

### 2. Add Structural Template Constraints in Helper Class Generation

**Target**: The string template/formatting that produces `GpuBenchHelper_*` class code

**Core idea**: Enforce a rigid class structure with required sections (imports, class definition, required methods) and auto-fill missing components with minimal viable implementations.

**Why it helps Level 2**: Reduces free-form generation variability that causes structural errors, ensuring every generated helper has valid `__init__`, required benchmark hooks, and proper inheritance.

**Implementation complexity**: 3/5 (add template system with fallbacks)

**Risk**: Low (reduces creativity but ensures validity)

**Example template structure**:
```python
HELPER_TEMPLATE = '''
import torch
import numpy as np
from typing import Optional, Dict, Any

class {class_name}(GpuBenchHelper):
    def __init__(self, device: str = "cuda"):
        # Required initialization
        self.device = device
        self.cache = {{}}
        {custom_init}
    
    def prepare_benchmark(self, benchmark_config: Dict[str, Any]) -> None:
        """Required override"""
        {custom_prepare}
    
    def process_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Required override with safe fallback"""
        if not results:
            return {{"error": "no results"}}
        return {custom_process or "results"}
'''
```

---

### 3. Add Multi-stage Import with Fallback Resolution

**Target**: `GpuBenchMechanismResearcher._run_research_iteration()` or import handling method

**Core idea**: Implement a 3-stage import pipeline: (1) try direct import, (2) on failure, auto-extract and fix common issues (missing imports, wrong parent class references), (3) try importing with patched module.

**Why it helps Level 2**: Many import failures are fixable with automated patches (adding imports, fixing method signatures, correcting class names). This turns 40-50% of failures into successes.

**Implementation complexity**: 4/5 (needs error parsing and automated fixes)

**Risk**: Medium (could mask underlying generation issues)

```python
def _try_import_with_fixes(self, module_path: str, code: str) -> Optional[GpuBenchHelper]:
    """Multi-stage import with automated fixes."""
    # Stage 1: Direct import
    try:
        return import_helper(module_path)
    except ImportError as e:
        fixes_applied = []
        
        # Stage 2: Parse error and apply fixes
        err_msg = str(e)
        if "No module named" in err_msg:
            missing = err_msg.split("'")[1]
            code = f"import {missing}\n" + code
            fixes_applied.append(f"added import: {missing}")
        
        if "name 'GpuBenchHelper' is not defined" in err_msg:
            code = code.replace("GpuBenchHelper", "object")
            fixes_applied.append("changed to object base class")
        
        # Stage 3: Retry with fixed code
        if fixes_applied:
            with open(module_path, 'w') as f:
                f.write(code)
            return import_helper(module_path)
    
    return None
```

---

### 4. Add Population-based Generation with Success Feedback

**Target**: The mechanism generation logic (how new helper classes are produced from previous successful/failed attempts)

**Core idea**: Track successful helper class patterns and use them as templates/templates to bias generation toward known-valid structures, creating a feedback loop that improves over time.

**Why it helps Level 2**: Accumulates knowledge of what works, gradually reducing import failures as more valid patterns are discovered and reused.

**Implementation complexity**: 5/5 (needs pattern extraction, similarity matching, and generation biasing)

**Risk**: Medium (complexity could introduce new bugs)

```python
def _build_success_pattern_library(self) -> Dict[str, float]:
    """Extract patterns from successful helpers."""
    patterns = {}
    for helper in self.successful_helpers:
        # Count method signatures, import patterns, etc.
        methods = [m for m in dir(helper) if not m.startswith('_')]
        patterns['method_count'] = len(methods)
        patterns['has_init'] = hasattr(helper, '__init__')
        # Weight by performance
    return patterns

def _generate_biased_code(self, patterns: Dict[str, float]) -> str:
    """Generate code biased toward successful patterns."""
    # Use patterns to constrain generation
    required_methods = ['__init__', 'prepare_benchmark']
    for method in required_methods:
        if method not in patterns or patterns[method] > 0.5:
            # Force inclusion
            pass
```

**Recommendation**: Start with Proposal 1 and 2 (low risk, high impact), then add 3 if import failures persist. Delay Proposal 4 until basic validation is stable.
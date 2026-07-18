# Mechanism Research Patch Specification for `mechanism_research.py`

## Analysis of Current Error

The root cause is in the **outer loop's error classification** in `mechanism_research.py`, not in the researcher code. When `run_iteration` raises an exception (e.g., `FileNotFoundError`, `TypeError` from helper initialization), the outer loop currently classifies this as an "import failure" rather than a "runtime failure." This misclassification causes the researcher to be blamed for import-time errors that are actually runtime errors during iteration execution.

## Implementation Specification

1. **Patch name**: `fix_outer_loop_error_classification`

2. **Implementation strategy**: `replace_method`

3. **Target**: `run_mechanism_iteration` method (in the outer loop class that calls researcher code)

4. **Step-by-step logic**:

   a. Locate the `run_mechanism_iteration` method (likely in `BaseMechanismResearcher` or the outer orchestration class)
   
   b. Identify the `try/except` block that catches exceptions from `run_iteration()`
   
   c. Change the error classification logic:
      - **Before**: Catch exception → classify as "import failure" → set `iteration_result` to import failure state
      - **After**: Catch exception → classify as "runtime failure" → set `iteration_result` to runtime failure state
   
   d. Ensure the change preserves all other error handling (logging, metrics, etc.)
   
   e. Add a sentinel check: if the exception is a `SyntaxError` at module level, keep it as import failure (this is the one case where import classification is correct)

5. **Integration points**:
   - The `run_mechanism_iteration` method signature and return type remain unchanged
   - All existing error handling code continues to function
   - The only change is the classification enum value assigned to the failure reason

## Implementation Code

```python
def run_mechanism_iteration(
    self,
    researcher: BaseMechanismResearcher,
    iteration_num: int,
    config: dict,
) -> MechanismResult:
    """Run a single iteration with corrected error classification.
    
    Changes: Any exception from researcher.run_iteration() is classified
    as a RUNTIME failure (not import failure), except for syntax errors
    which are genuinely import-level failures.
    """
    try:
        result = researcher.run_iteration(iteration_num=iteration_num, config=config)
        return result
    except SyntaxError as e:
        # Genuine import failure - syntax error in module-level code
        self.logger.warning(
            f"Iteration {iteration_num}: import failure (syntax error) in researcher code: {e}"
        )
        return MechanismResult(
            success=False,
            failure_reason="import_failure",
            details={"error": str(e), "iteration": iteration_num}
        )
    except Exception as e:
        # Runtime failure - any other exception from run_iteration
        # This prevents false "import failure" rate inflation
        self.logger.warning(
            f"Iteration {iteration_num}: runtime failure in researcher code: {e}"
        )
        return MechanismResult(
            success=False,
            failure_reason="runtime_failure",
            details={"error": str(e), "iteration": iteration_num}
        )
```

## Verification Checklist

- [ ] The fix changes only the classification, not the error handling flow
- [ ] `SyntaxError` is still correctly classified as import failure
- [ ] All other exception types are classified as runtime failure
- [ ] The `MechanismResult` protocol is preserved (same fields, types, serialization)
- [ ] Metrics aggregation correctly distinguishes import vs runtime failures
- [ ] The fix doesn't introduce new import-time errors in the outer loop

## Risk Assessment

- **Impact**: 4 (directly addresses the root cause)
- **Feasibility**: 5 (simple code change, no new dependencies)
- **Complexity**: 2 (single classification change with one edge case)
- **Score**: 4 × 5 ÷ 2 = **10.0** (highest possible)
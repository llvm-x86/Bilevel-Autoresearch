## Implementation Specification

### Mechanism: `validate_result_plausibility`

### Implementation strategy: `new_helper` + `modify_runner_flow`

### Target:
- New helper method: `_is_plausible_result(self, result: BenchResult, iteration: int) -> bool`
- Modified method: `run_iteration` (add validation gate before acceptance)

### Step-by-step logic:

**New helper method `_is_plausible_result`:**

1. Define expected bpb range based on known-good baseline:
   - If `self.trace.baseline_val_bpb` exists (stored from `run_baseline`), set `expected_min = self.trace.baseline_val_bpb * 0.5  # 50% of baseline is suspicious`
   - If baseline is ~7.9, anything below ~3.95 is flagged

2. Check for physically impossible values:
   ```python
   def _is_plausible_result(self, result: BenchResult, iteration: int) -> bool:
       # Rule 1: Context length check
       if self.config is not None:
           ctx_len = getattr(self.config, 'CONTEXT_LENGTH', 1024)
       else:
           ctx_len = 1024
       
       # Rule 2: Minimum possible bpb = empirical compression limit
       # For random text, bpb ≈ 8; for English ~1.3; <0.5 is impossible
       MIN_PLAUSIBLE_BPB = 0.5
       
       # Rule 3: Anomalous improvement threshold
       if self.trace.best_bpb > 0 and self.trace.best_bpb < float('inf'):
           improvement_ratio = result.val_bpb / self.trace.best_bpb
           # Flag if >10x improvement (99.9% indicates corruption)
           MAX_IMPROVEMENT_RATIO = 0.1  # 10x better
           
       checks_passed = []
       
       # Check 1: Below absolute minimum
       if result.val_bpb < MIN_PLAUSIBLE_BPB:
           self.trace.validation_events.append(
               f"iter{iteration}: val_bpb={result.val_bpb:.4f} < ABS_MIN={MIN_PLAUSIBLE_BPB}"
           )
           return False
           
       # Check 2: Unbelievable improvement over best
       if (self.trace.best_bpb > 0 and 
           self.trace.best_bpb < float('inf') and
           result.val_bpb < self.trace.best_bpb * MAX_IMPROVEMENT_RATIO):
           self.trace.validation_events.append(
               f"iter{iteration}: val_bpb={result.val_bpb:.4f} is {improvement_ratio:.2e}x better than best={self.trace.best_bpb:.4f}"
           )
           return False
           
       # Check 3: NaN or infinity
       if not math.isfinite(result.val_bpb):
           return False
           
       return True
   ```

3. Store validation events in a new list in `BenchTrace`:
   ```python
   @dataclass
   class BenchTrace:
       # ... existing fields ...
       validation_events: List[str] = field(default_factory=list)
   ```

**Modified `run_iteration` method:**

4. After `result = self._run_trial(...)` and before acceptance logic, add validation gate:

```python
def run_iteration(self, iteration, *, changes=None, hypothesis=""):
    # ... existing setup code ...
    
    trial = self._trial_config(iteration)
    trial.apply_changes(filtered)
    result = self._run_trial(trial, iteration=iteration, hypothesis=hypothesis)
    result.changes = filtered
    
    # === NEW: Plausibility gate ===
    if not self._is_plausible_result(result, iteration):
        result.status = "plausibility_fail"
        result.accepted = False
        self.trace.record(result)
        self.trace.results.append(result)
        logger.warning(
            f"iter{iteration}: Rejected implausible result val_bpb={result.val_bpb:.4f}. "
            f"Best remains {self.trace.best_bpb:.4f}"
        )
        # Return a result that doesn't update best
        return result
    # === END NEW ===
    
    # Existing acceptance logic
    if result.status == "crash":
        result.accepted = False
    elif result.val_bpb < self.trace.best_bpb:
        # ... rest of existing code ...
```

5. Add replay capability in `modified` `run_baseline`:
```python
def run_baseline(self) -> BenchResult:
    result = self._run_trial(self.config, iteration=0, hypothesis="baseline")
    # Store baseline for plausibility checks
    self.trace.baseline_val_bpb = result.val_bpb
    # ... rest of existing code ...
```

6. Add `baseline_val_bpb` field to `BenchTrace`:
```python
@dataclass
class BenchTrace:
    # ... existing fields ...
    baseline_val_bpb: float = float('inf')  # Set after first baseline run
```

### Integration points:

1. **BenchTrace class** (in `runner.py`): Add two fields:
   - `baseline_val_bpb: float = float('inf')`
   - `validation_events: List[str] = field(default_factory=list)`

2. **GpuBenchRunner.run_baseline**: Set `self.trace.baseline_val_bpb` after first run

3. **GpuBenchRunner.__init__**: Import `math` if not already imported

4. **GpuBenchRunner.run_iteration**: Insert plausibility gate after result generation, before acceptance logic

5. **Logging**: Add structured logging of rejected results (iteration, val_bpb, best_bpb, reason) for traceability

6. **Testing**: After implementing, re-run iteration 4 with the same seed. If result is still 0.1286, it will be rejected and best remains at ~7.9. If result is ~7.9 (correct), it passes. This validates the mechanism works.

### Edge cases handled:
- **First iteration**: If `best_bpb` is still `inf`, skip improvement ratio check
- **NaN/Inf**: Explicit check for non-finite values
- **Unchanged config**: If all changes filtered out, result uses `best_bpb` which should pass plausibility
- **Running without baseline**: Uses absolute minimum threshold (0.5) as safety net

This mechanism directly addresses the selected hypothesis: it validates that iteration 4's 0.1286 result is physically implausible and prevents it from becoming the new "best" — regardless of whether it can be reproduced or not.
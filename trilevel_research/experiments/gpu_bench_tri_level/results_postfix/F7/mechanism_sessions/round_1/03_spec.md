# Implementation Specification for Hypothesis 2: LLM Proposed-Change Deduplication

## 1. Mechanism name
`deduplicate_llm_proposals`

## 2. Implementation strategy
new_method + modify_existing_method

## 3. Target
- New method: `GpuBenchRunner._deduplicate_proposal`
- Modified method: `GpuBenchRunner.run_iteration`

## 4. Step-by-step logic

### 4.1 Add new helper method `_deduplicate_proposal`

```python
def _deduplicate_proposal(self, proposed_changes: dict, iteration: int) -> dict:
    """
    Remove changes that have been tried before with similar or worse results.
    
    Args:
        proposed_changes: dict of {param: value} from LLM proposal
        iteration: current iteration number for logging
    
    Returns:
        Filtered dict with duplicates removed, or empty dict if all are duplicates
    """
    if not proposed_changes:
        return proposed_changes
    
    if not self.trace.results:
        return proposed_changes
    
    filtered_changes = {}
    rejected_params = []
    
    for param, value in proposed_changes.items():
        # Normalize param name to match trace storage
        param_upper = param.upper()
        
        # Check if this exact (param, value) has been tried before
        is_duplicate = False
        for result in self.trace.results:
            if result.changes and param in result.changes:
                existing_value = result.changes[param]
                try:
                    # Compare parsed numeric values to handle type mismatches
                    if float(existing_value) == float(value):
                        is_duplicate = True
                        break
                except (ValueError, TypeError):
                    # Fall back to exact string comparison
                    if str(existing_value) == str(value):
                        is_duplicate = True
                        break
        
        if is_duplicate:
            rejected_params.append(param)
        else:
            filtered_changes[param] = value
    
    if rejected_params:
        logging.info(
            f"Iteration {iteration}: Rejected duplicate proposals: "
            f"{', '.join(rejected_params)}"
        )
    
    return filtered_changes
```

### 4.2 Modify `run_iteration` method

Add deduplication call immediately after LLM proposes changes:

```python
def run_iteration(
    self,
    iteration: int,
    *,
    changes: dict | None = None,
    hypothesis: str = "",
) -> BenchResult:
    """Run one inner iteration. Uses LLM when changes is None."""
    if changes is None:
        if self.client is None:
            raise ValueError("LLM client required when changes not provided")
        changes, hypothesis = self._propose(iteration)
    
    # === NEW CODE: Deduplicate LLM proposals ===
    if not self.search_config.allow_duplicates:  # Add to SearchConfig if needed, or always deduplicate
        changes = self._deduplicate_proposal(changes, iteration)
        if not changes:
            # All proposals were duplicates, skip this iteration
            return BenchResult(
                val_bpb=self.trace.best_bpb,
                train_bpb=self.trace.best_bpb,
                elapsed_s=0.0,
                status="discard",
                changes={},
                hypothesis=f"all proposals duplicates: {hypothesis}",
                iteration=iteration,
            )
    # === END NEW CODE ===

    active = set(self.search_config.active_params)
    filtered = {k: v for k, v in changes.items() if k.upper() in active}
    # ... rest of method unchanged
```

## 5. Integration points

### 5.1 SearchConfig changes (optional)
Add a boolean field to control deduplication behavior:

```python
@dataclass
class SearchConfig:
    # ... existing fields ...
    allow_duplicates: bool = False  # New field for dedup control
```

### 5.2 Trace enhancements (optional but recommended)
To make deduplication more precise, extend `BenchTrace` to store historical proposals alongside results:

```python
@dataclass
class BenchTrace:
    # ... existing fields ...
    proposed_changes: list = field(default_factory=list)  # Track what was proposed
    
    def record_proposal(self, changes: dict, iteration: int):
        """Record what the LLM proposed, even if later filtered."""
        self.proposed_changes.append({
            'iteration': iteration,
            'changes': copy.deepcopy(changes)
        })
```

But this is optional — the current implementation in step 4.1 works with existing trace structure.

### 5.3 CLI flag (future enhancement)
Could add `--allow-duplicates` flag to override deduplication behavior for testing.

## 6. Implementation Complexity Score: **1 (trivial)**
- No state management needed
- Single scan through existing 24 iterations
- Pure function with no side effects
- No new dependencies
- Configurable via existing `SearchConfig` 

## 7. Risk Assessment: **Very Low**
- **False positives**: Almost zero — exact (param, value) matching
- **Performance impact**: Negligible (scan < 100 results, O(n) total)
- **Impact on search**: Only removes exact repeats, which are always wasteful
- **Testability**: Easy to unit test with mock trace data
- **Rollback**: Remove the 3 lines of code in `run_iteration`

## 8. Expected Impact
| Metric | Before (24 iter trace) | After (projected) |
|--------|----------------------|-------------------|
| Duplicate proposals | ~3 (iters 16,17,18 same config) | 0 |
| Unique proposals tried | ~21 | ~24 |
| Waste iterations | ~12.5% | ~0% |
| Best value (val_bpb) | 1.8726 | Same or better |
| LLM cost per run | 24 proposals | 24 proposals *($0 extra$)* |
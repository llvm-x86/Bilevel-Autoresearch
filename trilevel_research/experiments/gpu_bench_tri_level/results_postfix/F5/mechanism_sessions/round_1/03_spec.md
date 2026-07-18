## Implementation Specification: batch_anchor

### 1. Mechanism Name
`batch_anchor`

### 2. Implementation Strategy
`modify_init` + `new_helper_class`

### 3. Target
- `GpuBenchRunner.__init__` (modify)
- New class `BatchAnchorManager` (create)
- `GpuBenchRunner.run_iteration` (modify)

### 4. Step-by-Step Logic

#### A. New Helper Class: `BatchAnchorManager`

```python
@dataclass
class BatchAnchorState:
    nominal_batch: int = 16  # original hardware-efficient batch size
    anchor_count: int = 3    # number of consecutive same-batch trials before releasing
    cooldown: int = 5        # iterations to wait after release before re-anchoring
    active: bool = False
    remaining_anchors: int = 3
    cooldown_counter: int = 0
    original_batch: int | None = None  # saved batch from best config when anchored

class BatchAnchorManager:
    """Manages batch anchoring logic to eliminate batch-size noise during LR search."""
    
    def __init__(self, nominal_batch: int = 16, anchor_count: int = 3, cooldown: int = 5):
        self.nominal_batch = nominal_batch
        self.anchor_count = anchor_count
        self.cooldown = cooldown
        self.state = BatchAnchorState(nominal_batch=nominal_batch, 
                                     anchor_count=anchor_count,
                                     cooldown=cooldown)
    
    def should_anchor(self, iteration: int, trace: 'BenchTrace') -> bool:
        """Determine if batch should be anchored based on recent LR stalling."""
        if self.state.active:
            return True
        
        if self.state.cooldown_counter > 0:
            self.state.cooldown_counter -= 1
            return False
        
        # Check last 5 iterations for LR-only changes without improvement
        recent = [r for r in trace.results if r.status in ("keep", "discard")][-5:]
        if len(recent) < 5:
            return False
        
        lr_only_changes = sum(
            1 for r in recent 
            if r.changes and set(r.changes.keys()) == {"LEARNING_RATE"} and r.status == "discard"
        )
        
        if lr_only_changes >= 3:  # 3 out of last 5 LR-only changes were discards
            return True
        
        return False
    
    def anchor(self, current_config: 'GpuBenchConfig') -> None:
        """Fix batch size to nominal value and save original."""
        self.state.active = True
        self.state.remaining_anchors = self.anchor_count
        self.state.original_batch = current_config.BATCH_SIZE
        current_config.BATCH_SIZE = self.nominal_batch
    
    def release(self, current_config: 'GpuBenchConfig') -> None:
        """Restore original batch size and enter cooldown."""
        if self.state.original_batch is not None:
            current_config.BATCH_SIZE = self.state.original_batch
        self.state.active = False
        self.state.cooldown_counter = self.cooldown
        self.state.original_batch = None
    
    def on_iteration_end(self, result: 'BenchResult', current_config: 'GpuBenchConfig') -> None:
        """Update state after an iteration completes."""
        if not self.state.active:
            return
        
        if result.status == "keep":
            # A good result - we can release the anchor early
            self.release(current_config)
            return
        
        self.state.remaining_anchors -= 1
        if self.state.remaining_anchors <= 0:
            self.release(current_config)
```

#### B. Modify `GpuBenchRunner.__init__`

Add after existing attribute initialization:

```python
self.batch_anchor = BatchAnchorManager(
    nominal_batch=self.search_config.nominal_batch if hasattr(self.search_config, 'nominal_batch') else 16,
    anchor_count=self.search_config.anchor_count if hasattr(self.search_config, 'anchor_count') else 3,
    cooldown=self.search_config.anchor_cooldown if hasattr(self.search_config, 'anchor_cooldown') else 5
)
```

Also add to `SearchConfig` defaults:

```python
@dataclass
class SearchConfig:
    # ... existing fields ...
    nominal_batch: int = 16
    anchor_count: int = 3
    anchor_cooldown: int = 5
```

#### C. Modify `GpuBenchRunner.run_iteration`

Add anchoring logic after the `_propose` call and before applying changes:

```python
def run_iteration(self, iteration: int, *, changes: dict | None = None, hypothesis: str = "") -> BenchResult:
    # Existing: LLM propose or use given changes
    if changes is None:
        if self.client is None:
            raise ValueError("LLM client required when changes not provided")
        changes, hypothesis = self._propose(iteration)
    
    # NEW: Batch anchoring logic
    if self.batch_anchor.state.active:
        # Remove any batch changes while anchored
        changes.pop("BATCH_SIZE", None)
        changes.pop("batch_size", None)
    
    if self.batch_anchor.should_anchor(iteration, self.trace):
        # Before proposing, we need to decide: anchor or not?
        # But proposals already made - fix the config instead
        self.batch_anchor.anchor(self.config)
        # Re-propose with anchored batch (skip if this is first anchor)
        if iteration > 0:
            # Reset LLM context: re-propose with fixed batch
            changes, hypothesis = self._propose(iteration)
            # Strip any batch changes from re-proposal
            changes.pop("BATCH_SIZE", None)
            changes.pop("batch_size", None)
    
    # ... rest of existing method ...
    
    # After recording result
    self.batch_anchor.on_iteration_end(result, self.config)
    
    return result
```

### 5. Integration Points

#### Point 1: `GpuBenchRunner.run_iteration` — Post-proposal, pre-trial
- **When to insert**: After `changes, hypothesis = self._propose(iteration)` but before `self._trial_config(iteration)`
- **What to modify**: Insert the anchoring override block that strips batch changes and possibly re-proposes
- **Side effects to check**: 
  - If re-proposing, we consume 2 LLM calls in one iteration — increase timeout or add flag to re-use first proposal
  - The `_propose` method must handle being called twice in one iteration without state corruption

#### Point 2: `GpuBenchRunner.run_iteration` — After result recording
- **When to insert**: After `self.trace.results.append(result)`
- **What to modify**: Call `self.batch_anchor.on_iteration_end(result, self.config)` to manage anchoring state
- **Side effects to check**: None — purely internal state update

#### Point 3: `SearchConfig.__init__` (or dataclass definition)
- **What to modify**: Add three new fields with sensible defaults
- **Why**: Makes anchoring configurable and testable

#### Point 4: `GpuBenchRunner.run_baseline`
- **What to modify**: Add `self.batch_anchor = BatchAnchorManager(...)` if not already initialized
- **Why**: Ensure baseline run doesn't trigger anchoring (too few iterations)

### Edge Cases and Guardrails

1. **LLM double-call**: Add a `_re_proposing` flag to prevent infinite recursion in `_propose`
2. **Empty changes after stripping**: If all changes were batch-related, return a discard result immediately
3. **Anchoring during baseline**: Explicitly skip anchor check for iteration 0
4. **Thread safety**: `BatchAnchorManager` is single-threaded by design (runner is synchronous)
5. **State reset**: Add `reset()` method to `BatchAnchorManager` for testability

### Test Harness Suggestions
- Unit test `BatchAnchorManager.should_anchor` with synthetic traces showing:
  - 5 iterations of LR-only discards → triggers anchor
  - Mixed changes → no trigger  
  - Less than 5 iterations → no trigger
- Integration test where BATCH_SIZE is varied but LR is stable → anchor should not engage
- E2E test with known stall pattern: anchor engages, batch fixed, LR improves
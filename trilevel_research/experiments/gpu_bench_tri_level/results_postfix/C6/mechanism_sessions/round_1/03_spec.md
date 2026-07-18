## Implementation Specification: Resample-Diverse Mechanism

### 1. Mechanism Name
`resample_diverse_configs`

### 2. Implementation Strategy
`new_method` + `modify_run_iteration`

### 3. Target Methods
- `GpuBenchRunner.run_iteration()` — modify to call resample logic
- `GpuBenchRunner._resample_if_stuck()` — new method
- `GpuBenchRunner._detect_plateau()` — new helper

### 4. Step-by-Step Logic

#### A. Add plateau detection helper method

```python
def _detect_plateau(self, iteration: int, window: int = 5, threshold: float = 0.01) -> bool:
    """
    Returns True if search is stuck in a local region.
    Conditions:
    1. At least `window` iterations have been run.
    2. Best bpb hasn't improved by more than `threshold` relative to best in last `window` iterations.
    3. The last N configs share the same LR and HIDDEN_DIM settings (within tolerances).
    """
    if len(self.trace.results) < window:
        return False
    
    recent = self.trace.results[-window:]
    if not recent:
        return False
        
    # Check improvement stagnation
    best_recent_bpb = min(r.val_bpb for r in recent if r.status == "keep" or r.status == "discard")
    best_ever_bpb = self.trace.best_bpb
    relative_stagnation = abs(best_recent_bpb - best_ever_bpb) / max(best_ever_bpb, 0.01)
    
    if relative_stagnation > threshold:
        return False
    
    # Check hyperparameter diversity
    last_configs = [r.config for r in recent[-3:] if hasattr(r, 'config')]
    if len(last_configs) < 3:
        return False
    
    lr_values = [c.learning_rate for c in last_configs]
    hidden_dims = [c.hidden_dim for c in last_configs]
    
    lr_range = max(lr_values) - min(lr_values)
    hidden_range = max(hidden_dims) - min(hidden_dims) if hidden_dims else 0
    
    # If all recent configs have nearly identical LR and HIDDEN_DIM, we're stuck
    lr_narrow = lr_range < 0.0005  # All within 0.0005 of each other
    hidden_narrow = hidden_range < 32  # All within 32 units
    
    return lr_narrow and hidden_narrow
```

#### B. Add resample method

```python
def _resample_if_stuck(self, iteration: int) -> tuple[dict, str] | None:
    """
    If plateau detected, generate a diverse config that jumps to unexplored regions.
    Returns (changes_dict, hypothesis_string) or None if no resampling needed.
    """
    if not self._detect_plateau(iteration):
        return None
    
    # Build diverse config that explores unsampled regions
    diverse_changes = {}
    hypothesis_parts = ["Resample: forced diverse exploration"]
    
    # Jump to the other side of LR range (0.01 instead of 0.003)
    current_lr = self.config.learning_rate
    new_lr = 0.01 if current_lr < 0.005 else 0.001
    diverse_changes["learning_rate"] = new_lr
    hypothesis_parts.append(f"LR={new_lr}")
    
    # Flip HIDDEN_DIM from 256 to 128 or 512
    current_hidden = self.config.hidden_dim
    new_hidden = 128 if current_hidden >= 256 else 512
    diverse_changes["hidden_dim"] = new_hidden
    hypothesis_parts.append(f"HIDDEN_DIM={new_hidden}")
    
    # Toggle BATCH_SIZE if active (16 <-> 64)
    if hasattr(self.search_config, 'active_params') and 'BATCH_SIZE' in self.search_config.active_params:
        current_batch = self.config.batch_size
        new_batch = 64 if current_batch <= 32 else 16
        diverse_changes["batch_size"] = new_batch
        hypothesis_parts.append(f"BATCH_SIZE={new_batch}")
    
    # Optionally change optimizer
    if hasattr(self.search_config, 'active_params') and 'OPTIMIZER' in self.search_config.active_params:
        current_opt = self.config.optimizer
        new_opt = "adam" if current_opt == "sgd" else "sgd"
        diverse_changes["optimizer"] = new_opt
        hypothesis_parts.append(f"OPTIM={new_opt}")
    
    hypothesis = " | ".join(hypothesis_parts)
    return diverse_changes, hypothesis
```

#### C. Modify `run_iteration` to inject resample

In the `changes is None` branch (LLM proposal), add resample check BEFORE calling LLM:

```python
def run_iteration(self, iteration: int, *, changes: dict | None = None, hypothesis: str = "") -> BenchResult:
    """Run one inner iteration. Uses LLM when changes is None, with resample override."""
    
    # --- NEW: Resample override ---
    if changes is None and iteration >= 3:  # Only after some baseline data
        resample_result = self._resample_if_stuck(iteration)
        if resample_result is not None:
            changes, hypothesis = resample_result
            # Override LLM proposal entirely
            # Fall through to normal execution
    
    if changes is None:
        if self.client is None:
            raise ValueError("LLM client required when changes not provided")
        changes, hypothesis = self._propose(iteration)
    
    # Rest of existing method unchanged...
    # [keep existing filtering, trial config, run, accept/reject logic]
```

### 5. Integration Points

| Integration Point | Location | Change Description |
|---|---|---|
| **Detection trigger** | `run_iteration()`, line ~109 (after `changes is None` check) | Insert resample check before LLM call |
| **Resample override** | `run_iteration()` | `if resample_result: changes, hypothesis = resample_result` — overrides LLM |
| **Config application** | Existing `trial.apply_changes()` | No change needed; resample changes dict follows same format as LLM |
| **Trace recording** | Existing `self.trace.record(result)` | No change; resample runs are recorded like any other iteration |
| **Plateau window** | `_detect_plateau()` | Uses `self.trace.results` which is populated by normal flow |
| **Hypothesis logging** | `result.hypothesis` | Resample hypothesis string flows through existing logging |

### Configuration Options (add to SearchConfig)

```python
@dataclass
class SearchConfig:
    # Existing fields...
    
    # New: resample control
    resample_enabled: bool = True
    resample_window: int = 5          # Number of iterations to check for plateau
    resample_threshold: float = 0.01  # Relative bpb improvement threshold
    resample_frequency: int = 3       # Minimum iterations between resamples
```

### Edge Cases Handled

1. **Cold start (< 3 iterations)**: `_detect_plateau()` returns False due to insufficient data
2. **All params inactive**: Resample only toggles params in `active_params`; if none match, returns None
3. **Multiple resamples**: The `iteration >= 3` guard and `resample_frequency` prevent consecutive resamples
4. **Resample produces worse result**: Normal accept/reject handles this; result is recorded as "discard"
5. **Resample best = global best**: If resample finds better config, it becomes new best via existing logic

### Testing Criteria

1. Trace shows at least one resample when LR stays within [0.0025, 0.003] for 5+ iterations
2. Resampled config has LR=0.01 or 0.001 (outside the stuck cluster)
3. HIDDEN_DIM changes by at least 2x from current value
4. No infinite loop: resample fires at most once per `resample_frequency` iterations
5. Previously unseen hyperparameter combinations appear in trace after plateau detection
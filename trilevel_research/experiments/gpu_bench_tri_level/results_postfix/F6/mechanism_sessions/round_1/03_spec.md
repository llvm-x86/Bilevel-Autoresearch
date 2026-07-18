## Sensitivity Clock Implementation Specification

### 1. Mechanism name (snake_case)
`sensitivity_clock`

### 2. Implementation strategy
**new_helper_class + modify_init**

### 3. Target
- New class: `SensitivityClock`
- Modified: `GpuBenchRunner.__init__`, `GpuBenchRunner.run_iteration`

### 4. Step-by-step logic

#### 4.1 New helper class
```python
class SensitivityClock:
    """Tracks parameter sensitivity and enforces exploration fairness.
    
    The clock maintains a matrix of (parameter, value) → val_bpb outcomes and
    computes per-parameter variance. Unlike naive variance, we sample multiple
    values of the *same parameter* while holding others constant where possible.
    """
    
    def __init__(self, active_params: set[str]):
        # Param name → { value: list of val_bpb results }
        self._param_trials: dict[str, dict[float, list[float]]] = {
            p.upper(): {} for p in active_params
        }
        # Param name → int: times parameter was varied intentionally
        self._variation_count: dict[str, int] = {
            p.upper(): 0 for p in active_params
        }
        # Param name → bool: is this parameter locked?
        self._locked: dict[str, bool] = {
            p.upper(): False for p in active_params
        }
        # Minimum variations before we consider locking
        self._min_variations = 5  # configurable threshold
        # Minimum distinct values needed for variance
        self._min_distinct_values = 3
        # Variance threshold (fraction of best_bpb) to be "sensitive"
        self._sensitivity_ratio = 0.02  # 2% of best_bpb
    
    def record_trial(self, param: str, value: float, val_bpb: float) -> None:
        """Record a single trial outcome for a parameter value."""
        p = param.upper()
        if p not in self._param_trials:
            return
        if value not in self._param_trials[p]:
            self._param_trials[p][value] = []
        self._param_trials[p][value].append(val_bpb)
        self._variation_count[p] += 1
    
    def record_variation(self, param: str) -> None:
        """Track intentional (proposer-requested) variation."""
        p = param.upper()
        if p in self._variation_count:
            self._variation_count[p] += 1
    
    def is_sensitive(self, param: str, best_bpb: float) -> bool | None:
        """Check if a parameter shows sensitivity across its trials.
        
        Returns:
            True if parameter is sensitive (should stay active)
            False if parameter is insensitive (candidate for locking)
            None if insufficient data to determine
        """
        p = param.upper()
        if p not in self._param_trials:
            return None
        
        # Count distinct values tried
        values_tried = len(self._param_trials[p])
        if values_tried < self._min_distinct_values:
            return None  # not enough distinct values
        
        # Count total variations
        if self._variation_count[p] < self._min_variations:
            return None  # not enough total trials
        
        # Compute per-value means and overall range
        value_means = []
        for v, results in self._param_trials[p].items():
            if results:
                value_means.append(sum(results) / len(results))
        
        if len(value_means) < 2:
            return None
        
        # Variance proxy: max - min of value means
        # Adjust for outlier isolation: if any single value shows >10% improvement
        # over the median value mean, it's sensitive regardless of variance
        median_mean = sorted(value_means)[len(value_means) // 2]
        for mean in value_means:
            # Negative means improvement (lower val_bpb is better)
            improvement_ratio = (median_mean - mean) / median_mean
            if improvement_ratio > 0.10:  # 10% improvement
                return True
        
        # Standard variance check: range vs best_bpb
        range_bpb = max(value_means) - min(value_means)
        threshold = best_bpb * self._sensitivity_ratio
        
        return range_bpb > threshold
    
    def lock_insensitive_params(self, best_bpb: float) -> list[str]:
        """Lock parameters that show clear insensitivity.
        
        Returns list of newly locked parameter names.
        """
        newly_locked = []
        for p in list(self._locked.keys()):
            if self._locked[p]:
                continue
            sensitivity = self.is_sensitive(p, best_bpb)
            if sensitivity is False:  # explicitly insensitive
                self._locked[p] = True
                newly_locked.append(p)
        return newly_locked
    
    def is_locked(self, param: str) -> bool:
        return self._locked.get(param.upper(), False)
    
    def count_variations(self, param: str) -> int:
        return self._variation_count.get(param.upper(), 0)
```

#### 4.2 Modified `GpuBenchRunner.__init__`
Add these lines after `self.search_config = search_config or SearchConfig()`:
```python
# Sensitivity clock for exploration fairness
self.sensitivity_clock = SensitivityClock(
    active_params=self.search_config.active_params
)
```

#### 4.3 Modified `GpuBenchRunner.run_iteration`

**Change 1 — Before proposer call** (insert before line `changes, hypothesis = self._propose(iteration)`):
```python
# Check if we need to override proposer with forced exploration
forced_params = self._get_under_explored_params()
if forced_params and iteration > self.trace.best_iteration + 3:
    # We fell behind best iteration; force exploration of under-explored params
    changes, hypothesis = self._force_exploration(iteration, forced_params)
    # Skip normal proposer; use forced changes
    filtered = {k: v for k, v in changes.items() if k.upper() in active}
    # ... continue to trial config
```

**Change 2 — After proposer returns** (after `changes, hypothesis = self._propose(iteration)`):
```python
# Remove any locked params from proposed changes
locked_params = [
    k for k in changes if self.sensitivity_clock.is_locked(k)
]
for param in locked_params:
    del changes[param]

if locked_params:
    hypothesis += f" [locked: {', '.join(locked_params)}]"
```

**Change 3 — After trial result** (after `result = self._run_trial(...)`):
```python
# Record all parameter values from the trial
for param, value in trial_config.to_dict().items():
    if param.upper() in self.search_config.active_params:
        self.sensitivity_clock.record_trial(
            param, value, result.val_bpb
        )
        self.sensitivity_clock.record_variation(param)

# After every 10 iterations, try to lock insensitive params
if iteration > 0 and iteration % 10 == 0:
    newly_locked = self.sensitivity_clock.lock_insensitive_params(
        self.trace.best_bpb
    )
    if newly_locked:
        hypothesis += f" [locked: {', '.join(newly_locked)}]"
```

**Change 4 — New helper methods**:
```python
def _get_under_explored_params(self) -> list[str]:
    """Find params with significantly fewer variations than average."""
    counts = {}
    for p in self.search_config.active_params:
        counts[p] = self.sensitivity_clock.count_variations(p)
    
    if not counts:
        return []
    
    avg_count = sum(counts.values()) / len(counts)
    threshold = avg_count * 0.5  # half the average
    
    return [p for p, c in counts.items() if c < threshold and not self.sensitivity_clock.is_locked(p)]

def _force_exploration(self, iteration: int, params: list[str]) -> tuple[dict, str]:
    """Generate forced parameter changes for under-explored params."""
    changes = {}
    param_descriptions = []
    
    # Use grid-like sampling: vary one param at a time, cycling through values
    for param in params:
        p = param.upper()
        # Try random value within config boundaries, or use specific strategy
        # For simplicity, propose +/-20% from best config
        if p == "HIDDEN_DIM":
            current = self.config.hidden_dim
            # Alternate between larger and smaller
            if iteration % 2 == 0:
                new_val = int(current * 1.2)
            else:
                new_val = int(current * 0.8)
            changes[p] = new_val
            param_descriptions.append(f"{p}={new_val}")
        elif p == "WEIGHT_DECAY":
            # Cycle through predefined values
            values = [0.0, 0.0001, 0.0003]
            idx = self.sensitivity_clock.count_variations(p) % len(values)
            changes[p] = values[idx]
            param_descriptions.append(f"{p}={values[idx]}")
        elif p == "LEARNING_RATE":
            # Allow LR to vary but clamp to prevent explosion
            values = [1e-3, 1e-4, 5e-4]
            idx = self.sensitivity_clock.count_variations(p) % len(values)
            changes[p] = values[idx]
            param_descriptions.append(f"{p}={values[idx]}")
    
    hypothesis = f"Forced exploration of under-explored params: {', '.join(param_descriptions)}"
    return changes, hypothesis
```

### 5. Integration points

#### 5.1 Probe points in existing flow
```python
# In run_iteration, after proposer but before trial:
#   1. Filter locked params from changes
#   2. Check for forced exploration override

# After trial execution:
#   3. Record all param values in sensitivity clock
#   4. Periodic lock check (every 10 iterations)
```

#### 5.2 Control flow changes
```
Normal flow (no override):
  1. Proposer generates changes
  2. Filter locked params → changes reduced
  3. Run trial
  4. Record in clock
  5. Every 10th iteration: check locks

Override flow (under-explored params detected):
  1. _get_under_explored_params() returns non-empty
  2. _force_exploration() bypasses proposer entirely
  3. Run forced trial
  4. Record in clock
  5. Skipped if within 3 iterations of best (avoid thrashing)
```

#### 5.3 Edge cases
- **First 10 iterations**: All params show `None` sensitivity; no locking occurs
- **Locked after 10 iterations**: Subsequent proposer calls get filtered; forced exploration may still try locked params if new evidence emerges
- **All params locked**: Clock stops interfering; proposer runs unfiltered
- **Best iteration regression**: Forced exploration suppresses its own override within 3 iterations of best to avoid disrupting promising configurations
- **Duplicate values**: Clock tracks per-value result lists; multiple trials of same value increase confidence but don't affect distinct value count

#### 5.4 Validation
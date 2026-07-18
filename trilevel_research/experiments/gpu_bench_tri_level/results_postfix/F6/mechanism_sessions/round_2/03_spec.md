## Implementation Specification for Hypothesis 4: Robustification Through Conditional Warmup Reset

### 1. Mechanism name (snake_case)
`conditional_warmup_reset`

### 2. Implementation strategy
`modify_init` + `new_method`

### 3. Target
- `GpuBenchRunner.__init__` — add new state tracking fields
- `GpuBenchRunner._run_trial` — add warmup reset logic as a wrapper/guard
- `GpuBenchRunner.run_iteration` — add divergence detection before trial execution

### 4. Step-by-step logic

**A. New state fields in `__init__` (after existing self.trace initialization):**

```python
# Warmup reset configuration
self.warmup_steps = 50  # extra steps when reset is triggered
self.divergence_threshold_multiplier = 3.0  # multiplier over recent moving average
self.recent_val_bpb_window = []  # sliding window of last N accepted val_bpb values
self.recent_window_size = 5  # number of recent values to track
self.consecutive_divergences = 0  # count of consecutive divergence failures
self.max_consecutive_divergences = 3  # threshold to skip warmup and discard immediately
```

**B. New method `_detect_divergence(self, val_bpb: float) -> bool`:**

1. If `self.recent_val_bpb_window` is empty (e.g., first trial), return `False` (no baseline to compare against)
2. Compute `moving_average = mean(self.recent_val_bpb_window[-self.recent_window_size:])`
3. Compute `threshold = moving_average * self.divergence_threshold_multiplier`
4. Return `val_bpb > threshold`

- **Special case**: If `val_bpb` is `float('inf')` or `float('nan')`, return `True` immediately (unambiguous divergence)

- **Edge case**: If `moving_average` is very small (< 1e-6) to avoid division issues, use `max(moving_average, 1e-6)`

**C. New method `_try_warmup_reset(self, trial_config: GpuBenchConfig, iteration: int, hypothesis: str) -> BenchResult | None`:**

1. Only apply warmup if `self.consecutive_divergences < self.max_consecutive_divergences`
   - If `>= max`, return `None` (caller will discard immediately)
2. Create a copy of the trial config (or construct a new config based on it)
3. Set `learning_rate = learning_rate * 0.1` (aggressive reduction for warmup stability)
4. Optionally set `warmup_steps = self.warmup_steps` if the config supports it
5. Run the trial with this modified config using `self._run_trial(trial_config, iteration=..., hypothesis=f"{hypothesis}_warmup")`
6. Return the result

**Alternative (simpler)**: Instead of modifying the config, just run the original trial configuration for `self.warmup_steps` additional steps. This requires a new parameter in `_run_trial` or a wrapper:

- Add `extra_steps: int = 0` parameter to `_run_trial` (default `0`)
- Inside `_run_trial`, if `extra_steps > 0`, run the model for `extra_steps` additional iterations (or epochs) before evaluating
- For simplicity, since we can't modify the binary: just re-run the trial with a modified config that has `training_steps += warmup_steps`

**Simpler approach chosen**: Create a new trial config with `training_steps` increased by `warmup_steps`, OR if `training_steps` doesn't exist, just re-run the same config (the warmup is already in the binary, we're just giving it more steps to recover).

**D. Modified `run_iteration` method (after `result = self._run_trial(...)`):**

Insert after line `result = self._run_trial(trial, iteration=iteration, hypothesis=hypothesis)` and before `result.changes = filtered`:

```python
# Divergence detection and warmup reset
if self._detect_divergence(result.val_bpb):
    result.status = "diverged"
    result.accepted = False
    self.consecutive_divergences += 1
    
    # Attempt warmup reset if not past max consecutive divergences
    warmup_result = self._try_warmup_reset(trial, iteration, hypothesis)
    if warmup_result is not None and warmup_result.val_bpb < self.trace.best_bpb:
        # Warmup reset succeeded
        self.consecutive_divergences = 0  # reset counter on success
        warmup_result.status = "keep"
        warmup_result.accepted = True
        warmup_result.changes = filtered
        warmup_result.hypothesis = f"{hypothesis}_warmup_reset"
        
        # Update best config
        self.trace.best_val_bpb = warmup_result.val_bpb
        self.trace.best_bpb = warmup_result.val_bpb
        self.trace.best_iteration = iteration
        self.trace.best_config = trial  # original config, not warmed up version
        self.config = trial
        
        # Update recent window with the successful value
        self.recent_val_bpb_window.append(warmup_result.val_bpb)
        if len(self.recent_val_bpb_window) > self.recent_window_size:
            self.recent_val_bpb_window.pop(0)
        
        self.trace.record(warmup_result)
        self.trace.results.append(warmup_result)
        return warmup_result
    else:
        # Warmup failed or not applied, discard
        if warmup_result is not None:
            warmup_result.status = "discard"
            warmup_result.accepted = False
            self.trace.record(warmup_result)
            self.trace.results.append(warmup_result)
        result.status = "discard"
        result.accepted = False
        # Don't update recent window with diverged value
        self.trace.record(result)
        self.trace.results.append(result)
        return result

# Normal path (no divergence)
self.consecutive_divergences = 0  # reset counter on success
self.recent_val_bpb_window.append(result.val_bpb)
if len(self.recent_val_bpb_window) > self.recent_window_size:
    self.recent_val_bpb_window.pop(0)
```

**E. Additional guard in `run_baseline`:**

After `result = self._run_trial(self.config, iteration=0, hypothesis="baseline")` and before `result.status = "keep"`:

```python
# Baseline must be non-divergent
if result.val_bpb == float('inf') or result.val_bpb == float('nan') or result.status == "crash":
    # Attempt baseline with warmup
    warmup_config = GpuBenchConfig()
    warmup_config.learning_rate = self.config.learning_rate * 0.1
    result = self._run_trial(warmup_config, iteration=0, hypothesis="baseline_warmup")
    if result.val_bpb == float('inf') or result.val_bpb == float('nan') or result.status == "crash":
        raise RuntimeError("Baseline cannot be established even with warmup reset")
    self.config = warmup_config  # use warmed-up config as base

# Initialize recent window with baseline value
self.recent_val_bpb_window = [result.val_bpb]
```

### 5. Integration points

| Integration Point | Changes Required | Risk Level |
|---|---|---|
| `__init__` | Add 5 new state fields after `self.trace.results = []` | Low — no existing behavior changes |
| `run_baseline()` | Add divergence guard + warmup fallback + initialize `recent_val_bpb_window` | Medium — baseline must be guaranteed stable |
| `run_iteration()` | Insert divergence detection after trial execution, before acceptance logic | Medium — modifies control flow; must ensure existing "keep"/"discard" logic still works |
| `_run_trial()` | No changes needed if using modified config approach OR add optional `extra_steps` parameter | Low — backwards compatible |

**Key interaction with existing `run_iteration` logic**: The divergence detection must run **after** the trial result is obtained but **before** the existing `if result.status == "crash":` / `elif result.val_bpb < self.trace.best_bpb:` chain. This requires careful insertion at the exact point shown in step D.

**Trace recording**: Both the original diverged result and the warmup attempt result are recorded in `self.trace.results`, allowing full audit of the recovery attempt.

**Dependency on `_trial_config`**: The warmup reset uses the same method to create a trial config, ensuring consistency with how other configs are constructed.
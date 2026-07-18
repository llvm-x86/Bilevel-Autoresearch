## Implementation Specification for Adaptive Perturbation

**1. Mechanism name (snake_case):**
`adaptive_perturbation`

**2. Implementation strategy:**
`new_helper_class` + `modify_init` + `replace_method`

**3. Target:**
- New helper class: `AdaptivePerturbationController`
- Modified: `GpuBenchRunner.__init__` (add controller attribute)
- Replaced logic: `GpuBenchRunner._propose` (internal proposal logic after LLM returns suggestions)

**4. Step-by-step logic:**

### A. New Helper Class: `AdaptivePerturbationController`

```python
class AdaptivePerturbationController:
    """Controls adaptive perturbation magnitude based on iteration history."""

    def __init__(
        self,
        initial_perturbation_pct: float = 0.50,
        min_perturbation_pct: float = 0.05,
        max_perturbation_pct: float = 0.75,
        shrink_factor_keep: float = 0.85,
        grow_factor_discard: float = 0.50,
        recent_window: int = 3,  # number of recent iterations to consider
        oscillation_penalty: float = 0.10,  # additional shrink when direction reverses
    ):
        self.current_pct = initial_perturbation_pct
        self.min_pct = min_perturbation_pct
        self.max_pct = max_perturbation_pct
        self.shrink_factor_keep = shrink_factor_keep
        self.grow_factor_discard = grow_factor_discard  # <= 1.0, likely < 1.0
        self.recent_window = recent_window
        self.oscillation_penalty = oscillation_penalty
        # track recent proposals by parameter for oscillation detection
        self.param_directions: dict[str, list[float]] = {}  # param -> list of signed changes (-1, 0, +1)
        self.last_perturbation_info: dict[str, dict] = {}  # for debug logging
```

### B. Methods on the controller:

**B1. `get_perturbation_pct(self, param_name: str, best_config: GpuBenchConfig) -> float`**
```python
def get_perturbation_pct(self, param_name: str, best_config: GpuBenchConfig) -> float:
    """Return the perturbation magnitude as a fraction of the param's current range."""
    # Base pct from global state
    pct = self.current_pct

    # Retrieve the param's bounds
    param_range = self._get_param_range(param_name)  # helper computing max - min from config's allowed range

    # If param is at an edge (near min or max), shrink perturbation
    if param_range > 0:
        current_val = getattr(best_config, param_name.lower())
        min_val, max_val = self._get_param_bounds(param_name)
        edge_distance = min(current_val - min_val, max_val - current_val) / param_range
        if edge_distance < 0.15:  # within 15% of edge
            pct *= 0.5  # halve perturbation to avoid repeated edge slamming

    return max(self.min_pct, min(pct, self.max_pct))
```

**B2. `update_after_iteration(self, iteration: int, result: BenchResult, proposed_changes: dict[str, tuple], best_config: GpuBenchConfig)`**
```python
def update_after_iteration(self, iteration: int, result: BenchResult,
                          proposed_changes: dict[str, tuple],
                          best_config: GpuBenchConfig):
    """Update perturbation magnitude based on result and direction history."""
    if not proposed_changes:
        return

    # Determine directional signs for this iteration's changes
    for param, (old_val, proposed_val) in proposed_changes.items():
        if old_val is not None and proposed_val is not None:
            direction = 1 if proposed_val > old_val else (-1 if proposed_val < old_val else 0)
            if param not in self.param_directions:
                self.param_directions[param] = []
            self.param_directions[param].append(direction)

    # Check for oscillation (direction reversal across iterations)
    oscillation_count = 0
    for param in proposed_changes:
        recent = self.param_directions.get(param, [])[-self.recent_window:]
        if len(recent) >= 2:
            # Count sign changes (0 is neutral, ignore)
            non_zero = [d for d in recent if d != 0]
            sign_changes = sum(1 for i in range(1, len(non_zero)) if non_zero[i] != non_zero[i-1])
            if sign_changes >= 2:  # oscillated at least twice
                oscillation_count += 1

    # Update global perturbation magnitude
    if result.accepted and result.status == "keep":
        # Successful iteration → shrink perturbation (we're close to optimum)
        self.current_pct *= self.shrink_factor_keep
    else:
        # Failed or discarded iteration → might need smaller steps, but could also grow
        # If oscillation detected, shrink more aggressively
        if oscillation_count >= 1:
            self.current_pct *= (1.0 - self.oscillation_penalty)
        else:
            # Discard without oscillation: param change was wrong direction, shrink modestly
            self.current_pct *= self.grow_factor_discard  # still < 1.0 (shrink)

    # Clamp
    self.current_pct = max(self.min_pct, min(self.current_pct, self.max_pct))
```

### C. Modified `GpuBenchRunner.__init__`:

Add after existing init lines:
```python
self.perturbation_controller = AdaptivePerturbationController(
    initial_perturbation_pct=self.search_config.initial_perturbation_pct if hasattr(self.search_config, 'initial_perturbation_pct') else 0.50,
    # other params from search_config or defaults
)
```

### D. Logic changes in `run_iteration`:

Replace the entire section after `result.changes = filtered` with adaptive logic:

```python
# --- Store proposed changes BEFORE result assessment ---
proposed_changes: dict[str, tuple] = {}
for k, v in filtered.items():
    old_val = getattr(self.config, k.lower(), None)
    proposed_changes[k] = (old_val, v)

# --- Run trial (existing code) ---
trial = self._trial_config(iteration)
trial.apply_changes(filtered)
result = self._run_trial(trial, iteration=iteration, hypothesis=hypothesis)
result.changes = filtered

# --- Update controller BEFORE acceptance decision ---
self.perturbation_controller.update_after_iteration(
    iteration, result, proposed_changes, self.config
)

# --- Existing acceptance logic ---
if result.status == "crash":
    result.accepted = False
elif result.val_bpb < self.trace.best_bpb:
    result.status = "keep"
    result.accepted = True
    self.trace.best_val_bpb = result.val_bpb
    self.trace.best_bpb = result.val_bpb
    self.trace.best_iteration = iteration
    self.trace.best_config = trial
    self.config = trial
else:
    result.status = "discard"
    result.accepted = False

self.trace.record(result)
self.trace.results.append(result)
return result
```

### E. Modified `_propose` method (or `_apply_changes`):

When applying LLM-suggested changes, scale the perturbation magnitude:

```python
def _apply_adaptive_perturbation(self, param_name: str, current_value: Any, suggested_value: Any, pert_pct: float) -> Any:
    """Adjust suggested value by clamping it within adaptive perturbation bounds."""
    if not isinstance(current_value, (int, float)) or not isinstance(suggested_value, (int, float)):
        return suggested_value

    param_range = self.perturbation_controller._get_param_range(param_name)
    if param_range == 0:
        return suggested_value

    max_delta = param_range * pert_pct
    actual_delta = suggested_value - current_value

    # Clamp the change magnitude
    if abs(actual_delta) > max_delta:
        clamped_delta = math.copysign(max_delta, actual_delta)
        new_val = current_value + clamped_delta
        # Round for integer params
        if isinstance(current_value, int):
            new_val = int(round(new_val))
        return new_val

    return suggested_value
```

**5. Integration points:**

### A. `SearchConfig` class (optional enhancement):
```python
@dataclass
class SearchConfig:
    # ... existing fields ...
    initial_perturbation_pct: float = 0.50
    adaptive_perturbation_enabled: bool = True
    perturbation_min_pct: float = 0.05
    perturbation_max_pct: float = 0.75
    perturbation_shrink_keep: float = 0.85
    perturbation_grow_discard: float = 0.50
```

### B. `BenchTrace` extension (optional, for debugging):
```python
@dataclass
class PerturbationInfo:
    iteration: int
    perturbation_pct: float
    oscillation_detected: bool
    param_directions: dict[str, list[int]]

# Add to BenchTrace:
# perturbation_history: list[PerturbationInfo] = field(default_factory=list)
```

### C. Edge cases handled:

1. **Integer parameters**: The clamping logic automatically rounds, ensuring integer params stay integer.
2. **Edge-bound parameters**: Controller halves perturbation when param is near its boundary, preventing repeated attempts to go out of bounds.
3. **First iteration**: Controller starts with conservative default (0.50), avoiding extreme jumps initially.
4. **Multiple parameter changes**: Oscillation detection considers all changed params, not just one.
5. **Crash handling**: Crashes count as failures, causing perturbation to shrink (avoiding further crashes from extreme values).
6. **Static parameters**: If a param's range is 0 (fixed), perturbation is ignored entirely.
## Implementation Specification: Local Perturbation Surrogate

### 1. Mechanism name
`local_perturbation_surrogate`

### 2. Implementation strategy
`new_helper_class` + `modify_init` + `modify_run_iteration`

### 3. Target
- `GpuBenchRunner.__init__` — add surrogate state
- `GpuBenchRunner.run_iteration` — intercept proposals near best point

### 4. Step-by-step logic

#### A. New helper class: `PerturbationSurrogate`
Create a class that wraps the LLM proposal mechanism for local search.

```python
class PerturbationSurrogate:
    def __init__(
        self,
        radius_initial: float = 0.15,       # fraction of parameter range
        radius_min: float = 0.05,            # floor to prevent collapse
        step_size_grid: dict[str, int] | None = None,  # {param_name: grid_step}
    ):
        self.radius = radius_initial
        self.radius_min = radius_min
        self.step_size_grid = step_size_grid or {
            "HIDDEN_DIM": 64,
            "NUM_LAYERS": 1,
            "LEARNING_RATE": 0,  # continuous, no rounding
        }
        self.active = True

    def propose(
        self,
        best_config: GpuBenchConfig,
        active_params: set[str],
        iteration: int,
    ) -> tuple[dict[str, Any], str]:
        """Generate perturbed proposal around best_config."""
        changes = {}
        hypothesis_parts = [f"local_perturbation(radius={self.radius:.3f})"]

        for param_name in sorted(active_params):
            param_key = param_name.upper()
            current_val = getattr(best_config, param_key, None)
            if current_val is None:
                continue

            grid = self.step_size_grid.get(param_key, 0)
            if isinstance(current_val, int) and grid > 0:
                # Discrete parameter with grid stepping
                noise = random.gauss(0, self.radius * current_val)
                new_val = round((current_val + noise) / grid) * grid
                new_val = max(int(min(param_key)), min(int(max(param_key)), new_val))
                new_val = max(1, new_val)  # ensure positive
            elif isinstance(current_val, float):
                # Continuous parameter (e.g., learning_rate)
                log_val = math.log(current_val) if current_val > 0 else -10
                noise = random.gauss(0, self.radius)
                new_log = log_val + noise
                new_val = math.exp(new_log)
                # Capping to avoid extreme values
                new_val = max(1e-6, min(1.0, new_val))
            else:
                continue

            if new_val != current_val:
                changes[param_key] = new_val
                hypothesis_parts.append(f"{param_key}={current_val}->{new_val}")

        if not changes:
            # Fallback: nudge hidden_dim by at least one grid step
            if "HIDDEN_DIM" in active_params:
                step = self.step_size_grid.get("HIDDEN_DIM", 64)
                new_dim = getattr(best_config, "HIDDEN_DIM", 384) + step
                changes["HIDDEN_DIM"] = new_dim
                hypothesis_parts.append(f"HIDDEN_DIM=384->{new_dim}")

        hypothesis = " | ".join(hypothesis_parts)
        return changes, hypothesis

    def adjust_radius(self, result: BenchResult) -> None:
        """Shrink radius on consecutive failures, expand on success."""
        if result.status == "keep":
            # Successful: slightly expand to explore more
            self.radius = min(self.radius * 1.1, 0.30)
        elif result.status == "discard":
            # Failure: shrink to focus more locally
            self.radius = max(self.radius * 0.9, self.radius_min)
```

#### B. Modify `GpuBenchRunner.__init__`
Add surrogate initialization after existing attribute setup.

```python
def __init__(self, ...):
    # ... existing code ...
    
    # Surrogate for local perturbation
    self.perturbation_surrogate = PerturbationSurrogate(
        radius_initial=0.15,
        radius_min=0.05,
        step_size_grid={
            "HIDDEN_DIM": 64,
            "NUM_LAYERS": 1,
            "LEARNING_RATE": 0,
        }
    )
    self.perturbation_active = False  # activated after first good config
```

#### C. Modify `GpuBenchRunner.run_iteration`
Add logic before the LLM proposal call.

```python
def run_iteration(
    self,
    iteration: int,
    *,
    changes: dict | None = None,
    hypothesis: str = "",
) -> BenchResult:
    # === NEW: Surrogate intercept ===
    if changes is None and self.perturbation_active and iteration > 2:
        # After iteration 2, start using surrogate near best point
        surrogate_changes, surrogate_hypothesis = self.perturbation_surrogate.propose(
            best_config=self.trace.best_config,
            active_params=self.search_config.active_params,
            iteration=iteration,
        )
        if surrogate_changes:
            changes = surrogate_changes
            hypothesis = surrogate_hypothesis
            # Use surrogate proposal directly
            filtered = {k: v for k, v in changes.items() if k.upper() in self.search_config.active_params}
            if filtered:
                trial = self._trial_config(iteration)
                trial.apply_changes(filtered)
                result = self._run_trial(trial, iteration=iteration, hypothesis=hypothesis)
                result.changes = filtered
                
                # Update radius based on outcome
                self.perturbation_surrogate.adjust_radius(result)
                
                # Record and return
                if result.status == "crash":
                    result.accepted = False
                elif result.val_bpb < self.trace.best_bpb:
                    result.status = "keep"
                    result.accepted = True
                    self._update_best(result, trial, iteration)
                else:
                    result.status = "discard"
                    result.accepted = False
                
                self.trace.record(result)
                self.trace.results.append(result)
                return result
    
    # === Activate surrogate after first keep ===
    if self.trace.best_iteration > 0 and not self.perturbation_active:
        self.perturbation_active = True
        if self.client:
            # Inform LLM that we're switching to local mode
            self.perturbation_hint = True
    
    # === Original code (unchanged) ===
    if changes is None:
        if self.client is None:
            raise ValueError(...)
        changes, hypothesis = self._propose(iteration)
    
    # ... rest of original run_iteration logic ...
```

#### D. Add `_update_best` helper method (optional refactor)
Extract the "accept as new best" logic to avoid duplication.

```python
def _update_best(self, result: BenchResult, trial: GpuBenchConfig, iteration: int) -> None:
    """Update trace with new best configuration."""
    result.status = "keep"
    self.trace.best_val_bpb = result.val_bpb
    self.trace.best_bpb = result.val_bpb
    self.trace.best_iteration = iteration
    self.trace.best_config = trial
    self.config = trial
    # Reset surrogate radius on significant improvement
    if self.perturbation_active:
        self.perturbation_surrogate.radius = max(
            self.perturbation_surrogate.radius * 1.2,
            0.10
        )
```

### 5. Integration points

1. **Injection point**: Before `self._propose(iteration)` call in `run_iteration`. The surrogate intercept must check `self.perturbation_active` first.

2. **State initialization**: After `self.config = GpuBenchConfig()` in `__init__`, add surrogate creation. This ensures the surrogate is available immediately.

3. **Radius persistence**: The surrogate's adaptive radius is automatically adjusted via `adjust_radius()` after each trial. The radius shrinks on discard (focus more) and expands on keep (explore more).

4. **Activation trigger**: `perturbation_active` flips to `True` as soon as `self.trace.best_iteration > 0` (first keep). This ensures we don't perturb the baseline but start local search after the first improvement.

5. **Fallback mechanism**: If surrogate produces no changes (e.g., all params hit boundaries), the code falls through to original LLM proposal as backup.

6. **Debug logging**: Add to result's `hypothesis` field a marker like `"local_perturbation(radius=0.15)"` so trace analysis can identify surrogate-generated trials.

### Complexity scoring update

- **Impact**: 4 (directly addresses search step aggressiveness)
- **Feasibility**: 4 (simple class, no optimizer state to reset)
- **Complexity**: 3 (stateful radius adaptation adds moderate complexity)
- **Final**: 4 × 4 ÷ 3 = **5.3**
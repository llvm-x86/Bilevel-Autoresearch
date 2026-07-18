## Implementation Specification: Adaptive Perturbation Magnitude Scaling

**1. Mechanism name**: `adaptive_perturbation_scaling`

**2. Implementation strategy**: `new_helper_class` + `modify_init`

**3. Target**: 
- New class: `PerturbationController`
- Modify: `GpuBenchRunner.__init__`, `GpuBenchRunner.run_iteration`

**4. Step-by-step logic**:

### New class: `PerturbationController`

```python
@dataclass
class PerturbationController:
    """Tracks perturbation history and adapts magnitude dynamically."""
    
    # Configuration
    base_magnitude: float = 0.1  # Starting perturbation scale (fraction of param range)
    min_magnitude: float = 0.01
    max_magnitude: float = 0.5
    stuck_window: int = 3  # Consecutive non-improving iterations before scaling up
    overshoot_window: int = 2  # Consecutive improvements below threshold before scaling down
    improvement_threshold: float = 0.01  # Minimum relative improvement to avoid "stuck" classification
    oscillation_window: int = 4  # Number of recent results to check for oscillation
    
    # State
    consecutive_no_improvement: int = 0
    consecutive_small_improvement: int = 0
    current_magnitude: float = 0.1
    last_val_bpb: float | None = None
    recent_improvements: list[float] = field(default_factory=list)
    
    def update(self, current_val_bpb: float) -> float:
        """Update internal state based on new result and return new magnitude."""
        if self.last_val_bpb is None:
            self.last_val_bpb = current_val_bpb
            return self.current_magnitude
        
        # Calculate improvement (negative means better)
        improvement = self.last_val_bpb - current_val_bpb
        relative_improvement = improvement / max(self.last_val_bpb, 1e-10)
        
        # Track recent improvements for oscillation detection
        self.recent_improvements.append(relative_improvement)
        if len(self.recent_improvements) > self.oscillation_window:
            self.recent_improvements.pop(0)
        
        # Check for oscillation (alternating signs)
        if len(self.recent_improvements) >= 3:
            # Oscillation detected: improvements alternate sign
            signs = [1 if x > 0 else -1 for x in self.recent_improvements[-3:]]
            if signs[0] == signs[2] and signs[0] != signs[1]:
                # Oscillation pattern: + - + or - + -
                # Reduce magnitude to dampen oscillation
                self.current_magnitude = max(
                    self.current_magnitude * 0.7, 
                    self.min_magnitude
                )
                self.consecutive_no_improvement = 0
                self.consecutive_small_improvement = 0
                self.last_val_bpb = current_val_bpb
                return self.current_magnitude
        
        # Check for stuck pattern (no significant improvement)
        if relative_improvement < self.improvement_threshold:
            self.consecutive_no_improvement += 1
            self.consecutive_small_improvement = 0
        elif relative_improvement < 0.05:  # Small but positive improvement
            self.consecutive_no_improvement = 0
            self.consecutive_small_improvement += 1
        else:  # Significant improvement
            self.consecutive_no_improvement = 0
            self.consecutive_small_improvement = 0
        
        # Adjust magnitude based on stuck detection
        if self.consecutive_no_improvement >= self.stuck_window:
            # Scale up to escape local minima
            self.current_magnitude = min(
                self.current_magnitude * 1.5,
                self.max_magnitude
            )
            self.consecutive_no_improvement = 0
            self.consecutive_small_improvement = 0
        
        # Adjust magnitude based on overshoot detection
        if self.consecutive_small_improvement >= self.overshoot_window:
            # Scale down for finer exploration
            self.current_magnitude = max(
                self.current_magnitude * 0.8,
                self.min_magnitude
            )
            self.consecutive_small_improvement = 0
        
        self.last_val_bpb = current_val_bpb
        return self.current_magnitude
    
    def get_perturbed_value(self, current_value: float, param_range: tuple[float, float]) -> float:
        """Generate perturbed value using current magnitude."""
        lower, upper = param_range
        param_range_size = upper - lower
        
        # Calculate max allowed step based on magnitude and range
        max_step = self.current_magnitude * param_range_size
        
        # Generate random perturbation within bounds
        perturbation = random.uniform(-max_step, max_step)
        
        # Clip to avoid boundary clipping artifacts
        new_value = current_value + perturbation
        new_value = max(lower * 1.05, min(upper * 0.95, new_value))  # Stay away from edges
        
        return new_value
```

### Modify `GpuBenchRunner.__init__`:

Add after `self.simple_mode = simple_mode`:
```python
self.perturbation_controller = PerturbationController(
    base_magnitude=search_config.perturb_base_magnitude if hasattr(search_config, 'perturb_base_magnitude') else 0.1,
    min_magnitude=search_config.perturb_min_magnitude if hasattr(search_config, 'perturb_min_magnitude') else 0.01,
    max_magnitude=search_config.perturb_max_magnitude if hasattr(search_config, 'perturb_max_magnitude') else 0.5,
)
```

### Modify `GpuBenchRunner.run_iteration`:

Replace the current acceptance logic block (starting at `if result.status == "crash":`) with:

```python
if result.status == "crash":
    result.accepted = False
else:
    # Update perturbation magnitude based on result
    new_magnitude = self.perturbation_controller.update(result.val_bpb)
    
    if result.val_bpb < self.trace.best_bpb:
        result.status = "keep"
        result.accepted = True
        self.trace.best_val_bpb = result.val_bpb
        self.trace.best_bpb = result.val_bpb
        self.trace.best_iteration = iteration
        self.trace.best_config = trial
        self.config = trial
        
        # Log magnitude for debugging
        result.metadata["perturbation_magnitude"] = new_magnitude
    else:
        result.status = "discard"
        result.accepted = False
        result.metadata["perturbation_magnitude"] = new_magnitude
```

Additionally, modify how changes are generated in the LLM proposal path. After `changes, hypothesis = self._propose(iteration)`, add perturbation scaling:

```python
if changes is None:
    if self.client is None:
        raise ValueError("LLM client required when changes not provided")
    changes, hypothesis = self._propose(iteration)
    
    # Apply adaptive perturbation to continuous parameters
    for param_name in ['HIDDEN_DIM', 'LR', 'BATCH_SIZE']:
        if param_name in changes:
            # Get current value and valid range
            current_val = getattr(self.config, param_name.lower())
            param_range = self.search_config.param_ranges.get(param_name, (0.1, 1000))
            
            # Scale the proposed change by current magnitude
            if isinstance(changes[param_name], (int, float)):
                relative_change = abs(changes[param_name] - current_val) / max(current_val, 1)
                max_relative_change = self.perturbation_controller.current_magnitude
                
                if relative_change > max_relative_change:
                    # Clamp change to current magnitude
                    direction = 1 if changes[param_name] > current_val else -1
                    clamped_value = current_val * (1 + direction * max_relative_change)
                    if isinstance(changes[param_name], int):
                        changes[param_name] = int(clamped_value)
                    else:
                        changes[param_name] = clamped_value
```

**5. Integration points**:

1. **SearchConfig extension**: Add optional fields for perturbation controller configuration:
   ```python
   @dataclass
   class SearchConfig:
       # ... existing fields ...
       perturb_base_magnitude: float = 0.1
       perturb_min_magnitude: float = 0.01
       perturb_max_magnitude: float = 0.5
   ```

2. **BenchResult metadata**: Ensure `BenchResult` has a `metadata` dict field for logging:
   ```python
   @dataclass
   class BenchResult:
       # ... existing fields ...
       metadata: dict = field(default_factory=dict)
   ```

3. **Trace reporting**: Add perturbation magnitude to trace output for monitoring:
   ```python
   # In trace reporting section
   if hasattr(self, 'perturbation_controller'):
       trace_data['perturbation_magnitude'] = self.perturbation_controller.current_magnitude
   ```

4. **Reset on hypothesis change**: When hypothesis changes (detected via `hypothesis` parameter), reset perturbation controller:
   ```python
   def run_iteration(self, iteration: int, *, changes: dict | None = None, hypothesis: str = ""):
       # Detect hypothesis change from trace
       if self.trace.results and self.trace.results[-1].hypothesis != hypothesis:
           self.perturbation_controller = PerturbationController(
               base_magnitude=self.search_config.perturb_base_magnitude,
               min_magnitude=self.search_config.perturb_min_magnitude,
               max_magnitude=self.search_config.perturb_max_magnitude,
           )
   ```

5. **Failure mode detection**: Add monitoring for oscillation detection:
   ```python
   # After running perturbation_controller.update()
   if len(self.perturbation_controller.recent_improvements) >= 4:
       oscillation_count = sum(
           1 for i in range(1, len(self.perturbation_controller.recent_improvements))
           if self.perturbation_controller.recent_improvements[i] * 
              self.perturbation_controller.recent_improvements[i-1] < 0
       )
       if oscillation_count >= 3:
           logger.warning(
               f"Oscillation detected in iteration {iteration}, "
               f"magnitude reduced to {self.perturbation_controller.current_magnitude:.4f}"
           )
   ```
## Implementation Specification: Multi-Axis Random Subspace Sampling

### 1. Mechanism Name
`multi_axis_subspace_sampling`

### 2. Implementation Strategy
**new_helper_class** + **modify_init** + **modify_run_iteration**

### 3. Target
- New class: `MultiAxisSampler`
- Modified class: `GpuBenchRunner`
- Modified method: `GpuBenchRunner.run_iteration`

### 4. Step-by-Step Logic

#### 4.1 New Helper Class: `MultiAxisSampler`

```python
class MultiAxisSampler:
    """
    Generates multi-axis perturbations by randomly selecting multiple parameters
    and scaling their perturbation magnitudes inversely to their observed sensitivity.
    """
    
    def __init__(
        self,
        active_params: set[str],
        perturbation_scale: float = 0.1,  # Base perturbation as fraction of default
        history_window: int = 10,  # Recent iterations to track for sensitivity
        sample_size: int = 3,  # Number of axes to perturb simultaneously
        p_random: float = 0.3,  # Probability of fully random multi-axis vs guided
    ):
        self.active_params = active_params
        self.perturbation_scale = perturbation_scale
        self.history_window = history_window
        self.sample_size = min(sample_size, len(active_params))
        self.p_random = p_random
        
        # Track sensitivity per parameter
        self.param_history: dict[str, list[float]] = {p: [] for p in active_params}
        self.param_sensitivity: dict[str, float] = {p: 1.0 for p in active_params}
        
    def propose(self, current_config: GpuBenchConfig) -> dict:
        """
        Generate multi-axis perturbation proposal relative to current_config.
        Returns dict of changes to apply.
        """
        # Decide: random exploration or guided by sensitivity
        if random.random() < self.p_random:
            return self._random_proposal(current_config)
        else:
            return self._guided_proposal(current_config)
    
    def _random_proposal(self, config: GpuBenchConfig) -> dict:
        """Randomly select sample_size parameters and perturb them."""
        selected = random.sample(list(self.active_params), self.sample_size)
        changes = {}
        for param in selected:
            changes[param] = self._perturb_param(config, param)
        return changes
    
    def _guided_proposal(self, config: GpuBenchConfig) -> dict:
        """
        Select parameters with low sensitivity (more freedom to change)
        and perturb them with magnitude inversely proportional to sensitivity.
        """
        # Sort params by sensitivity (ascending = most sensitive first)
        sorted_params = sorted(
            self.active_params,
            key=lambda p: self.param_sensitivity.get(p, 1.0)
        )
        
        # Weight selection inversely to sensitivity
        sensitivities = [self.param_sensitivity[p] for p in sorted_params]
        total_sens = sum(sensitivities)
        # Inverse weights: less sensitive params get higher selection probability
        weights = [total_sens / s if s > 0 else 1.0 for s in sensitivities]
        
        selected = random.choices(
            sorted_params, 
            weights=weights, 
            k=self.sample_size
        )
        
        changes = {}
        for param in selected:
            # Larger perturbation for less sensitive params
            sensitivity = self.param_sensitivity[param]
            scale_factor = 1.0 / max(sensitivity, 0.1)  # Clamp to avoid explosion
            changes[param] = self._perturb_param(config, param, scale_factor)
        
        return changes
    
    def _perturb_param(
        self, 
        config: GpuBenchConfig, 
        param: str, 
        scale_factor: float = 1.0
    ) -> float:
        """Generate a single parameter perturbation."""
        old_val = getattr(config, param.lower(), None)
        if old_val is None:
            return 0.0
        
        # Get parameter bounds
        param_upper = param.upper()
        if param_upper == "LR":
            base = 0.001  # default learning rate
            min_val, max_val = 1e-6, 0.01
        elif param_upper == "BATCH_SIZE":
            base = 32
            min_val, max_val = 8, 128
        elif param_upper == "HIDDEN_DIM":
            base = 128
            min_val, max_val = 32, 512
        else:
            return old_val
        
        # Perturbation magnitude relative to base, scaled
        magnitude = self.perturbation_scale * base * scale_factor
        
        # Random direction and magnitude (log-uniform for positive params)
        if random.random() < 0.5:
            new_val = old_val * (1 + magnitude * random.uniform(0, 1))
        else:
            new_val = old_val * (1 - magnitude * random.uniform(0, 1))
        
        # Clamp to bounds
        if param_upper in ("BATCH_SIZE", "HIDDEN_DIM"):
            new_val = int(round(new_val))
        return max(min_val, min(max_val, new_val))
    
    def update_sensitivity(
        self, 
        changes: dict, 
        bpb_change: float
    ) -> None:
        """
        Update per-parameter sensitivity based on observed BPB change.
        Called after each iteration with the actual result.
        """
        # For each changed parameter, record magnitude of change vs BPB change
        for param, new_val in changes.items():
            change_magnitude = abs(bpb_change)
            if len(self.param_history[param]) >= self.history_window:
                self.param_history[param].pop(0)
            self.param_history[param].append(change_magnitude)
            
            # Update sensitivity as average |BPB change| per unit parameter change
            if self.param_history[param]:
                self.param_sensitivity[param] = np.mean(self.param_history[param])
```

#### 4.2 Modified `GpuBenchRunner.__init__`

Add initialization of the sampler:

```python
def __init__(self, ...):
    # ... existing code ...
    
    # Multi-axis sampler (add after self.config = GpuBenchConfig())
    self.multi_axis_sampler = MultiAxisSampler(
        active_params=self.search_config.active_params,
        perturbation_scale=0.15,  # Slightly larger than single-axis default
        history_window=10,
        sample_size=3 if len(self.search_config.active_params) >= 3 else len(self.search_config.active_params),
        p_random=0.3,
    )
    
    # Track whether we're in multi-axis mode or LLM mode
    self._use_multi_axis = False  # Start with LLM, switch adaptively
    self._consecutive_improvements = 0
    self._stagnation_counter = 0
```

#### 4.3 Modified `GpuBenchRunner.run_iteration`

Add logic to detect stagnation and trigger multi-axis exploration:

```python
def run_iteration(self, iteration: int, *, changes: dict | None = None, hypothesis: str = "") -> BenchResult:
    # ... existing preamble ...
    
    # Determine whether to use multi-axis sampling
    use_multi_axis = self._should_use_multi_axis(iteration)
    
    if changes is None:
        if use_multi_axis:
            # Override LLM proposal with multi-axis exploration
            if self.trace.best_config is None:
                changes = self.multi_axis_sampler.propose(GpuBenchConfig())
            else:
                changes = self.multi_axis_sampler.propose(self.trace.best_config)
            hypothesis = "multi_axis_subspace_sampling"
        elif self.client is None:
            raise ValueError("LLM client required when changes not provided")
        else:
            changes, hypothesis = self._propose(iteration)
    
    # ... existing filtering and trial running code ...
    
    # After acceptance/rejection logic, update sampler:
    if use_multi_axis and result.status != "crash":
        self.multi_axis_sampler.update_sensitivity(
            filtered,
            bpb_change=result.val_bpb - self.trace.best_bpb
        )
    
    # Update stagnation detection
    if result.status == "keep" and result.accepted:
        self._consecutive_improvements += 1
        self._stagnation_counter = 0
    else:
        self._consecutive_improvements = 0
        self._stagnation_counter += 1
    
    self.trace.record(result)
    self.trace.results.append(result)
    return result
```

#### 4.4 New Method: `_should_use_multi_axis`

```python
def _should_use_multi_axis(self, iteration: int) -> bool:
    """
    Decide whether to use multi-axis sampling based on stagnation.
    Use multi-axis when: 
    - No improvement for 3+ consecutive LLM iterations
    - OR every 5th iteration as routine exploration
    """
    if iteration < 2:  # Let baseline settle
        return False
    
    # Stagnation threshold: 3 consecutive non-improvements
    if self._stagnation_counter >= 3:
        return True
    
    # Routine multi-axis exploration every 5 iterations
    if iteration % 5 == 0 and iteration > 0:
        return True
    
    # Random exploration with 20% probability
    if random.random() < 0.2:
        return True
    
    return False
```

### 5. Integration Points

1. **Import**: Add `import random` and `import numpy as np` to top of runner.py
2. **Class addition**: Add `MultiAxisSampler` class before `GpuBenchRunner`
3. **Constructor modification**: Add sampler initialization and tracking variables after `self.config = GpuBenchConfig()`
4. **Method modification**: Replace `run_iteration` with version containing multi-axis logic
5. **New method**: Add `_should_use_multi_axis` to `GpuBenchRunner`

### Edge Cases & Safety

1. **Sample size adjustment**: If `active_params` has fewer than `sample_size` params, reduce to available count
2. **Stale best config**: If `best_config` is None (baseline not run), start from defaults
3. **Crashes**: Don't update sensitivity on crashes (BPB change is meaningless)
4. **Integer parameters**: `batch_size` and `hidden_dim` are rounded and cast to int
5. **Bound clamping**: All perturbations are clamped to physically meaningful ranges
6. **Sensitivity initialization**: Start with uniform sensitivity (1.0) and let history accumulate

### Expected Behavior

1. **Normal operation**: LLM drives single-axis proposals; multi-axis kicks in during stagnation
2. **Stagnation**: After 3 failed LLM proposals, multi-axis tries coordinated changes across multiple parameters
3. **Sensitivity learning**: Over time, the sampler learns which parameters have the most impact and adjusts perturbation magnitudes accordingly
4. **Recovery**: Once multi-axis finds a better configuration, control returns to LLM for refinement
5. **Trace integration**: Multi-axis results are tagged with "multi_axis_subspace_sampling" hypothesis for trace analysis
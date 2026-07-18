## Implementation Specification

**Mechanism name**: structured_random_explorer

**Implementation strategy**: new_helper_class + modify_init + modify_run_iteration

**Target**: GpuBenchRunner class, run_iteration method, __init__ method

## Step-by-step logic

### 1. Create new helper class `StructuredRandomExplorer`

```python
@dataclass
class ExplorationState:
    """Tracks exploration history to avoid flailing."""
    dimension_attempts: Dict[str, int]  # param_name -> number of times attempted
    dimension_successes: Dict[str, int]  # param_name -> number of improvements
    last_direction: Dict[str, float]  # param_name -> last delta direction (+/-1)
    consecutive_failures: int = 0
    stagnation_counter: int = 0
    exploration_budget: int = 10  # Max iterations before forced exploitation
    last_improvement_iteration: int = 0

class StructuredRandomExplorer:
    """
    Principled exploration that learns from past iterations.
    Instead of random flailing, uses systematic parameter perturbation
    with memory of what worked before.
    """
    
    def __init__(self, search_config: SearchConfig, trace: BenchTrace):
        self.search_config = search_config
        self.trace = trace
        self.state = ExplorationState(
            dimension_attempts={},
            dimension_successes={},
            last_direction={}
        )
        # Parameter bounds from search config
        self.param_bounds = self._get_param_bounds()
        
    def _get_param_bounds(self) -> Dict[str, tuple]:
        """Extract parameter bounds from search config."""
        bounds = {}
        if hasattr(self.search_config, 'lr_range'):
            bounds['lr'] = self.search_config.lr_range
        if hasattr(self.search_config, 'batch_size_range'):
            bounds['batch_size'] = self.search_config.batch_size_range
        if hasattr(self.search_config, 'hidden_dim_range'):
            bounds['hidden_dim'] = self.search_config.hidden_dim_range
        return bounds
    
    def propose_exploration(self, iteration: int, best_config: GpuBenchConfig) -> tuple[dict, str]:
        """
        Generate structured exploration proposal with coherent strategy.
        Returns (changes_dict, hypothesis_string).
        """
        active_params = set(self.search_config.active_params)
        
        # Strategy 1: If we've stagnated (no improvement for > 5 iterations)
        # Do a "reset" exploration - try a moderate change in one dimension
        if iteration - self.state.last_improvement_iteration > 5:
            return self._propose_reset_exploration(active_params, best_config)
        
        # Strategy 2: Focus on least-explored parameter dimension
        least_explored = self._find_least_explored_param(active_params)
        if least_explored:
            return self._probe_parameter(least_explored, best_config, active_params)
        
        # Strategy 3: If all dimensions explored equally, try perturbation
        # of the most promising direction
        return self._propose_local_perturbation(active_params, best_config)
    
    def _propose_reset_exploration(self, active_params: set, best_config: GpuBenchConfig) -> tuple[dict, str]:
        """Reset exploration to avoid local optima."""
        # Pick an under-explored parameter
        param = self._find_least_explored_param(active_params)
        if not param:
            param = next(iter(active_params))
        
        changes = {}
        if param == 'lr':
            # Try middle of range
            if 'lr' in self.param_bounds:
                min_lr, max_lr = self.param_bounds['lr']
                changes['lr'] = 10 ** ((math.log10(min_lr) + math.log10(max_lr)) / 2)
        elif param == 'batch_size':
            if 'batch_size' in self.param_bounds:
                min_bs, max_bs = self.param_bounds['batch_size']
                changes['batch_size'] = (min_bs + max_bs) // 2
        elif param == 'hidden_dim':
            if 'hidden_dim' in self.param_bounds:
                min_hd, max_hd = self.param_bounds['hidden_dim']
                changes['hidden_dim'] = (min_hd + max_hd) // 2
        
        hypothesis = f"reset_exploration:{param}:midpoint"
        return changes, hypothesis
    
    def _find_least_explored_param(self, active_params: set) -> str | None:
        """Find parameter with fewest total attempts."""
        if not active_params:
            return None
        
        attempts = {p: self.state.dimension_attempts.get(p, 0) for p in active_params}
        min_attempts = min(attempts.values())
        least_explored = [p for p, a in attempts.items() if a == min_attempts]
        return least_explored[0] if least_explored else None
    
    def _probe_parameter(self, param: str, best_config: GpuBenchConfig, active_params: set) -> tuple[dict, str]:
        """Systematically probe a single parameter dimension."""
        if param not in self.param_bounds:
            return self._propose_local_perturbation(active_params, best_config)
        
        min_val, max_val = self.param_bounds[param]
        current_val = getattr(best_config, param.lower(), None)
        if current_val is None:
            return self._propose_local_perturbation(active_params, best_config)
        
        # Determine direction: alternate based on past success
        last_dir = self.state.last_direction.get(param, 1)
        
        # Try opposite direction of last attempt if it failed
        failures_in_dim = self.state.dimension_attempts.get(param, 0) - self.state.dimension_successes.get(param, 0)
        if failures_in_dim > 2:
            direction = -last_dir
        else:
            direction = last_dir
        
        # Calculate step size adaptively
        if param == 'lr':
            step_factor = 1.5 if direction > 0 else 1/1.5
            new_val = current_val * step_factor
        elif param == 'batch_size':
            step = 16 * direction  # ±16 batch size
            new_val = max(16, min(current_val + step, max_val, max_val))
        elif param == 'hidden_dim':
            step = 32 * direction  # ±32 hidden dim
            new_val = max(32, min(current_val + step, max_val, max_val))
        
        # Clamp to bounds
        new_val = max(min_val, min(new_val, max_val))
        
        # Record attempt direction
        self.state.last_direction[param] = direction
        
        changes = {param: new_val}
        hypothesis = f"probe:{param}:dir={direction}:val={new_val}"
        
        return changes, hypothesis
    
    def _propose_local_perturbation(self, active_params: set, best_config: GpuBenchConfig) -> tuple[dict, str]:
        """Small random perturbation around best config with learned direction bias."""
        import random
        
        changes = {}
        param = random.choice(list(active_params))
        
        if param == 'lr':
            current_lr = best_config.lr
            # Log-normal perturbation with bias toward recent successful direction
            last_dir = self.state.last_direction.get('lr', 1)
            bias = 0.2 * last_dir
            noise = random.gauss(bias, 0.3)  # Mean biased, std 0.3 in log space
            new_lr = current_lr * (10 ** noise)
            changes['lr'] = new_lr
        elif param == 'batch_size':
            current_bs = best_config.batch_size
            step = random.choice([-16, -8, 8, 16])
            new_bs = max(16, min(current_bs + step, 256))
            changes['batch_size'] = new_bs
        elif param == 'hidden_dim':
            current_hd = best_config.hidden_dim
            step = random.choice([-32, -16, 16, 32])
            new_hd = max(32, min(current_hd + step, 1024))
            changes['hidden_dim'] = new_hd
        
        hypothesis = f"local_perturbation:{param}:multi_modal"
        return changes, hypothesis
    
    def update_state(self, changes: dict, was_improvement: bool):
        """Update exploration state based on result."""
        for param in changes:
            self.state.dimension_attempts[param] = self.state.dimension_attempts.get(param, 0) + 1
            if was_improvement:
                self.state.dimension_successes[param] = self.state.dimension_successes.get(param, 0) + 1
        
        if was_improvement:
            self.state.consecutive_failures = 0
            self.state.last_improvement_iteration = self.trace.results[-1].iteration
        else:
            self.state.consecutive_failures += 1
```

### 2. Modify `__init__` method

Add the explorer initialization after existing state setup:

```python
def __init__(self, ...):
    # ... existing code ...
    self.trace = BenchTrace()
    
    # NEW: Add structured explorer
    self.explorer = StructuredRandomExplorer(
        search_config=self.search_config,
        trace=self.trace
    )
```

### 3. Modify `run_iteration` method

Replace the LLM-based proposal logic when `changes is None`:

```python
def run_iteration(
    self,
    iteration: int,
    *,
    changes: dict | None = None,
    hypothesis: str = "",
) -> BenchResult:
    """Run one inner iteration. Uses structured explorer when changes is None."""
    if changes is None:
        # NEW: Use structured explorer instead of LLM for coherent exploration
        changes, hypothesis = self.explorer.propose_exploration(
            iteration=iteration,
            best_config=self.config
        )

    # Rest of existing logic unchanged...
    active = set(self.search_config.active_params)
    filtered = {k: v for k, v in changes.items() if k.upper() in active}
    # ... existing filtering, trial setup, execution, recording ...
    
    # NEW: Update exploration state after recording result
    if result.status != "crash":
        self.explorer.update_state(
            changes=changes,
            was_improvement=(result.status == "keep")
        )
    
    return result
```

## Integration points

1. **SearchConfig**: Ensure all parameter bounds are accessible via `lr_range`, `batch_size_range`, `hidden_dim_range` attributes (or add them to the config class).

2. **GpuBenchConfig**: Ensure config attributes are accessible by lowercase name (`lr`, `batch_size`, `hidden_dim`) for the explorer to read current values.

3. **BenchTrace**: The explorer uses `trace.results[-1].iteration` to track improvement timing—ensure trace results have `iteration` attribute.

4. **Configuration file**: Add new config fields for explorer parameters (exploration_budget, perturbation_steps, etc.) in a new `[exploration]` section.

5. **Testing**: Add unit tests for `StructuredRandomExplorer.propose_ex
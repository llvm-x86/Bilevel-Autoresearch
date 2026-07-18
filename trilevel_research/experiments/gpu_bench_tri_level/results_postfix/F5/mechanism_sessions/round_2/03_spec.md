## Implementation Specification: Stagnation Detection with Restart Mechanism

### 1. Mechanism name
`stagnation_restart_controller`

### 2. Implementation strategy
`new_helper_class` + `modify_init` + `modify_run_iteration`

### 3. Target
New class: `StagnationRestartController`
Modified: `GpuBenchRunner.__init__`, `GpuBenchRunner.run_iteration`

### 4. Step-by-step logic

**A. Create `StagnationRestartController` class with:**

1. **Constructor parameters:**
   - `window_size: int = 5` (number of iterations to look back)
   - `stagnation_threshold_bpb: float = 0.001` (min absolute improvement to consider non-stagnant)
   - `top_k: int = 3` (number of best configs to keep for restart)
   - `perturbation_std: float = 0.01` (standard deviation of Gaussian noise for restart)
   - `restart_cooldown: int = 3` (minimum iterations between restarts to prevent oscillation)

2. **Internal state attributes:**
   - `self.best_bpb_history: list[float]` - sliding window of best-so-far values
   - `self.best_configs: list[tuple[float, GpuBenchConfig]]` - sorted by val_bpb, kept to top_k
   - `self.restart_count: int = 0`
   - `self.last_restart_iteration: int = -restart_cooldown`
   - `self.stagnant_iterations: int = 0`
   - `self.restart_active: bool = False` - flag that run_iteration checks

3. **Method `update(iteration: int, config: GpuBenchConfig, val_bpb: float) -> bool`:**
   ```python
   def update(self, iteration: int, config: GpuBenchConfig, val_bpb: float) -> bool:
       """
       Updates state with new result.
       Returns True if a restart should be triggered.
       """
       # Update best history window
       if not self.best_bpb_history or val_bpb < self.best_bpb_history[-1]:
           self.best_bpb_history.append(val_bpb)
       else:
           self.best_bpb_history.append(self.best_bpb_history[-1])
       
       # Keep window size bounded
       if len(self.best_bpb_history) > self.window_size:
           self.best_bpb_history.pop(0)
       
       # Update top-K configs (keep sorted by val_bpb, ascending)
       self.best_configs.append((val_bpb, config))
       self.best_configs.sort(key=lambda x: x[0])
       self.best_configs = self.best_configs[:self.top_k]
       
       # Check cooldown
       if iteration - self.last_restart_iteration < self.restart_cooldown:
           return False
       
       # Check stagnation: window full and no meaningful improvement
       if len(self.best_bpb_history) < self.window_size:
           return False
       
       improvement = self.best_bpb_history[-1] - self.best_bpb_history[0]
       if improvement < self.stagnation_threshold_bpb:
           self.stagnant_iterations += 1
           if self.stagnant_iterations >= 2:  # Second consecutive stagnant check
               self.restart_count += 1
               self.last_restart_iteration = iteration
               self.stagnant_iterations = 0
               return True
       else:
           self.stagnant_iterations = 0
       
       return False
   ```

4. **Method `generate_restart_config() -> GpuBenchConfig`:**
   ```python
   def generate_restart_config(self) -> GpuBenchConfig:
       """
       Returns a perturbed copy of the best known config, or random if none exist.
       """
       if not self.best_configs:
           return GpuBenchConfig()  # Random config
       
       best_val, best_config = self.best_configs[0]
       restart_config = deepest_copy(best_config)
       
       # Apply Gaussian perturbation to all float params
       for param_name, param_value in vars(restart_config).items():
           if isinstance(param_value, float):
               noise = random.gauss(0, self.perturbation_std)
               original = param_value
               new_value = param_value * (1.0 + noise)
               # Optionally clip to reasonable bounds
               setattr(restart_config, param_name, new_value)
       
       return restart_config
   ```

**B. Modify `GpuBenchRunner.__init__`:**

Add after existing init logic:
```python
# Initialize stagnation controller
self.stagnation_controller = StagnationRestartController(
    window_size=self.search_config.stagnation_window_size or 5,
    stagnation_threshold_bpb=self.search_config.stagnation_threshold_bpb or 0.001,
    top_k=self.search_config.stagnation_top_k or 3,
    perturbation_std=self.search_config.stagnation_perturbation_std or 0.01,
    restart_cooldown=self.search_config.stagnation_restart_cooldown or 3,
)
```

Also initialize restart flag:
```python
self.pending_restart = False
self.pending_restart_config = None
```

**C. Modify `GpuBenchRunner.run_iteration`:**

1. At the very beginning of `run_iteration`, BEFORE the `if changes is None:` block:
```python
# Check if restart is pending from previous iteration's stagnation detection
if self.pending_restart:
    changes = self.pending_restart_config
    hypothesis = f"restart_from_stagnation_{self.stagnation_controller.restart_count}"
    self.pending_restart = False
    self.pending_restart_config = None
```

2. After computing `result.val_bpb` and updating `self.trace.best_bpb`, call stagnation check:
```python
# After updating best_bpb and best_config in the "keep" branch:
if result.status == "keep":
    should_restart = self.stagnation_controller.update(
        iteration, 
        self.trace.best_config, 
        self.trace.best_bpb
    )
    if should_restart:
        restart_config = self.stagnation_controller.generate_restart_config()
        self.pending_restart = True
        self.pending_restart_config = restart_config
        
        # Also set the restart flag for the return value
        result.status = "restart_scheduled"
```

Also handle the "discard" branch:
```python
elif result.status == "discard" and result.val_bpb < self.trace.best_bpb + self.stagnation_controller.stagnation_threshold_bpb * 2:
    # Even discarded results can inform stagnation (they didn't improve, but are close)
    should_restart = self.stagnation_controller.update(
        iteration,
        self.config,  # Use current working config
        result.val_bpb
    )
    if should_restart:
        restart_config = self.stagnation_controller.generate_restart_config()
        self.pending_restart = True
        self.pending_restart_config = restart_config
        result.status = "restart_scheduled"
```

### 5. Integration points

1. **SearchConfig class** - Add optional stagnation parameters:
   - `stagnation_window_size: int = 5`
   - `stagnation_threshold_bpb: float = 0.001`
   - `stagnation_top_k: int = 3`
   - `stagnation_perturbation_std: float = 0.01`
   - `stagnation_restart_cooldown: int = 3`

2. **trace.record** - The trace should log restart events with a new field indicating restart source:
   - Add `restart_count` field to `BenchTrace`
   - Add `restart_iteration` to track which iteration triggered restart

3. **Result status enum** - Add `restart_scheduled` to allowed statuses (the runner will handle it by injecting the restart config on the next iteration)

4. **Logging** - Add INFO-level logging:
   ```python
   logging.info(f"Stagnation detected at iteration {iteration}. "
                f"Best val_bpb: {self.trace.best_bpb:.4f}, "
                f"Restart count: {self.stagnation_controller.restart_count}")
   ```

5. **Testing** - Verify:
   - Sliding window correctly tracks best-so-far (not current)
   - Restart cooldown prevents oscillation
   - Perturbation preserves config structure
   - Restart config is applied on NEXT iteration, not current one
   - Multiple stagnation cycles work correctly
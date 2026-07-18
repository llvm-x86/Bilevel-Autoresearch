## Implementation Specification: Crossover Search for GpuBenchRunner

Based on the selected hypothesis to reject the four original proposals and instead focus on search space expansion, this specification implements a crossover-based search augmentation that allows the runner to explore larger model sizes (the core limitation identified in the trace analysis).

### 1. Mechanism name
`crossover_search`

### 2. Implementation strategy
`new_helper_class` + `modify_init`

### 3. Target
`GpuBenchRunner` class (`runner.py`)

### 4. Step-by-step logic

**Step 1: Add CrossoverConfig dataclass to search configuration**
Add a new configuration dataclass at the module level (before `GpuBenchRunner`):

```python
@dataclass
class CrossoverConfig:
    """Configuration for crossover search space expansion."""
    enabled: bool = True
    pool_size: int = 4          # Number of elite configs to maintain
    crossover_population: int = 2  # Number of crossover configs per iteration
    mutation_rate: float = 0.2  # Probability of mutating a crossed-over parameter
    scale_factor: float = 1.5   # How much to scale up model sizes during crossover
    max_model_params: int = 100_000_000  # Upper bound for model parameters
```

**Step 2: Add CrossoverPool helper class**

```python
class CrossoverPool:
    """
    Maintains a pool of elite configurations and generates new 
    configurations via crossover (mixing parameters from two parents).
    """
    def __init__(self, config: CrossoverConfig):
        self.config = config
        self.elite_pool: list[tuple[float, GpuBenchConfig]] = []  # (val_bpb, config)
        
    def add_result(self, val_bpb: float, config: GpuBenchConfig):
        """Add a result to the elite pool, keeping only top performers."""
        self.elite_pool.append((val_bpb, config))
        self.elite_pool.sort(key=lambda x: x[0])  # Lower val_bpb is better
        self.elite_pool = self.elite_pool[:self.config.pool_size]
        
    def generate_crossover_configs(self, base_config: GpuBenchConfig) -> list[GpuBenchConfig]:
        """Generate new configs by crossing over elite configs with the base config."""
        if len(self.elite_pool) < 2:
            return []
            
        new_configs = []
        elite_configs = [c for _, c in self.elite_pool]
        
        for _ in range(self.config.crossover_population):
            # Select two parents: one from elite pool, one from base or another elite
            parent1 = random.choice(elite_configs)
            
            if random.random() < 0.5 and len(elite_configs) > 1:
                # Two elite parents
                parent2 = random.choice([c for c in elite_configs if c is not parent1])
            else:
                # One elite + one base config
                parent2 = base_config
                
            child = self._crossover(parent1, parent2)
            new_configs.append(child)
            
        return new_configs
    
    def _crossover(self, parent1: GpuBenchConfig, parent2: GpuBenchConfig) -> GpuBenchConfig:
        """Create a new child config from two parents using uniform crossover + mutation."""
        # Create child as copy of parent1
        child = copy.deepcopy(parent1)
        
        # Define which parameters to crossover (focus on architectural parameters)
        crossover_params = [
            'N_LAYER', 'N_HEAD', 'N_EMBD', 'BLOCK_SIZE',
            'VOCAB_SIZE', 'N_EMBD_HEAD', 'COMPILE'
        ]
        
        child_params = {}
        parent1_dict = parent1.to_dict()
        parent2_dict = parent2.to_dict()
        child_dict = child.to_dict()
        
        for param in crossover_params:
            if param in parent1_dict and param in parent2_dict:
                # Uniform crossover: 50% chance from either parent
                if random.random() < 0.5:
                    child_dict[param] = parent2_dict[param]
                    
        # Apply config scaling for model size expansion
        if random.random() < self.config.mutation_rate:
            # Mutate N_LAYER and N_EMBD to explore larger models
            for scale_param in ['N_LAYER', 'N_EMBD']:
                if scale_param in child_dict:
                    # Randomly scale up or stay the same
                    if random.random() < 0.3:
                        child_dict[scale_param] = int(
                            child_dict[scale_param] * self.config.scale_factor
                        )
        
        # Apply constraints
        child_dict['N_LAYER'] = min(child_dict.get('N_LAYER', 12), 48)
        child_dict['N_EMBD'] = min(child_dict.get('N_EMBD', 768), 2048)
        
        # Reconstruct child config
        child = GpuBenchConfig.from_dict(child_dict, check_constraints=False)
        
        # Validate total parameters don't exceed max
        total_params = child.total_params()
        if total_params > self.config.max_model_params:
            # Scale down proportionally
            scale = self.config.max_model_params / total_params
            child_dict['N_LAYER'] = max(4, int(child_dict['N_LAYER'] * scale ** 0.5))
            child_dict['N_EMBD'] = max(64, int(child_dict['N_EMBD'] * scale ** 0.5))
            child = GpuBenchConfig.from_dict(child_dict, check_constraints=False)
            
        return child
```

**Step 3: Modify `__init__` to add CrossoverPool and crossover configuration**
Add to `__init__`:
```python
self.crossover_config = CrossoverConfig()
self.crossover_pool = CrossoverPool(self.crossover_config)
self.all_crossover_configs: list[GpuBenchConfig] = []  # Track generated crossover configs
```

**Step 4: Modify `run_iteration` to integrate crossover logic**
After the existing iteration logic, add:

```python
# After running the trial and updating best config
if (self.crossover_config.enabled and 
    result.status == "keep" and 
    iteration % 3 == 0):  # Every 3 iterations, generate crossover variations
    
    # Add current best to crossover pool
    self.crossover_pool.add_result(result.val_bpb, trial)
    
    # Generate new configurations from crossover
    crossover_configs = self.crossover_pool.generate_crossover_configs(self.config)
    
    for i, crossover_config in enumerate(crossover_configs):
        # Run each crossover configuration
        crossover_iter = iteration * 100 + i + 1  # Unique iteration number
        crossover_result = self._run_trial(
            crossover_config, 
            iteration=crossover_iter,
            hypothesis="crossover"
        )
        crossover_result.changes = self._compute_changes(self.config, crossover_config)
        
        # Record and possibly accept crossover results
        if crossover_result.status != "crash":
            if crossover_result.val_bpb < self.trace.best_bpb:
                crossover_result.status = "keep"
                crossover_result.accepted = True
                self.trace.best_val_bpb = crossover_result.val_bpb
                self.trace.best_bpb = crossover_result.val_bpb
                self.trace.best_iteration = crossover_iter
                self.trace.best_config = crossover_config
                self.config = crossover_config
            else:
                crossover_result.status = "discard"
                crossover_result.accepted = False
                
            self.trace.record(crossover_result)
            self.trace.results.append(crossover_result)
            self.all_crossover_configs.append(crossover_config)
```

**Step 5: Add helper method `_compute_changes`**
```python
def _compute_changes(self, base: GpuBenchConfig, target: GpuBenchConfig) -> dict:
    """Compute the parameter changes between base and target configs."""
    base_dict = base.to_dict()
    target_dict = target.to_dict()
    changes = {}
    for key in base_dict:
        if base_dict[key] != target_dict.get(key):
            changes[key] = target_dict[key]
    return changes
```

### 5. Integration points

**What to modify in existing `runner.py`:**
- Add `from dataclasses import dataclass` to imports
- Add `import copy` and `import random` to imports  
- Add `CrossoverConfig` dataclass before `GpuBenchRunner` class
- Add `CrossoverPool` class before `GpuBenchRunner` class
- Modify `__init__`: Add `self.crossover_config` and `self.crossover_pool` initialization
- Modify `run_iteration`: Add crossover logic block after existing trial evaluation
- Add `_compute_changes` helper method

**Dependencies:**
- `GpuBenchConfig.to_dict()` - expect this method exists or add it
- `GpuBenchConfig.from_dict()` - expect this exists or add it  
- `GpuBenchConfig.total_params()` - expect this exists or add it
- The existing `_run_trial` method and `BenchResult` structure

**Testing considerations:**
1. Test with `crossover_config.enabled = False` first to verify no regression
2. Test with `elite_pool` artificially seeded with known good configs
3. Verify max model parameter constraint works correctly
4. Monitor GPU memory usage when testing larger models

**Rollback plan:**
- Add to `SearchConfig` a parameter `crossover_enabled: bool = False` defaulting to disabled
- When disabled, the entire crossover block in `run_iteration` is skipped
- All crossover-specific imports and classes are guarded by `if self.crossover_config.enabled`
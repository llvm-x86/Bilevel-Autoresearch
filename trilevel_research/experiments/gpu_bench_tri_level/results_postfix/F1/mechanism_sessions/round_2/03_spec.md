## Implementation Specification: Elastic Parameter Radius with Directed Exploration

**Mechanism name**: elastic_radius_directed_exploration

**Implementation strategy**: new_helper_class + modify_init + replace_method

**Target**: `GpuBenchRunner` class, specifically `run_iteration` method and `__init__`

### Step-by-step logic

#### 1. Create new helper class: `ElasticRadiusExplorer`

```python
class ElasticRadiusExplorer:
    """Maintains an elastic search radius around best known config.
    Expands when configs are promising, contracts when they fail,
    and systematically explores orthogonal parameter pairs."""
    
    def __init__(self, initial_best_config: GpuBenchConfig, 
                 active_params: set,
                 elasticity_scale: float = 2.0,
                 min_radius: float = 0.1,
                 max_radius: float = 10.0):
        self.best_config = initial_best_config
        self.active_params = active_params
        self.radius = 1.0  # Base radius multiplier
        self.elasticity_scale = elasticity_scale  # How much to expand/contract
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.consecutive_improvements = 0
        self.consecutive_failures = 0
        self.adjustment_threshold = 3  # Adjust radius after this many consecutives
        
        # Track which parameters have been explored together
        self.explored_pairs = set()
        
        # Parameter scaling factors (how much 1 unit of radius means)
        self.param_scales = {
            'LR': 0.001,        # 1 radius = ±0.001 LR
            'BATCH_SIZE': 8,     # 1 radius = ±8 BATCH_SIZE
            'MOMENTUM': 0.05,    # 1 radius = ±0.05 MOMENTUM
            'DROPOUT': 0.02,     # 1 radius = ±0.02 DROPOUT
            'LAYERS': 1,         # 1 radius = ±1 LAYER
            'HIDDEN_DIM': 16,    # 1 radius = ±16 HIDDEN_DIM
        }
```

#### 2. Modify `__init__` to instantiate the explorer

```python
def __init__(self, ...):
    # ... existing init code ...
    self.search_config = search_config or SearchConfig()
    self.elastic_explorer = None  # Will be initialized after first baseline
```

#### 3. Replace/modify `run_baseline` to initialize explorer

```python
def run_baseline(self) -> BenchResult:
    result = self._run_trial(self.config, iteration=0, hypothesis="baseline")
    self.trace.best_val_bpb = result.val_bpb
    self.trace.best_bpb = result.val_bpb
    self.trace.best_iteration = 0
    self.trace.best_config = GpuBenchConfig()
    result.status = "keep"
    result.accepted = True
    self.trace.results.append(result)
    
    # Initialize elastic explorer with baseline config
    self.elastic_explorer = ElasticRadiusExplorer(
        initial_best_config=GpuBenchConfig(),
        active_params=set(self.search_config.active_params)
    )
    
    return result
```

#### 4. Replace `run_iteration` with directed exploration logic

```python
def run_iteration(self, iteration: int, *, changes: dict | None = None, hypothesis: str = "") -> BenchResult:
    # If changes are explicitly provided, use existing logic
    if changes is not None:
        return self._run_with_changes(iteration, changes, hypothesis)
    
    # Use elastic explorer for LLM-guided proposals
    return self._run_elastic_exploration(iteration, hypothesis)

def _run_elastic_exploration(self, iteration: int, hypothesis: str) -> BenchResult:
    """Run exploration using elastic radius around best config."""
    
    if self.client is None:
        raise ValueError("LLM client required for elastic exploration")
    
    # Determine the exploration mode based on iteration parity
    if iteration % 2 == 0:
        # Even iterations: Systematically explore LR-BATCH_SIZE pairs
        changes, hypothesis = self._explore_lr_batch_pairs(iteration)
    else:
        # Odd iterations: Use LLM to propose other parameter changes
        changes, hypothesis = self._llm_propose_elastic(iteration)
    
    # Filter to active params
    active = set(self.search_config.active_params)
    filtered = {k: v for k, v in changes.items() if k.upper() in active}
    
    if not filtered:
        return BenchResult(
            val_bpb=self.trace.best_bpb,
            train_bpb=self.trace.best_bpb,
            elapsed_s=0.0,
            status="discard",
            changes={},
            hypothesis=hypothesis or "no active param changes",
            iteration=iteration,
        )
    
    trial = self._trial_config(iteration)
    trial.apply_changes(filtered)
    result = self._run_trial(trial, iteration=iteration, hypothesis=hypothesis)
    result.changes = filtered
    
    # Update elastic explorer based on result
    self._update_elastic_state(result, trial)
    
    # Accept/reject logic
    if result.status == "crash":
        result.accepted = False
        self.elastic_explorer.consecutive_failures += 1
        self.elastic_explorer.consecutive_improvements = 0
    elif result.val_bpb < self.trace.best_bpb:
        result.status = "keep"
        result.accepted = True
        self.trace.best_val_bpb = result.val_bpb
        self.trace.best_bpb = result.val_bpb
        self.trace.best_iteration = iteration
        self.trace.best_config = trial
        self.config = trial
        
        # Store best config in explorer
        self.elastic_explorer.best_config = trial
        self.elastic_explorer.consecutive_improvements += 1
        self.elastic_explorer.consecutive_failures = 0
    else:
        result.status = "discard"
        result.accepted = False
        self.elastic_explorer.consecutive_failures += 1
        self.elastic_explorer.consecutive_improvements = 0
    
    # Adjust radius based on consecutive outcomes
    self._adjust_exploration_radius()
    
    self.trace.record(result)
    self.trace.results.append(result)
    return result
```

#### 5. Add helper methods for elastic exploration

```python
def _explore_lr_batch_pairs(self, iteration: int) -> tuple[dict, str]:
    """Systematically explore (LR, BATCH_SIZE) pairs within elastic radius."""
    
    explorer = self.elastic_explorer
    best = explorer.best_config
    
    # Generate candidate LR and BATCH_SIZE values
    radius = explorer.radius
    lr_step = explorer.param_scales['LR'] * radius
    bs_step = explorer.param_scales['BATCH_SIZE'] * radius
    
    # Create exploration grid: 4 directions (LR up, LR down, BS up, BS down)
    # but only 2 at a time to avoid blowing up
    direction = (iteration // 2) % 4
    
    changes = {}
    hypothesis_parts = ["Directed exploration around best config"]
    
    if direction == 0:
        changes['LR'] = min(max(best.lr + lr_step, 1e-5), 10.0)
        changes['BATCH_SIZE'] = max(32, best.batch_size + bs_step)
        hypothesis_parts.append(f"LR={best.lr}→{changes['LR']}, BS={best.batch_size}→{changes['BATCH_SIZE']}")
    elif direction == 1:
        changes['LR'] = max(1e-5, best.lr - lr_step)
        changes['BATCH_SIZE'] = max(32, best.batch_size - bs_step)
        hypothesis_parts.append(f"LR={best.lr}→{changes['LR']}, BS={best.batch_size}→{changes['BATCH_SIZE']}")
    elif direction == 2:
        changes['LR'] = min(max(best.lr + lr_step * 0.5, 1e-5), 10.0)
        changes['BATCH_SIZE'] = max(32, best.batch_size - bs_step * 0.5)
        hypothesis_parts.append(f"LR={best.lr}→{changes['LR']}, BS={best.batch_size}→{changes['BATCH_SIZE']}")
    else:
        changes['LR'] = max(1e-5, best.lr - lr_step * 0.5)
        changes['BATCH_SIZE'] = max(32, best.batch_size + bs_step * 0.5)
        hypothesis_parts.append(f"LR={best.lr}→{changes['LR']}, BS={best.batch_size}→{changes['BATCH_SIZE']}")
    
    return changes, " | ".join(hypothesis_parts)

def _llm_propose_elastic(self, iteration: int) -> tuple[dict, str]:
    """Modified LLM proposal that constrains suggestions to elastic radius."""
    
    # Prepare enhanced context for LLM
    context = self._build_elastic_context(iteration)
    
    # Use existing LLM client but with constrained proposal space
    changes, hypothesis = self.client.propose_config(context)
    
    # Constrain changes to elastic radius
    explorer = self.elastic_explorer
    best = explorer.best_config
    constrained_changes = {}
    
    for param, value in changes.items():
        if param.upper() not in explorer.active_params:
            continue
        
        param_upper = param.upper()
        current_value = getattr(best, param_upper.lower(), None)
        if current_value is None:
            constrained_changes[param] = value
            continue
        
        max_delta = explorer.param_scales.get(param_upper, 1.0) * explorer.radius
        new_value = current_value + max(-max_delta, min(max_delta, value - current_value))
        constrained_changes[param] = new_value
    
    return constrained_changes, hypothesis

def _build_elastic_context(self, iteration: int) -> dict:
    """Build context including elastic radius info for LLM proposals."""
    explorer = self.elastic_explorer
    best = explorer.best_config
    
    return {
        'best_config': {
            'lr': best.lr,
            'batch_size': best.batch_size,
            'momentum': best.momentum,
            'dropout': best.dropout,
            'layers': best.layers,
            'hidden_dim': best.hidden_dim
        },
        'elastic_radius': explorer.radius,
        'param_scales': explorer.param_scales,
        'exploration_history': [
            {
                'iteration': r.iteration,
                'val_bpb': r.val_bpb,
                'changes': r.changes,
                'accepted': r.accepted
            }
            for r in self.trace.results[-5:]  # Last 5 results
        ],
        'constraint': f"Propose changes
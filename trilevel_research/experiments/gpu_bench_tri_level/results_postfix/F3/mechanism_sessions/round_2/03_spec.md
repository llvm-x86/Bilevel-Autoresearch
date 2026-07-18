## Implementation Specification: Relative Validation BPD Offset Reporting

**1. Mechanism name**: `relative_val_bpd_reporting`

**2. Implementation strategy**: `new_helper_class` + `modify_init` + `replace_method`

**3. Target**: `GpuBenchRunner` class, specifically:
- `__init__` (modify)
- `run_baseline` (modify)
- `run_iteration` (modify)
- New helper class: `RelativeBPBTracker`

**4. Step-by-step logic**:

### Step 1: Create RelativeBPBTracker class

```python
class RelativeBPBTracker:
    """Tracks best BPD and computes relative offsets for proposer feedback."""
    
    def __init__(self):
        self._best_val_bpb: float = float('inf')  # Track minimum bpb
        self._is_initialized: bool = False
        self._baseline_val_bpb: float = 0.0
        
    def initialize(self, val_bpb: float, is_baseline: bool = True) -> None:
        """Set initial best value from baseline run."""
        self._best_val_bpb = val_bpb
        self._is_initialized = True
        if is_baseline:
            self._baseline_val_bpb = val_bpb
    
    def update_best(self, val_bpb: float) -> bool:
        """Update best if new value is lower. Returns True if improved."""
        if not self._is_initialized:
            self.initialize(val_bpb, is_baseline=False)
            return True
        if val_bpb < self._best_val_bpb:
            self._best_val_bpb = val_bpb
            return True
        return False
    
    def get_relative_offset(self, val_bpb: float) -> float:
        """Return (val_bpb - best) or 0.0 if equal, ensuring negative==good."""
        if not self._is_initialized:
            return 0.0
        # Clamp to 0.0 for perfect match to avoid noise
        diff = val_bpb - self._best_val_bpb
        return 0.0 if abs(diff) < 1e-10 else diff
    
    def get_best_absolute(self) -> float:
        """Return absolute best value for logging/checkpoint naming."""
        return self._best_val_bpb if self._is_initialized else float('inf')
    
    @property
    def best_val_bpb(self) -> float:
        return self._best_val_bpb
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
```

### Step 2: Modify `__init__`

Add after `self.trace = BenchTrace()`:
```python
# Add relative BPD tracker
self._bpd_tracker = RelativeBPBTracker()
```

### Step 3: Modify `run_baseline`

After `result = self._run_trial(...)` and before `self.trace.best_val_bpb = result.val_bpb`:
```python
# Initialize relative BPD tracker with baseline
self._bpd_tracker.initialize(result.val_bpb, is_baseline=True)
```

Keep the existing `self.trace.best_val_bpb = result.val_bpb` for backward compatibility.

### Step 4: Modify proposal preparation in `run_iteration`

Before calling `self._propose(iteration)`, add a new method `_prepare_proposal_context`:

```python
def _prepare_proposal_context(self, iteration: int) -> dict:
    """Prepare context with relative BPD values for proposer."""
    context = {
        'iteration': iteration,
        'best_val_bpb': 0.0,  # Relative: best is always 0.0
        'best_val_bpb_absolute': self._bpd_tracker.get_best_absolute(),
        'baseline_val_bpb': self._bpd_tracker._baseline_val_bpb,  # For reference
        'relative_reporting': True,  # Signal to proposer
    }
    
    # Prepare recent results with relative offsets
    recent_trace = []
    for r in self.trace.results[-3:]:  # Last 3 results
        if r.status not in ('discard', 'crash'):
            recent_trace.append({
                'iteration': r.iteration,
                'val_bpd_relative': self._bpd_tracker.get_relative_offset(r.val_bpb),
                'val_bpd_absolute': r.val_bpb,
                'status': r.status,
                'changes': r.changes,
            })
    context['recent_results'] = recent_trace
    return context
```

Then modify the `_propose` call location:
```python
# In run_iteration, before proposing:
proposal_context = self._prepare_proposal_context(iteration)
changes, hypothesis = self._propose(iteration, context=proposal_context)
```

### Step 5: Modify `_propose` method

Replace `_propose` method signature and logic:

```python
def _propose(
    self, 
    iteration: int, 
    context: dict | None = None
) -> tuple[dict, str]:
    """Propose next config with relative BPD context."""
    
    # Build prompt with relative values
    if context and context.get('relative_reporting'):
        best_bpd = context['best_val_bpb']  # Always 0.0
        recent = context.get('recent_results', [])
        prompt_parts = [
            f"Round {iteration} optimization proposal.",
            f"Best validation BPD so far (relative): {best_bpd:.4f} (baseline: {context['baseline_val_bpb']:.4f} absolute)",
            "Note: All validation BPD values are reported as offsets from the best seen.",
            "Negative values mean improvement, zero means equal to best, positive means worse.",
            "Recent iterations:"
        ]
        for r in recent:
            sign = "better" if r['val_bpd_relative'] < 0 else "same" if r['val_bpd_relative'] == 0 else "worse"
            prompt_parts.append(
                f"  Iter {r['iteration']}: val_bpd={r['val_bpd_relative']:.4f} ({sign}) [status: {r['status']}]"
            )
        prompt_parts.append(f"Propose parameter changes to minimize validation BPD.")
        prompt = "\n".join(prompt_parts)
    else:
        # Fallback to original prompt
        prompt = f"Propose parameter changes for round {iteration}. Current best BPD: {self.trace.best_bpb:.4f}"
    
    # Rest of proposal logic unchanged...
    system = self._build_system_prompt()
    response = self.client.chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": prompt}]
    )
    
    hypothesis = f"proposer_via_llm_iter_{iteration}_rel_bpd"
    changes = self._parse_response(response)
    
    if not changes:
        # If proposer returns nothing, suggest slight perturbation
        changes = self._generate_fallback_proposal(context)
        hypothesis = f"fallback_random_iter_{iteration}"
    
    return changes, hypothesis
```

### Step 6: Add fallback proposal method

```python
def _generate_fallback_proposal(self, context: dict | None = None) -> dict:
    """Generate fallback proposal when proposer gives empty changes."""
    # Use current best config as base
    best_config = self.trace.best_config or GpuBenchConfig()
    
    # Add small random perturbation to active params
    changes = {}
    for param in self.search_config.active_params:
        if hasattr(best_config, param):
            current = getattr(best_config, param)
            if isinstance(current, (int, float)):
                # Perturb by 10% but ensure different
                import random
                delta = current * 0.1 * random.uniform(-1, 1)
                new_val = current + delta
                # Avoid exact same value
                if abs(new_val - current) < 1e-6:
                    new_val = current * (1 + 0.05 * random.choice([-1, 1]))
                changes[param] = new_val
    return changes
```

### Step 7: Modify result acceptance logic in `run_iteration`

Replace the acceptance logic (after line `if result.val_bpb < self.trace.best_bpb:`):

```python
# New acceptance logic with relative tracker
if result.status == "crash":
    result.accepted = False
elif self._bpd_tracker.update_best(result.val_bpb):
    # New best found (absolute improvement)
    result.status = "keep"
    result.accepted = True
    self.trace.best_val_bpb = result.val_bpb
    self.trace.best_bpb = result.val_bpb
    self.trace.best_iteration = iteration
    self.trace.best_config = trial
elif result.val_bpb == self.trace.best_bpb:
    # Equal to best - could be useful for exploration
    result.status = "keep"  # Changed from discard to keep
    result.accepted = True  # Accept even if equal to maintain momentum
else:
    # Worse than best - still record but mark as discard for proposer feedback
    result.status = "discard"
    result.accepted = False
```

**5. Integration points**:

1. **LLM Client Interface**: The `context` dict passed to `_propose` must be thread-safe if the client is async. Ensure the client's `chat()` method can accept optional keyword arguments for context.

2. **Trace Logging**: The `self.trace` object's `BenchResult` entries should store both absolute and relative BPD:
   ```python
   # Add to BenchResult dataclass (or as extra fields):
   @dataclass
   class BenchResult:
       # ... existing fields ...
       val_bpd_relative: float = 0.0  # NEW
       best_val_bpd_absolute: float = 0.0  # NEW
   ```
   And set these after result creation:
   ```python
   result.val_bpd_relative = self._bpd_tracker.get_relative_offset(result.val_bpb)
   result.best_val_bpd_absolute = self._bpd_tracker.get_best_absolute()
   ```

3. **Baseline Comparison**: The `run_baseline` method should report `val_bpd_relative = 0.0` for the baseline run, and set `best_val_bpd_absolute = result.val_bpb`.

4. **Failure Recovery**: If the runner restarts mid-search, implement serialization of `_bpd_tracker` state:
   ```python
   def save_state(self, path: Path) -> None:
       import json
       state = {
           'best_val_bpb': self._bpd_tracker.get_best_absolute(),
           'baseline_val_bpb': self._bpd_tracker._baseline_val_bpb,
           'is_initialized': self._bpd_tracker.is_initialized,
           'iteration': self.trace.best_iteration,
           'config': self.trace.best_config.to_dict() if self.trace.best_config else None,
       }
       path.write_text(json.dumps(state, indent=2))
   
   def load_state(self, path: Path) ->
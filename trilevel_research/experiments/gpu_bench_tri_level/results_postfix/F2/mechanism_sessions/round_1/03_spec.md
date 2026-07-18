## Implementation Specification: Bootstrap Aggregation for Noise-Resistant Acceptance

### 1. Mechanism name
`bootstrap_replicate_aggregation`

### 2. Implementation strategy
`modify_init` + `new_helper_class` + `replace_method` (replace `run_iteration`)

### 3. Target
- **modify_init**: `GpuBenchRunner.__init__` — add configuration parameter
- **new_helper_class**: `ConfigCache` — stores replicate results per unique config
- **replace_method**: `GpuBenchRunner.run_iteration` — wrap with replicate aggregation logic

### 4. Step-by-step logic

#### 4a. New helper class: `ConfigCache`
```python
class ConfigCache:
    """Stores multiple runs per unique config and provides aggregated metrics."""
    
    def __init__(self, min_replicates: int = 2, cache_dir: Path | None = None):
        self.min_replicates = min_replicates
        # Key: tuple of frozen config params, Value: list of BenchResult
        self._cache: dict[frozenset, list[BenchResult]] = {}
        self._cache_dir = cache_dir  # optional persistence
    
    def _config_key(self, config: GpuBenchConfig) -> frozenset:
        """Create hashable key from sorted active parameters."""
        params = {
            k: getattr(config, k.lower()) 
            for k in ['HIDDEN_DIM', 'LR', 'BATCH_SIZE', 'GRAD_CLIP']
            if hasattr(config, k.lower())
        }
        return frozenset(params.items())
    
    def add_result(self, config: GpuBenchConfig, result: BenchResult) -> None:
        key = self._config_key(config)
        if key not in self._cache:
            self._cache[key] = []
        self._cache[key].append(result)
    
    def get_aggregated(self, config: GpuBenchConfig) -> float | None:
        """Return mean val_bpb if min_replicates reached, else None."""
        key = self._config_key(config)
        entries = self._cache.get(key, [])
        if len(entries) < self.min_replicates:
            return None
        vals = [r.val_bpb for r in entries if r.status != 'crash']
        if not vals:
            return None
        return sum(vals) / len(vals)
    
    def needs_replicate(self, config: GpuBenchConfig) -> bool:
        """Check if config needs more runs to reach min_replicates."""
        key = self._config_key(config)
        entries = self._cache.get(key, [])
        valid = [r for r in entries if r.status != 'crash']
        return len(valid) < self.min_replicates
```

#### 4b. Modify `__init__`
Add at end of existing `__init__`:
```python
self.config_cache = ConfigCache(min_replicates=self.search_config.config_replicates)
```

Add to `SearchConfig` (or use existing):
- New field: `config_replicates: int = 2`  # default 2, min 1

#### 4c. Replace `run_iteration` method

```python
def run_iteration(
    self,
    iteration: int,
    *,
    changes: dict | None = None,
    hypothesis: str = "",
) -> BenchResult:
    """Run one inner iteration with bootstrap aggregation for noise resistance."""
    # --- Phase 1: Config proposal (unchanged) ---
    if changes is None:
        if self.client is None:
            raise ValueError("LLM client required when changes not provided")
        changes, hypothesis = self._propose(iteration)

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

    # --- Phase 2: Trial construction ---
    trial = self._trial_config(iteration)
    trial.apply_changes(filtered)
    
    # Check if we should replicate instead of accept
    best_key = self.config_cache._config_key(self.trace.best_config)
    trial_key = self.config_cache._config_key(trial)
    
    # If the proposed config is different from best, and best needs replicates,
    # interpolate a replicate run for the best config
    if (best_key != trial_key and 
        self.config_cache.needs_replicate(self.trace.best_config)):
        
        # Run a replicate of the best config first
        replicate_result = self._run_trial(
            self.trace.best_config, 
            iteration=iteration, 
            hypothesis=f"replicate best (iter {self.trace.best_iteration})"
        )
        replicate_result.changes = {}
        self.config_cache.add_result(self.trace.best_config, replicate_result)
        
        # Re-evaluate best config's aggregate
        agg_best = self.config_cache.get_aggregated(self.trace.best_config)
        if agg_best is not None:
            self.trace.best_val_bpb = agg_best
            self.trace.best_bpb = agg_best
        
        # Now run the proposed trial (at next iteration)
        # Return replicate as this iteration's result
        replicate_result.iteration = iteration
        replicate_result.status = "keep" if replicate_result.val_bpb < self.trace.best_bpb else "discard"
        replicate_result.accepted = replicate_result.status == "keep"
        self.trace.record(replicate_result)
        self.trace.results.append(replicate_result)
        return replicate_result
    
    # --- Phase 3: Original trial run ---
    result = self._run_trial(trial, iteration=iteration, hypothesis=hypothesis)
    result.changes = filtered
    self.config_cache.add_result(trial, result)
    
    # --- Phase 4: Aggregated acceptance decision ---
    # Get aggregated value for the proposed config
    agg_proposed = self.config_cache.get_aggregated(trial)
    
    # Get aggregated value for current best
    agg_best = self.config_cache.get_aggregated(self.trace.best_config)
    if agg_best is None:
        agg_best = self.trace.best_bpb
    
    # Decision logic using aggregated metrics
    if result.status == "crash":
        result.accepted = False
        result.status = "discard"
    elif agg_proposed is not None and agg_best is not None:
        # Use aggregated values for comparison
        if agg_proposed < agg_best - 0.001:  # small epsilon to avoid flip-flopping
            result.status = "keep"
            result.accepted = True
            self.trace.best_val_bpb = agg_proposed
            self.trace.best_bpb = agg_proposed
            self.trace.best_iteration = iteration
            self.trace.best_config = trial
            self.config = trial
        else:
            result.status = "discard"
            result.accepted = False
    else:
        # Fall back to single-run comparison if not enough replicates yet
        if result.val_bpb < self.trace.best_bpb:
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

### 5. Integration points

#### Pre-requisites
- `SearchConfig` needs new field `config_replicates: int = 2` with validator `field_validator('config_replicates')(lambda v: max(1, v))`
- Add import: `from pathlib import Path`

#### Trace changes
- `BenchTrace` results may now contain replicate runs with `changes={}` and hypothesis prefix `"replicate best..."` — downstream analytics should handle these
- Add method `BenchTrace.replicates()` to filter replicate entries (heuristic: `changes == {}` or hypothesis starts with `"replicate"`)

#### Artifacts
- `ConfigCache` optionally persists to `artifacts_dir / "config_cache.json"` — dump as dict of config_key → list of dicts after each iteration
- Load on init if file exists (for resuming experiments)

#### Testing
- Test: Run with `config_replicates=1` should behave identically to original
- Test: With `config_replicates=2`, if first run of config A gives 3.5, second gives 4.0, mean 3.75 wins against config B at 3.8
- Test: Best config that gets 3.5 initially, reset to 3.8 on replicate, should not trigger acceptance of worse configs until replicated view is stable

#### Performance note
- Each iteration may now run 2 trials (1 replicate + 1 proposed) — this doubles runtime but is bounded by `config_replicates` (max 2 for default)
- Total iterations remain fixed; the search simply spends more budget on confirmation

#### Error handling
- If `self.trace.best_config` has no replicates yet, `needs_replicate` returns True — we always replicate best at least once to get a stable baseline
- If replicate crashes, `needs_replicate` still returns True (crashes not counted as valid), preventing infinite loops by capping max replicates to `config_replicates * 2` with a warning log
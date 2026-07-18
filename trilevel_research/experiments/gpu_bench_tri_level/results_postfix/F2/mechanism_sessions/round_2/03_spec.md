## Implementation Specification

### 1. Mechanism name
`perturb_hidden_dim_sampler`

### 2. Implementation strategy
`new_helper_class` + `modify_init` + `replace_method`

### 3. Target
- `GpuBenchRunner.__init__` (modify)
- `GpuBenchRunner.run_iteration` (modify)
- New class: `HiddenDimPerturber`

### 4. Step-by-step logic

#### 4.1 Create new helper class `HiddenDimPerturber`

```python
@dataclass
class HiddenDimPerturber:
    """Samples hidden_dim near user-provided values using ±10% perturbation."""
    
    base_config: GpuBenchConfig
    rng: random.Random
    perturbation_range: float = 0.10  # ±10%
    active_params: set[str] = field(default_factory=lambda: {"HIDDEN_DIM"})
    
    def perturb(self, user_changes: dict) -> tuple[dict, str]:
        """Apply ±10% perturbation to hidden_dim if present in changes.
        
        Returns:
            - perturbed_changes: dict with HIDDEN_DIM slightly adjusted
            - perturbation_note: str describing the perturbation for hypothesis
        """
        hidden_key = None
        for key in user_changes:
            if key.upper() == "HIDDEN_DIM":
                hidden_key = key
                break
        
        if hidden_key is None:
            return user_changes, ""  # No hidden_dim change, return as-is
        
        original_value = user_changes[hidden_key]
        
        # Generate perturbation factor in range [1 - range, 1 + range]
        # Use log-uniform to keep relative scale
        log_factor = math.log1p(self.perturbation_range)  # ~ln(1.10)
        perturbation = math.exp(self.rng.uniform(-log_factor, log_factor))
        
        new_value = int(round(original_value * perturbation))
        # Enforce minimum hidden_dim (common constraint: ≥ 64)
        new_value = max(64, new_value)
        
        perturbed_changes = dict(user_changes)
        perturbed_changes[hidden_key] = new_value
        
        delta_pct = ((new_value - original_value) / original_value) * 100
        perturbation_note = f"±10%: {original_value}→{new_value} ({delta_pct:+.1f}%)"
        
        return perturbed_changes, perturbation_note
    
    def should_perturb(self, iteration: int, trace: BenchTrace) -> bool:
        """Decision logic: perturb unless too many consecutive failures.
        
        Skip perturbation if:
        - Last 3 perturbations all resulted in 'discard' or 'crash'
        - First iteration (baseline) — no perturbation
        """
        if iteration == 0:
            return False
        
        # Count recent perturbation outcomes
        recent = [r for r in trace.results[-5:] 
                 if "±10%" in (r.hypothesis or "")]
        if len(recent) >= 3:
            failures = sum(1 for r in recent[-3:] 
                          if r.status in ("discard", "crash"))
            if failures >= 3:
                return False  # Back off if failing consistently
        
        return True
```

#### 4.2 Modify `__init__`

Add initialization of `HiddenDimPerturber`:

```python
def __init__(
    self,
    bench_bin: Path | None = None,
    timeout_s: int = 120,
    llm_client: LLMClient | None = None,
    search_config: SearchConfig | None = None,
    artifacts_dir: Path | None = None,
    simple_mode: bool = True,
    seed: int = 42,  # NEW: reproducible random perturb
) -> None:
    # ... existing code ...
    self.simple_mode = simple_mode
    self.config = GpuBenchConfig()
    self.trace = BenchTrace()
    # NEW: Initialize perturber
    self.perturber = HiddenDimPerturber(
        base_config=self.config,
        rng=random.Random(seed),
        perturbation_range=0.10,
        active_params=set(self.search_config.active_params)
    )
```

#### 4.3 Modify `run_iteration`

Insert perturbation logic before the existing filtered-changes flow:

```python
def run_iteration(
    self,
    iteration: int,
    *,
    changes: dict | None = None,
    hypothesis: str = "",
) -> BenchResult:
    """Run one inner iteration. Applies HiddenDim perturbation when appropriate."""
    
    # === NEW: HiddenDim perturbation block ===
    if changes is not None and self.perturber.should_perturb(iteration, self.trace):
        perturbed_changes, pert_note = self.perturber.perturb(changes)
        if pert_note:
            # Only apply if actual perturbation occurred
            changes = perturbed_changes
            hypothesis = f"{hypothesis} | {pert_note}" if hypothesis else pert_note
    # === End perturbation block ===
    
    if changes is None:
        if self.client is None:
            raise ValueError("LLM client required when changes not provided")
        changes, hypothesis = self._propose(iteration)
    
    # Rest of existing method unchanged
    active = set(self.search_config.active_params)
    filtered = {k: v for k, v in changes.items() if k.upper() in active}
    # ... continue with existing code ...
```

### 5. Integration points

| Integration Point | Details |
|------------------|---------|
| **Config file** | Add `seed` to `SearchConfig` if not already present, default `42`. The perturbation RNG uses this for reproducibility. |
| **Trace output** | Modify `BenchResult.__repr__` or `__str__` to optionally show perturbation info. Not strictly required but helps debugging. |
| **CLI parameters** | Add `--seed` flag to benchmark runner's argument parser for reproducibility. |
| **SearchConfig** | Add `seed: int = 42` field. |

### Edge Cases & Safety Checks

1. **Non-hidden_dim changes pass through unchanged**: If user changes don't touch `HIDDEN_DIM`, `perturb()` returns the dict unchanged with empty note. Logic in `run_iteration` checks `pert_note` before applying.

2. **Minimum bound**: Perturbed value is clamped to `max(64, new_value)` to prevent degenerate tiny hidden dims.

3. **Consecutive failure backoff**: If last 3 perturbations all failed (discard or crash), `should_perturb` returns `False` until a non-perturbed iteration succeeds. This prevents infinite loops in bad regions.

4. **First iteration (baseline)**: Explicitly skipped via `iteration == 0` check.

5. **Log-uniform perturbation**: Uses exponential of uniform in log space, ensuring equal probability of +10% vs -9.1% (since scales are proportional).

6. **Seed reproducibility**: `random.Random(seed)` ensures same perturbation sequence across runs, critical for debugging.

### Testing Notes

- **Unit test for `perturb()`**: Verify that HIDDEN_DIM changes are perturbed by ~10%, other params unchanged,
          repeated calls give different values, clamping works.
- **Integration test**: Run with `simple_mode=True` and `active_params=["HIDDEN_DIM"]`, measure whether
          we explore values near 512 (e.g., 461, 563) vs jumping to 256/1024 directly.
- **End-to-end**: Compare baseline (no perturbation) vs perturbed search on hidden_dim sweeps;
          should show finer-grained sampling around peak regions.
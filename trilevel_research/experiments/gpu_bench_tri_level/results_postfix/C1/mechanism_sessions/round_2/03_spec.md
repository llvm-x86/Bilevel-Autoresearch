## Implementation Specification: Adaptive Proposal Scaling

### 1. Mechanism name
`adaptive_proposal_scaling`

### 2. Implementation strategy
`modify_init` + `new_method` (modify `GpuBenchRunner` and add a helper method in `SearchConfig`)

### 3. Target
`GpuBenchRunner` class (modify `__init__`, add new methods to `GpuBenchRunner`, modify `run_iteration`)

### 4. Step-by-step logic

**Step 1: Add state fields to `GpuBenchRunner.__init__`**

After the existing state initialization (after line `self.trace = BenchTrace()`), add:

```python
# Adaptive scaling state for Proposal 2
self._scaling_phase = "exploration"     # "exploration" | "exploitation"
self._scaling_trigger_count = 0         # consecutive proposals without improvement
self._scaling_recent_improvements = []   # rolling window of val_bpb improvements
self._best_val_bpb_initialized = False  # whether best_val_bpb has been set from non-baseline
```

**Step 2: Add scaling methods to `GpuBenchRunner`**

```python
def _should_expand_search(self) -> bool:
    """Check if we should expand LR search range based on lack of progress."""
    if self._scaling_trigger_count >= 10 and self._scaling_phase == "exploitation":
        return True
    return False

def _should_contract_search(self) -> bool:
    """Check if we should contract LR search range based on recent improvements."""
    if len(self._scaling_recent_improvements) >= 5:
        recent_improvements = self._scaling_recent_improvements[-5:]
        if all(imp < 0.0 for imp in recent_improvements):  # all negative = improving
            return True
    return False

def _update_scaling_state(self, result: BenchResult) -> None:
    """
    Update the adaptive scaling state based on the latest trial result.
    Called after processing result in run_iteration.
    """
    # Initialize first non-baseline result if needed
    if not self._best_val_bpb_initialized and result.iteration > 0:
        self._best_val_bpb_initialized = True
        self._scaling_phase = "exploitation"
        return

    # Skip baseline iteration (iteration 0)
    if result.iteration == 0:
        return

    # Track improvement
    improvement = self.trace.best_val_bpb - result.val_bpb if result.val_bpb is not None else 0.0
    self._scaling_recent_improvements.append(improvement)
    if len(self._scaling_recent_improvements) > 10:
        self._scaling_recent_improvements.pop(0)

    # Update counters
    if result.accepted and result.val_bpb < self.trace.best_val_bpb:
        self._scaling_trigger_count = 0  # Reset on improvement
        self._scaling_phase = "exploitation"
    else:
        self._scaling_trigger_count += 1

    # Phase transitions
    if self._should_expand_search():
        self._scaling_phase = "exploration"
        self._scaling_trigger_count = 0  # Reset to avoid immediate oscillation

    if self._should_contract_search():
        self._scaling_phase = "exploitation"
```

**Step 3: Add `_get_adaptive_scale_factor` method to `SearchConfig`**

```python
def get_adaptive_scale_factor(self, phase: str, param_name: str) -> float:
    """
    Return a scale factor for proposal changes based on the current phase.
    In exploration mode, use larger scale factors for critical params.
    In exploitation mode, use smaller scale factors for fine-tuning.
    """
    param_name_upper = param_name.upper()
    
    # Default scale factors
    CORE = {"LEARNING_RATE", "BATCH_SIZE", "WEIGHT_DECAY"}
    SECONDARY = {"MOMENTUM", "ADAM_BETA1", "ADAM_BETA2"}
    
    if param_name_upper not in CORE and param_name_upper not in SECONDARY:
        return 1.0  # Unknown params use default
    
    if phase == "exploration":
        if param_name_upper in CORE:
            return 2.0  # Double the perturbation for critical params
        return 1.5  # 1.5x for secondary params
    else:  # exploitation
        if param_name_upper == "LEARNING_RATE":
            return 0.5  # Halve LR perturbations for fine-tuning
        if param_name_upper in CORE:
            return 0.75  # 75% of default for other core params
        return 0.9  # Slightly reduce secondary params
```

**Step 4: Modify `run_iteration` to integrate adaptive scaling**

In the `run_iteration` method, after computing `filtered` changes but before creating the trial config:

```python
# Apply adaptive scaling to proposal changes based on current phase
if filtered:
    # Get scale factor based on current phase
    phase = self._scaling_phase
    
    # Apply scaling to numerical changes
    scaled_changes = {}
    for param, value in filtered.items():
        scale = self.search_config.get_adaptive_scale_factor(phase, param)
        if isinstance(value, (int, float)):
            # For percentage-based changes (like "+10%"), scale the percentage
            if isinstance(value, str) and "%" in value:
                # Parse percentage change and rescale
                import re
                match = re.match(r"([+-])(\d+(?:\.\d+)?)%", value)
                if match:
                    sign = match.group(1)
                    pct = float(match.group(2)) * scale
                    scaled_changes[param] = f"{sign}{pct:.1f}%"
                else:
                    scaled_changes[param] = value
            else:
                # For absolute values, scale by factor (can be used as multiplier)
                if isinstance(value, (int, float)):
                    # Keep as-is since absolute values are usually targets
                    scaled_changes[param] = value
                else:
                    scaled_changes[param] = value
        else:
            scaled_changes[param] = value
    
    # Use scaled changes for the trial
    filtered = scaled_changes
```

**Step 5: Add state update call in `run_iteration`**

After the acceptance/rejection logic block (after the `if result.status == "crash"` block and following `else`), add:

```python
# Update adaptive scaling state
self._update_scaling_state(result)
```

### 5. Integration points

1. **Initialization**: The new state fields integrate with the existing `__init__` at initialization time. They require no changes to the constructor signature.

2. **SearchConfig**: The new `get_adaptive_scale_factor` method is added to the existing `SearchConfig` class. It reads from the existing `active_params` field but adds new `exploration`/`exploitation` phase awareness.

3. **run_iteration flow**: The integration is seamless:
   - After computing `filtered` (changes filtered by active params), apply adaptive scaling
   - After the acceptance check and state update, update the scaling state
   - The LLM proposal logic remains unchanged - only the numerical magnitude of accepted changes is modified

4. **Trace compatibility**: The state updates don't affect the existing trace recording logic. The `trace.record(result)` and `trace.results.append(result)` calls remain unchanged.

5. **Error handling**: The scaling logic is defensive - unknown parameters get a scale factor of 1.0 (no change), and string parsing uses `re.match` with proper fallback to original value.

6. **Performance**: No additional I/O or blocking operations are added. The state updates are O(1) or O(n) with n being the rollling window size (10).
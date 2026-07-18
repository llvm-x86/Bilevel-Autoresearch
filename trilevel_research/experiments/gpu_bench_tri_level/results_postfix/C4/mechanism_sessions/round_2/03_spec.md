# Implementation Specification: `adaptive_perturbation_engine`

## 1. Mechanism Name
`adaptive_perturbation_with_global_decay`

## 2. Implementation Strategy
**replace_method** — Replace the inner `HyperparameterPerturbationEngine.generate_perturbation` method to implement adaptive perturbation sizing with global iteration-based decay.

## 3. Target
- Class: `HyperparameterPerturbationEngine` (inner class of `GpuBenchRunner.__init__`)
- Method: `generate_perturbation`

## 4. Step-by-step Logic

### 4.1 New Configuration Parameters (modify `__init__`)
Add to `SearchConfig` or as new attributes:
```python
# Perturbation scaling
self.min_perturbation_factor = 0.1  # Minimum perturbation magnitude
self.max_perturbation_factor = 2.0  # Maximum perturbation magnitude
self.decay_rate = 0.95  # Per-iteration decay of perturbation magnitude
self.global_start_iteration = 0  # Track when best config was last updated
```

### 4.2 Store Global Iteration Counter
Modify `GpuBenchRunner.__init__` to add:
```python
self.global_iteration_count = 0  # Tracks total perturbations attempted
```

### 4.3 New `generate_perturbation` Method Logic

```python
def generate_perturbation(self, current_config, best_config, iteration: int) -> dict:
    import random
    import numpy as np
    
    changes = {}
    
    # Calculate decay factor based on global iteration count
    iterations_since_best_update = self.global_iteration_count - best_config.last_updated_iteration
    decay_factor = self.decay_rate ** iterations_since_best_update
    
    # Calculate effective perturbation factor
    perturbation_factor = (
        self.max_perturbation_factor * (1 - decay_factor) +
        self.min_perturbation_factor * decay_factor
    )
    
    # Ensure perturbation_factor stays within bounds
    perturbation_factor = max(self.min_perturbation_factor, 
                              min(self.max_perturbation_factor, perturbation_factor))
    
    # Select parameter to perturb (weight toward LR and BATCH_SIZE initially,
    # then broaden exploration as iterations increase)
    if iteration < 10:
        param_weights = {'LR': 0.4, 'BATCH_SIZE': 0.4, 'OPTIMIZER': 0.2}
    elif iteration < 30:
        param_weights = {'LR': 0.3, 'BATCH_SIZE': 0.3, 'OPTIMIZER': 0.4}
    else:
        param_weights = {'LR': 0.2, 'BATCH_SIZE': 0.2, 'OPTIMIZER': 0.6}
    
    # Sample parameter based on weights
    param = random.choices(
        list(param_weights.keys()),
        weights=list(param_weights.values())
    )[0]
    
    if param == 'LR':
        # Use log-uniform sampling with adaptive range
        current_lr = current_config.lr
        
        # Calculate log-scale perturbation
        log_lr = np.log10(current_lr)
        perturbation_width = np.log10(perturbation_factor)  # Convert to log scale
        
        # Sample in log space
        new_log_lr = log_lr + random.uniform(-perturbation_width, perturbation_width)
        
        # Clip to valid range [1e-6, 1.0]
        new_log_lr = max(np.log10(1e-6), min(np.log10(1.0), new_log_lr))
        
        changes['LR'] = 10.0 ** new_log_lr
    
    elif param == 'BATCH_SIZE':
        # Use power-of-two perturbation for batch size
        current_bs = current_config.batch_size
        log2_bs = np.log2(current_bs)
        
        # Calculate allowed shift in log2 space
        max_shift = max(1, int(np.abs(np.log2(perturbation_factor)) + 0.5))
        
        # Random shift
        shift = random.randint(-max_shift, max_shift)
        new_log2_bs = log2_bs + shift
        
        # Clip to reasonable range [4, 1024]
        new_log2_bs = max(2, min(10, int(new_log2_bs)))
        
        changes['BATCH_SIZE'] = 2 ** new_log2_bs
    
    elif param == 'OPTIMIZER':
        # Only perturb optimizer less frequently as we approach optimum
        if decay_factor > 0.5:  # High decay factor means we're near optimum
            return self._conservative_optimizer_perturb(current_config)
        else:
            return self._exploratory_optimizer_perturb(current_config)
    
    return changes
```

### 4.4 Helper Methods

```python
def _conservative_optimizer_perturb(self, current_config):
    """When near optimum, only try similar optimizer variants"""
    changes = {}
    if current_config.optimizer in ['adam', 'adamw']:
        # Stay in Adam family
        alternatives = {
            'adam': ['adamw', 'adam_amsgrad'],
            'adamw': ['adam', 'adam_amsgrad'],
        }
        base = current_config.optimizer
        changes['OPTIMIZER'] = random.choice(alternatives.get(base, [base]))
    elif current_config.optimizer == 'sgd':
        # Try SGD with momentum
        changes['OPTIMIZER'] = 'sgd_momentum'  # if available
    return changes

def _exploratory_optimizer_perturb(self, current_config):
    """When far from optimum, try any optimizer"""
    changes = {}
    current_opt = current_config.optimizer
    all_opts = ['adam', 'adamw', 'sgd', 'adagrad', 'rmsprop']
    alternatives = [opt for opt in all_opts if opt != current_opt]
    changes['OPTIMIZER'] = random.choice(alternatives)
    return changes
```

### 4.5 Integration into Main Loop

Modify the perturbation generation call in `GpuBenchRunner`:
```python
# Before perturbation
self.global_iteration_count += 1

# Generate perturbation with access to global state
perturbation = engine.generate_perturbation(
    current_config=current_config,
    best_config=self.best_config,  # Pass best config for reference
    iteration=self.global_iteration_count
)
```

## 5. Integration Points

### 5.1 Files Modified
1. **`runner.py`**: Replace `HyperparameterPerturbationEngine.generate_perturbation` method
2. **`search_config.py`** (if it exists): Add new configuration parameters for perturbation scaling

### 5.2 Dependencies
- `numpy` (already imported) for log operations
- No new external dependencies

### 5.3 Backward Compatibility
- All existing `GpuBenchRunner` API remains unchanged
- New configuration parameters have sensible defaults that match current behavior if not specified
- The `simple_mode` flag can still be respected to use simpler perturbations if needed

### 5.4 Testing Considerations
1. **Verify decay behavior**: Test that perturbation magnitude decreases over iterations when best config is stable
2. **Verify best-update reset**: Test that `last_updated_iteration` is properly tracked and perturbation magnitude doesn't increase when best config is updated
3. **Test edge cases**: Minimum perturbation factor at very high iteration counts
4. **Regression test**: Val_bpb=0.0052 configuration should trigger conservative local refinement after ~10 iterations

### 5.5 Risk Mitigation
- **Risk**: Decay might be too aggressive, preventing necessary exploration
  - **Mitigation**: Add safeguard: if `val_bpb` improvement < 0.01 after 20 consecutive iterations, temporarily increase `perturbation_factor` by 50%
- **Risk**: Global iteration counter might not properly reset when best config changes
  - **Mitigation**: Store timestamp of last best-update in config object itself, not in runner state

### 5.6 Monitoring Points
Add logging for:
- Current perturbation_factor value per iteration
- Number of iterations since last best-config update
- Whether perturbation was exploratory or conservative (based on decay factor)
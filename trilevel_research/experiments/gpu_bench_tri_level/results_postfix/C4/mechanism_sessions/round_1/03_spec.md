## Implementation Specification: Wider Perturbation Sampling

1. **Mechanism name**: `wider_perturbation_sampling`

2. **Implementation strategy**: `modify_init` + `new_helper_class`

3. **Target**: `GpuBenchRunner.__init__` and new helper class `HyperparameterPerturbationEngine`

4. **Step-by-step logic**:

   **Step 1: Add perturbation configuration to `__init__`**
   ```python
   # In __init__, after existing initialization:
   self.perturbation_config = {
       'LR': {'min_factor': 0.1, 'max_factor': 10.0},  # 10x range instead of ~2x
       'BATCH_SIZE': {
           'min_factor': 0.25,  # divide by 4
           'max_factor': 4.0    # multiply by 4
       },
       'OPTIMIZER': {
           'alternatives': ['AdamW', 'Adam', 'SGD'],
           'must_switch': False  # sometimes keep optimizer, vary other params
       },
       'random_seed': {
           'strategy': 'log_uniform',  # sample from log-uniform distribution
           're_seed_probability': 0.3  # 30% chance to also change random seed
       }
   }
   
   # Track parameters that were recently changed (last 3 iterations)
   self.recent_changes = {'LR': [], 'BATCH_SIZE': [], 'OPTIMIZER': []}  # param -> [iteration_numbers]
   self.iteration_counter = 0
   ```

   **Step 2: Create `HyperparameterPerturbationEngine`**
   ```python
   class HyperparameterPerturbationEngine:
       def __init__(self, config: dict, recent_changes: dict):
           self.config = config
           self.recent_changes = recent_changes
       
       def generate_perturbation(self, current_config: GpuBenchConfig, 
                                  best_config: GpuBenchConfig,
                                  iteration: int) -> dict:
           """Generate wider-range perturbations with strategic exploration."""
           
           # New logic:
           # 1. 30% chance: try a completely different regime (large jump)
           # 2. 40% chance: perturb one parameter significantly (3-10x)
           # 3. 30% chance: try tuning a parameter that hasn't been changed lately
           
           strategy = random.choices(
               ['large_jump', 'significant_perturb', 'neglected_param'],
               weights=[0.3, 0.4, 0.3]
           )[0]
           
           changes = {}
           
           if strategy == 'large_jump':
               # Pick one parameter to change dramatically
               param = random.choice(self.config.keys())
               if param == 'LR':
                   changes['LR'] = self._log_uniform_sample(
                       current_config.lr / 10,  # 10x lower
                       current_config.lr * 10    # 10x higher
                   )
               elif param == 'BATCH_SIZE':
                   # Round to power of 2
                   factors = [0.125, 0.25, 0.5, 2.0, 4.0, 8.0]
                   factor = random.choice(factors)
                   new_bs = int(current_config.batch_size * factor)
                   # Round to nearest power of 2
                   changes['BATCH_SIZE'] = 2 ** int(round(np.log2(new_bs)))
               elif param == 'OPTIMIZER':
                   current_opt = current_config.optimizer
                   alternatives = [opt for opt in self.config['OPTIMIZER']['alternatives'] 
                                  if opt != current_opt]
                   changes['OPTIMIZER'] = random.choice(alternatives)
           
           elif strategy == 'significant_perturb':
               # Perturb one parameter significantly (not just tiny tweaks)
               param = random.choice(['LR', 'BATCH_SIZE'])
               if param == 'LR':
                   # 3-10x change, not 1.2x
                   factor = random.choice([0.1, 0.2, 0.33, 3.0, 5.0, 10.0])
                   changes['LR'] = current_config.lr * factor
               else:  # BATCH_SIZE
                   factors = [0.25, 0.5, 2.0, 4.0]
                   factor = random.choice(factors)
                   new_bs = int(current_config.batch_size * factor)
                   changes['BATCH_SIZE'] = 2 ** int(round(np.log2(new_bs)))
           
           else:  # neglected_param
               # Find parameter not changed in last 3 iterations
               param_ages = {}
               for param in self.config.keys():
                   if param in ['random_seed', 'OPTIMIZER']:
                       continue
                   history = self.recent_changes.get(param, [])
                   if not history:
                       param_ages[param] = 999  # Never changed
                   else:
                       param_ages[param] = iteration - max(history)
               
               # Pick the most neglected parameter
               neglected = max(param_ages, key=param_ages.get)
               
               if neglected == 'LR':
                   # When revisiting LR, ensure significant change
                   changes['LR'] = current_config.lr * random.choice([0.2, 0.33, 3.0, 5.0])
               else:  # BATCH_SIZE
                   changes['BATCH_SIZE'] = 2 ** int(round(np.log2(
                       current_config.batch_size * random.choice([0.25, 0.5, 2.0, 4.0])
                   )))
           
           return changes
       
       def _log_uniform_sample(self, low, high):
           """Sample from log-uniform distribution between low and high."""
           log_low = np.log(low)
           log_high = np.log(high)
           return np.exp(np.random.uniform(log_low, log_high))
   ```

   **Step 3: Modify `run_iteration` to use wider perturbations**
   ```python
   # After the existing `if self.client is None` check, but before:
   # changes, hypothesis = self._propose(iteration)
   
   # Replace the propose call:
   if changes is None:
       if self.client is None:
           raise ValueError("LLM client required when changes not provided")
       
       # 70% chance: use perturbation engine for wider exploration
       # 30% chance: use LLM for guided suggestions
       if random.random() < 0.7 or self.iteration_counter < 3:
           # Use perturbation engine
           engine = HyperparameterPerturbationEngine(
               self.perturbation_config,
               self.recent_changes
           )
           changes = engine.generate_perturbation(
               self.config,
               self.trace.best_config,
               self.iteration_counter
           )
           hypothesis = f"Wider perturbation: {changes}"
       else:
           # Use LLM but with explicit instruction to explore wider
           changes, hypothesis = self._propose(iteration)
   
   # Track changes for future reference
   for param in changes:
       if param in self.recent_changes:
           self.recent_changes[param].append(self.iteration_counter)
           # Keep only last 3 changes
           self.recent_changes[param] = self.recent_changes[param][-3:]
   
   self.iteration_counter += 1
   ```

   **Step 4: Add safety bounds to prevent unreasonable values**
   ```python
   def _clamp_perturbation(self, changes: dict) -> dict:
       """Ensure perturbations stay within reasonable bounds."""
       clamped = {}
       for param, value in changes.items():
           if param == 'LR':
               clamped['LR'] = max(1e-6, min(1.0, value))  # Safe LR range
           elif param == 'BATCH_SIZE':
               # Ensure power of 2 and reasonable range
               value = max(1, min(2048, value))
               clamped['BATCH_SIZE'] = 2 ** int(round(np.log2(value)))
           elif param == 'OPTIMIZER':
               # Only allow known optimizers
               valid_optimizers = ['AdamW', 'Adam', 'SGD', 'Adamax']
               if value in valid_optimizers:
                   clamped['OPTIMIZER'] = value
               else:
                   clamped['OPTIMIZER'] = 'AdamW'  # fallback
           else:
               clamped[param] = value
       return clamped
   ```

5. **Integration points**:

   - **Import**: Add `import random` and `import numpy as np` at top of file
   - **`__init__`**: Add perturbation configuration, iteration counter, and recent changes tracking. These are initialized once and persist across all runner calls.
   - **`run_iteration`**: The perturbation engine replaces some LLM calls but preserves the existing loop logic (run trial, evaluate result, update trace). The engine runs before `_propose()`, meaning wide exploration happens before guided search.
   - **`_trial_config`**: No changes needed; it already applies changes via `apply_changes()`.
   - **`BenchTrace`**: No changes needed; it still records results as before.
   - **Edge case**: When changing `BATCH_SIZE`, ensure the value is compatible with GPU memory. Add a check in `_run_trial` to detect OOM crashes. If crash occurs, mark as "crash" and the runner will already handle it.
   - **Existing behavior preserved**: If `self.client is None` or `changes` is explicitly provided, the new logic doesn't interfere. The LLM can still propose changes in the 30% case.
   - **Testing**: Unit test the `HyperparameterPerturbationEngine.generate_perturbation()` in isolation to verify it produces wider ranges than current logic. Integration test to verify it integrates without breaking existing "accept/discard" cycle.
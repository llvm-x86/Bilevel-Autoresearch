## Implementation Specification

1. **Mechanism name**: `restart_with_higher_variance`

2. **Implementation strategy**: `new_method` + `modify_init`

3. **Target**: `GpuBenchRunner` class (`__init__` and new method `_handle_stuck_state`)

4. **Step-by-step logic**:

   **A. Modify `__init__` to add state tracking**:
   - Add `self._iteration_since_improvement = 0` counter
   - Add `self._best_score = float('-inf')` to track best observed score
   - Add `self._stuck_threshold = 4` (iterations without improvement before triggering restart)
   - Add `self._restart_count = 0` to track number of restarts
   - Add `self._last_restart_iteration = -1` to prevent restart loops

   **B. New method `_handle_stuck_state(self, current_score: float) -> bool`**:
   ```
   1. If current_score > self._best_score:
      a. Update self._best_score = current_score
      b. Reset self._iteration_since_improvement = 0
      c. Return False (not stuck)
   
   2. If current_score <= self._best_score:
      a. Increment self._iteration_since_improvement += 1
      b. If self._iteration_since_improvement >= self._stuck_threshold:
         - Check if (self._iteration_number - self._last_restart_iteration) >= 3 (anti-oscillation guard)
         - If guard passes:
           * Increment self._restart_count
           * Set self._last_restart_iteration = self._iteration_number
           * Apply restart logic (step C)
           * Return True (stuck state handled)
         - Else: Return False (cannot restart yet)
   
   3. Return False (not stuck or already handled)
   ```

   **C. Restart logic (inside `_handle_stuck_state`)**:
   ```
   When restart triggered:
   1. Calculate new variance multiplier:
      variance_mult = min(4.0 * (1.25 ** self._restart_count), 20.0)
   
   2. Apply to config perturbation:
      a. For LR: new_perturb = uniform(-0.5 * variance_mult, 0.5 * variance_mult) * current_LR
      b. For WEIGHT_DECAY: new_perturb = 10^(-6 + uniform(0, 2 * variance_mult))
      c. For BATCH_SIZE: new_perturb = round(uniform(-32 * variance_mult, 32 * variance_mult))
      d. For HIDDEN_DIM: new_perturb = round(uniform(-64 * variance_mult, 64 * variance_mult))
   
   3. Clip all parameters to valid ranges:
      - LR: max(0.0001, min(0.1, new_LR))
      - WEIGHT_DECAY: max(1e-8, min(0.1, new_WD))
      - BATCH_SIZE: max(16, min(512, new_BS))
      - HIDDEN_DIM: max(64, min(1024, new_HD))
   
   4. Reset dedup validator's seen configs to prevent immediate rejection
      - Call self.dedup_validator.reset() (need to store reference)
   
   5. Set flag: self._restart_applied = True
   ```

   **D. Modify the main optimization loop** (where configurations are generated):
   ```
   After each evaluation (score received):
   1. Call self._handle_stuck_state(evaluated_score)
   2. If restart was triggered and applied:
      a. Add the new high-variance config to the next iteration's candidates
      b. Clear any pending configs from previous stuck iterations
   3. Continue normal iteration flow
   ```

5. **Integration points**:

   **In `__init__`**:
   ```python
   # Add after existing __init__ code
   self._stuck_tracker = {
       'iteration_since_improvement': 0,
       'best_score': float('-inf'),
       'stuck_threshold': 4,
       'restart_count': 0,
       'last_restart_iteration': -1,
       'restart_applied': False,
   }
   # Store dedup_validator reference if not already stored
   # Assume dedup_validator exists after line ~166
   ```

   **In iteration loop** (likely `run` method or `_evaluate_config`):
   ```python
   # After evaluating a config and getting score:
   score = evaluate_result.score
   if score > self._stuck_tracker['best_score']:
       self._stuck_tracker['best_score'] = score
       self._stuck_tracker['iteration_since_improvement'] = 0
   else:
       self._stuck_tracker['iteration_since_improvement'] += 1
       if (self._stuck_tracker['iteration_since_improvement'] >= 
           self._stuck_tracker['stuck_threshold']):
           if (iteration_num - self._stuck_tracker['last_restart_iteration'] >= 3):
               apply_restart()
               self._stuck_tracker['restart_count'] += 1
               self._stuck_tracker['last_restart_iteration'] = iteration_num
   ```

   **Dedup validator modification**:
   ```python
   # Add reset method to ConfigDedupValidator
   def reset(self) -> None:
       self._seen_configs.clear()
   ```

   **Config generation integration**:
   ```python
   # In config generation step (probably _propose_config or similar)
   if self._stuck_tracker['restart_applied']:
       # Use higher variance perturbation
       config = self._generate_high_variance_config()
       self._stuck_tracker['restart_applied'] = False
   else:
       config = normal_generation_logic()
   ```

This implementation assumes there's a main optimization loop that iterates over eval-configs, evaluates them, and tracks scores. The key modification is adding state awareness for stagnation and triggering controlled, escalating variance when normal exploration fails to find improvements.
1. **Mechanism name** (snake_case): conservative_candidate_generation

2. **Implementation strategy**: new_method + modify_init + modify_run_iteration

3. **Target**: GpuBenchRunner class

4. **Step-by-step logic**:

   **A. INIT CHANGES** (`__init__`):
   - Add `self.conservative_mode = True` after `self.simple_mode` initialization
   - Add `self.min_step_fraction = 0.2` (minimum step size as fraction of current parameter range)
   - Add `self.conservative_iterations = 0` (counter for conservative proposals accepted)

   **B. NEW HELPER METHODS**:

   ```python
   def _compute_conservative_candidate(self, llm_changes: dict) -> dict:
       """
       Given raw LLM changes, produce a conservative version that:
       1. Caps any parameter change to at most 50% of current value
       2. Ensures at least one parameter changes (to prevent stagnation)
       3. Prefers small perturbations over large jumps
       """
       conservative = {}
       current = self.config.to_dict()
       
       # First, filter to active params only
       active = set(self.search_config.active_params)
       filtered_changes = {k.upper(): v for k, v in llm_changes.items() 
                          if k.upper() in active}
       
       # If no active changes, generate a minimal perturbation
       if not filtered_changes:
           # Pick one random active param and nudge it by 10%
           import random
           param = random.choice(list(active))
           param_lower = param.lower()
           current_val = getattr(self.config, param_lower, None)
           if current_val is not None:
               delta = current_val * 0.1
               new_val = current_val + delta
               # Clamp to valid range
               min_val, max_val = self.search_config.param_ranges.get(
                   param, (0.0, 1.0)
               )
               new_val = max(min_val, min(max_val, new_val))
               conservative[param_lower] = round(new_val, 6)
           return conservative
       
       # Apply conservative constraints to each proposed change
       for param, proposed_value in filtered_changes.items():
           param_lower = param.lower()
           current_val = getattr(self.config, param_lower, None)
           if current_val is None:
               continue
               
           # Get valid range for this parameter
           min_val, max_val = self.search_config.param_ranges.get(
               param, (0.0, float('inf'))
           )
           param_range = max_val - min_val
           if param_range <= 0:
               continue
               
           # Calculate proposed change magnitude
           proposed_change = abs(proposed_value - current_val)
           max_allowed_change = max(
               param_range * self.min_step_fraction,  # At least 20% of range
               current_val * 0.5  # At most 50% of current value
           )
           
           # Cap the change
           if proposed_change > max_allowed_change:
               direction = 1 if proposed_value > current_val else -1
               new_value = current_val + direction * max_allowed_change
               new_value = max(min_val, min(max_val, new_value))
               conservative[param_lower] = round(new_value, 6)
           else:
               conservative[param_lower] = round(proposed_value, 6)
       
       # Ensure at least one change is made
       if not conservative and filtered_changes:
           # Pick the first param and make a minimal change
           param = list(filtered_changes.keys())[0]
           param_lower = param.lower()
           current_val = getattr(self.config, param_lower)
           min_val, max_val = self.search_config.param_ranges.get(
               param, (0.0, 1.0)
           )
           step = (max_val - min_val) * 0.05  # 5% of range
           new_val = current_val + step
           new_val = max(min_val, min(max_val, new_val))
           conservative[param_lower] = round(new_val, 6)
       
       return conservative
   ```

   **C. MODIFIED RUN_ITERATION** (`run_iteration`):

   - After `filtered` is computed from `changes` (when `changes` is provided by LLM), add:
   ```python
   if self.conservative_mode and changes is not None and not isinstance(changes, dict):
       # Only apply conservative mode during LLM-generated proposals
       filtered = self._compute_conservative_candidate(filtered)
       hypothesis = f"conservative_{hypothesis}"
       self.conservative_iterations += 1
   ```

   - When result is `keep`, add a check:
   ```python
   if result.status == "keep" and self.conservative_iterations > 5:
       # After 5 conservative iterations without improvement, allow one bold move
       self.conservative_mode = False
   ```

   **D. STATE RESET**:

   - After a `keep` result, reset `self.conservative_iterations = 0`
   - After a `discard` result, increment counter and if > 3 consecutive discards, temporarily disable conservative mode for one iteration (allow LLM bold proposal)

5. **Integration points**:

   - **In `__init__`**: Add initialization of conservative mode flags and counters after existing parameter initialization
   
   - **In `run_iteration`**: Insert conservative check after `filtered` computation but before `trial.apply_changes(filtered)`. The modification is minimal—just passing `filtered` through `_compute_conservative_candidate`
   
   - **No changes needed to**: `run_baseline`, `_run_trial`, `BenchTrace`, `GpuBenchConfig`—these remain untouched
   
   - **SearchConfig dependency**: Requires `param_ranges` dictionary to exist (should already be present for validation)
   
   - **LLM interaction**: The conservative layer sits between LLM output and actual parameter application, so LLM behavior is preserved but moderated

   **Edge cases handled**:
   - All LLM changes filtered out → generate minimal random perturbation
   - Parameter at boundary of valid range → clamp safely
   - Consecutive discards → temporarily allow bold exploration to escape local minima
   - Zero-range parameters → skip (shouldn't happen but defensive)
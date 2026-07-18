## Implementation Specification: Expand Search Radius After Discards

1. **Mechanism name**: `expand_radius_on_discards`

2. **Implementation strategy**: `modify_init` + `new_helper_class` + `modify_method`

3. **Target**: `GpuBenchRunner.__init__`, new class `DiscardTracker`, `GpuBenchRunner.run_iteration`

4. **Step-by-step logic**:

   **A. In `__init__`: Add discard tracking state**
   - Add `self.discard_tracker = DiscardTracker()` after `self.trace = BenchTrace()`
   - This will track consecutive discards and manage radius expansion

   **B. New helper class `DiscardTracker`:**
   ```python
   class DiscardTracker:
       def __init__(self, base_radius: float = 0.1, expansion_factor: float = 1.5):
           # base_radius: starting perturbation radius as fraction of param range
           # expansion_factor: multiply radius after consecutive discards
           self.consecutive_discards = 0
           self.current_radius = base_radius  # e.g., 0.1 = 10% of parameter range
           self.base_radius = base_radius
           self.expansion_factor = expansion_factor
           self.max_radius = 0.5  # cap at 50% to prevent wild jumps
       
       def record_result(self, status: str, config: GpuBenchConfig) -> None:
           """Track consecutive discards and compute new radius."""
           if status == "discard":
               self.consecutive_discards += 1
               # Exponential growth: each discard multiplies radius
               expansion = self.expansion_factor ** min(self.consecutive_discards, 5)  # cap at 5 consecutive
               self.current_radius = min(
                   self.base_radius * expansion,
                   self.max_radius
               )
           else:  # keep or crash resets
               self.consecutive_discards = 0
               self.current_radius = self.base_radius
       
       def get_perturbation_range(self, param_name: str, config: GpuBenchConfig) -> float:
           """Return absolute perturbation range for a parameter."""
           # Get parameter range from config's known ranges
           ranges = {
               'LEARNING_RATE': (0.0001, 0.01),  # log scale handled separately
               'HIDDEN_DIM': (32, 512),
               'BATCH_SIZE': (16, 128),
           }
           low, high = ranges[param_name]
           param_range = high - low
           return param_range * self.current_radius
   ```

   **C. In `run_iteration`: Modify discard logic to expand radius**
   - After the existing discard logic block (lines ~74-77), add a call to `self.discard_tracker.record_result()`
   - Store the current radius in the result for debugging visibility
   - **Critical integration point**: The `_propose` method (not shown, used for LLM proposals) should read `self.discard_tracker.current_radius` when generating new parameter suggestions

5. **Integration points**:

   **Point 1**: After `result.status = "discard"` assignment (line 77), insert:
   ```python
   self.discard_tracker.record_result("discard", trial)
   result.current_radius = self.discard_tracker.current_radius
   ```
   
   **Point 2**: After `result.status = "keep"` assignment (line 70), insert:
   ```python
   self.discard_tracker.record_result("keep", trial)
   result.current_radius = self.discard_tracker.current_radius
   ```
   
   **Point 3**: In `run_baseline`, after `self.trace.best_config = GpuBenchConfig()`, add:
   ```python
   self.discard_tracker = DiscardTracker()  # Reset tracker for new optimization run
   ```
   
   **Point 4**: The `DiscardTracker` must be passed or accessible from the LLM client's `_propose` method. Since `_propose` accesses `self.search_config`, add a new field:
   ```python
   # In SearchConfig or as a property of GpuBenchRunner
   @property
   def current_radius(self) -> float:
       return getattr(self.discard_tracker, 'current_radius', 0.1)
   ```
   Then the LLM prompt template should include: "Current search radius: {radius}% of parameter range" so the LLM knows to suggest wider or narrower changes.

   **Point 5**: Add to `BenchResult` class a new optional field `current_radius: float = 0.1` for trace logging.

   **Edge Cases**:
   - If all params have been explored and radius reaches max, log a warning: "Search radius at max (50%). Consider restarting with new seed config."
   - On crash/error, reset consecutive counter (don't penalize for system failures)
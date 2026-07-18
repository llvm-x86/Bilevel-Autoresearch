## Implementation Specification for Momentum-Based Exploration

1. **Mechanism name**: momentum_batch_exploration

2. **Implementation strategy**: modify_run_iteration_with_momentum

3. **Target**: `run_iteration` method in `GpuBenchRunner`

4. **Step-by-step logic**:

   a. **Add state tracking to `run_iteration`**:
   - Track the last successful direction of change by comparing `changes` from the previous accepted iteration with the current config
   - Initialize a `momentum_direction` dictionary as `{}` at the start of each series of iterations (reset when hypothesis changes)
   - Track `consecutive_improvements` counter per parameter

   b. **Modify acceptance logic in `run_iteration`**:
   - After computing `result.val_bpb`, if the result is accepted (`result.val_bpb < self.trace.best_bpb`):
     - For each parameter in `filtered` that shows a monotonic improvement, record the direction of change (+1 or -1)
     - Increment `consecutive_improvements` for that parameter
     - Store `momentum_direction[param] = direction` and `momentum_strength[param] = consecutive_improvements`
   - If the result is rejected (`result.val_bpb >= self.trace.best_bpb`):
     - Reset `consecutive_improvements` to 0 for all changed parameters
     - Do NOT reset `momentum_direction` — keep it for the next proposal

   c. **Modify proposal logic when `changes is None`**:
   - Before calling `self._propose(iteration)`, inject momentum hints:
     - Create a `momentum_hints` dictionary: `{param: (direction, strength)}` for params with `momentum_strength[param] >= 2`
     - Pass this to `_propose` via a new keyword argument `momentum_hints=`
   - In `_propose`, use momentum hints to bias the LLM proposal:
     - If `momentum_hints` is non-empty, include a prompt extension: "The previous {n} improvements used direction {direction} for parameter {param}. Continue in the same direction with a similar magnitude."
     - Do NOT override the LLM; just provide as context

   d. **Handle edge cases**:
   - If momentum direction contradicts the current proposal (e.g., momentum says +1 but LLM proposes -1), do NOT force the direction — trust the LLM but log the contradiction
   - Reset momentum state when iteration count exceeds `search_config.max_iterations` or when a new hypothesis is generated

5. **Integration points**:

   - **`run_iteration` signature**: Add `momentum_hints: dict | None = None` parameter before `changes` parameter (or as a keyword-only arg)
   - **`_propose` method**: Modify to accept and use `momentum_hints`. The prompt injection should be minimal — a single sentence appended to the existing hypothesis prompt
   - **Storage**: Add `self._momentum_state: dict = {}` as instance variable in `__init__` 
   - **Reset logic**: In `run_baseline`, reset `self._momentum_state = {}` to start fresh for each optimization run
   - **Trace integration**: Optionally log momentum state in `BenchTrace` for debugging (store as a list of dicts per iteration)
   - **Configuration**: No new config parameters needed — momentum is always-on when available but self-correcting if it leads to worse proposals

**Key safety mechanism**: The momentum is a soft hint, not a hard override. If two consecutive improvements in the same direction are followed by a worse result, the momentum resets. This prevents runaway exploitation of a local minimum while still accelerating along monotonic gradients.